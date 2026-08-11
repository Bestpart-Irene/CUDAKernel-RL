"""2026-08-10 audit finding-7 regression net: signature-class consistency.

The extern "C" contract shown to the policy (training/task_support.py:
infer_signature_class -> ops6k_contract_text) and the ctypes signature the
evaluator actually dlsym-invokes (eval_service.eval_core.
_infer_extern_c_signature -> _ctypes_argtypes) are inferred INDEPENDENTLY
from the same task_code. If they ever diverge, the policy is trained
against a contract the evaluator does not enforce — every rollout for that
task fails at invocation for a task-side reason, silently pinned at
reward=-1 (the EXP-018d contract-contradiction family).

This suite asserts, for EVERY live supported ops6k row in
datasets/combined_kernelforge.jsonl, that the training-side inferred class
equals the eval-side inferred class (eval_service imported read-only).

Torch-unavailable hosts (e.g. the local macOS control plane) cannot run
the eval-side inference (it execs task_code with torch), so there the
suite still checks the static AST fallback agreement where that fallback
is exact — rows whose get_inputs() return literal contains only
constructor calls, where element count == tensor count — and skips the
torch-dependent comparisons.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from training.task_support import (
    _infer_signature_class_static,
    infer_signature_class,
    normalize_task_row,
)

try:  # pragma: no cover - environment probe
    import torch  # noqa: F401
    HAS_TORCH = True
except Exception:  # pragma: no cover
    HAS_TORCH = False

DATASET_PATH = Path(__file__).resolve().parents[1] / "datasets" / "combined_kernelforge.jsonl"


def _live_ops6k_rows() -> list[dict]:
    rows = []
    with open(DATASET_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            norm = normalize_task_row(json.loads(line))
            if norm["evaluation_backend"] == "ops6k":
                rows.append(norm)
    return rows


LIVE_ROWS = _live_ops6k_rows()


def _row_label(row: dict) -> str:
    return f"task_id={row.get('task_id')!r} ops={row.get('ops')!r}"


def _static_count_is_exact(task_code: str) -> bool:
    """True when get_inputs()'s return literal holds only constructor calls.

    The static fallback counts *elements* of the return literal while the
    dynamic path counts *tensors*; they agree exactly only when every
    element is a call (e.g. torch.randn(...)) rather than a scalar literal
    (e.g. `return [torch.randn(...), 0]` — the torch.cumsum row).
    """
    try:
        tree = ast.parse(task_code)
    except SyntaxError:
        return False
    fn = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "get_inputs"),
        None,
    )
    if fn is None:
        return False
    returns = [n for n in ast.walk(fn)
               if isinstance(n, ast.Return) and n.value is not None]
    if not returns:
        return False
    value = returns[-1].value
    elts = value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value]
    return all(isinstance(e, ast.Call) for e in elts)


def test_dataset_has_live_ops6k_rows():
    """Guard: an empty live pool would make every other test vacuous."""
    assert len(LIVE_ROWS) > 0, f"no live supported ops6k rows in {DATASET_PATH}"


@pytest.mark.skipif(not HAS_TORCH, reason="eval-side inference execs task_code with torch")
def test_train_eval_signature_agreement():
    """THE finding-7 net: training-side class == eval-side class, per live row.

    None == None is a legitimate agreement (both sides refuse the task);
    a None-vs-class split is exactly the divergence this test exists to catch.
    """
    from eval_service.eval_core import _infer_extern_c_signature

    mismatches = []
    for row in LIVE_ROWS:
        task_code = row.get("task_code") or ""
        train_class = infer_signature_class(task_code)
        eval_sig = _infer_extern_c_signature(task_code)
        eval_class = (eval_sig or {}).get("class")
        if train_class != eval_class:
            mismatches.append(
                f"{_row_label(row)}: train={train_class!r} eval={eval_class!r}"
            )
    assert not mismatches, (
        "training-side and eval-side extern-C signature inference diverge "
        "(policy would be prompted for one contract and invoked under "
        "another):\n  " + "\n  ".join(mismatches)
    )


def test_static_fallback_agreement_where_exact():
    """Static AST fallback must match full inference on statically-exact rows.

    Runs with or without torch. With torch, this compares the dynamic
    (exec-based) result against the AST fallback; without torch both take
    the static path, which still guards the fallback against crashes and
    non-contract values. Rows whose return literal mixes scalars into the
    list (element count != tensor count) are excluded — the fallback is
    documented best-effort there.
    """
    checked = 0
    for row in LIVE_ROWS:
        task_code = row.get("task_code") or ""
        static_class = _infer_signature_class_static(task_code)
        assert static_class in {None, "E1", "E2"}, _row_label(row)
        if not _static_count_is_exact(task_code):
            continue
        checked += 1
        full_class = infer_signature_class(task_code)
        assert static_class == full_class, (
            f"{_row_label(row)}: static fallback {static_class!r} != "
            f"full inference {full_class!r} on a statically-exact row"
        )
    assert checked > 0, "no statically-exact live rows — helper is broken"


def test_spike_pool_probe_task_signatures():
    """EXP-018c-p0 spike-pool tasks must keep their known signature classes.

    Static-only, so this runs on torch-free hosts too. If a dataset worker
    removes or reshapes one of these rows, the probe's substitution logic
    fires — this test pins the expectation so the change is loud.
    """
    expectations = {
        "vector_add_e2": "E2",
        "F.elu": "E1",
        "F.softplus": "E1",
    }

    va = next((r for r in LIVE_ROWS if str(r.get("task_id") or "") == "vector_add_e2"), None)
    if va is None:
        pytest.skip("vector_add_e2 row absent — probe will substitute and flag")
    assert _infer_signature_class_static(va.get("task_code") or "") == expectations["vector_add_e2"]

    for op in ("F.elu", "F.softplus"):
        row = next((r for r in LIVE_ROWS if r.get("ops") == [op]), None)
        if row is None:
            pytest.skip(f"single-op {op} row absent — probe will substitute and flag")
        assert _infer_signature_class_static(row.get("task_code") or "") == expectations[op], op
