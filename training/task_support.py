"""Task routing helpers for the KernelForge RL pipeline.

This module keeps task-family decisions in one place so that dataset loading,
rollout generation, offline evaluation, and OpenEnv all agree on what can be
evaluated live today.
"""

from __future__ import annotations

import ast
import json
from collections import Counter
from typing import Any

SUPPORTED_WCC_OPS = {"weakly_connected_components", "wcc"}
STATEFUL_MODULE_TOKENS = (
    "nn.Conv",
    "nn.Linear",
    "nn.BatchNorm",
    "nn.LSTM",
    "nn.GRU",
    "nn.Embedding",
    "nn.Parameter",
    "nn.MultiheadAttention",
    "register_buffer(",
    "register_parameter(",
)


def parse_ops(raw_ops: Any) -> list[str]:
    """Best-effort parser for serialized operator lists."""
    if isinstance(raw_ops, list):
        return [str(item) for item in raw_ops]

    if isinstance(raw_ops, str):
        text = raw_ops.strip()
        if not text:
            return []

        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
            except Exception:
                continue
            if isinstance(parsed, list):
                return [str(item) for item in parsed]

        return [text]

    return [str(raw_ops)] if raw_ops is not None else []


def _task_has_empty_init_inputs(task_code: str) -> bool:
    """Return True when get_init_inputs is absent or trivially empty."""
    if not task_code.strip():
        return False

    try:
        tree = ast.parse(task_code)
    except SyntaxError:
        return False

    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name != "get_init_inputs":
            continue

        returns = [sub.value for sub in ast.walk(node) if isinstance(sub, ast.Return)]
        if not returns:
            return True

        value = returns[-1]
        if isinstance(value, (ast.List, ast.Tuple)) and len(value.elts) == 0:
            return True
        if isinstance(value, ast.Constant) and value.value is None:
            return True
        return False

    return True


def supports_ops6k_live_eval(task_code: str) -> bool:
    """Return True for the stateless subset we can evaluate end to end today."""
    text = task_code.strip()
    if not text:
        return False

    if any(token in text for token in STATEFUL_MODULE_TOKENS):
        return False

    return _task_has_empty_init_inputs(text)


def infer_evaluation_backend(row: dict[str, Any]) -> str:
    """Infer which evaluator can score a row live."""
    # Owner decision 2026-08-10: explicit dataset-level unsupported labels are
    # sticky. Curated-out rows (e.g. 3+-tensor ops6k rows with no extern-C
    # signature class — E3 deferred, see EXP-018d) must not be re-included by
    # recomputation. Only the unsupported direction is sticky; claimed
    # "ops6k"/"wcc" labels are still re-verified below.
    if row.get("supports_evaluation") is False or row.get("evaluation_backend") == "unsupported":
        return "unsupported"
    task_code = str(row.get("task_code") or "").strip()
    if task_code:
        if not supports_ops6k_live_eval(task_code):
            return "unsupported"
        # The evaluator only has E1/E2 extern-C signatures; a row it cannot
        # infer a signature for is permanently dead (EXP-018d open item), so
        # a dataset rebuild must never resurrect signature-less rows.
        return "ops6k" if infer_signature_class(task_code) is not None else "unsupported"

    ops = {op.lower() for op in parse_ops(row.get("ops"))}
    prompt = str(row.get("prompt", "")).lower()
    kernel_id = str(row.get("kernel_id", "")).lower()

    if (
        ops & SUPPORTED_WCC_OPS
        or "weakly connected" in prompt
        or "weakly_connected_components" in kernel_id
    ):
        return "wcc"

    return "unsupported"


def support_reason(row: dict[str, Any]) -> str:
    """Return a short explanation describing evaluator support status."""
    backend = infer_evaluation_backend(row)
    if backend == "wcc":
        return "routed to the WCC verifier/benchmark path"
    if backend == "ops6k":
        return "routed to the stateless Ops-6K extension harness"
    if row.get("task_code"):
        return (
            "task requires stateful module or non-empty init inputs; "
            "the current Ops-6K harness only supports stateless tasks"
        )
    return "graph task has no correctness/runtime harness beyond WCC today"


def normalize_task_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize task metadata so every pipeline stage sees the same schema."""
    normalized = dict(row)
    normalized["prompt"] = str(normalized.get("prompt", "")).strip()
    normalized["ops"] = parse_ops(normalized.get("ops"))
    normalized["task_code"] = (
        str(normalized.get("task_code")).strip() if normalized.get("task_code") else None
    )
    # EXP-018a: preserve extern_c_signature dict if present. The eval path uses
    # it to pick the right ctypes argtypes; absent means evaluator falls back
    # to inferring from task_code's get_inputs() tensor count.
    sig = normalized.get("extern_c_signature")
    normalized["extern_c_signature"] = sig if isinstance(sig, dict) else None
    backend = infer_evaluation_backend(normalized)
    normalized["evaluation_backend"] = backend
    normalized["supports_evaluation"] = backend in {"wcc", "ops6k"}
    normalized["support_reason"] = support_reason(normalized)
    return normalized


def filter_supported_tasks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only tasks that have a live evaluator."""
    supported: list[dict[str, Any]] = []
    for row in rows:
        normalized = normalize_task_row(row)
        if normalized["prompt"] and normalized["supports_evaluation"]:
            supported.append(normalized)
    return supported


def build_prompt_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build a prompt -> metadata lookup for rollout/eval paths."""
    lookup: dict[str, dict[str, Any]] = {}
    for row in rows:
        normalized = normalize_task_row(row)
        if normalized["prompt"]:
            lookup[normalized["prompt"]] = normalized
    return lookup


def infer_signature_class(task_code: str) -> str | None:
    """Infer the extern-C signature class the evaluator will use for a task.

    Mirrors ``eval_service.eval_core._infer_extern_c_signature`` (1 input
    tensor -> E1, 2 -> E2, anything else -> None). Falls back to AST analysis
    of ``get_inputs()``'s return statement when torch is unavailable or the
    task code cannot execute on this host (e.g. CPU-only box, `.cuda()` inputs).
    """
    if not task_code:
        return None
    try:
        import torch
    except Exception:
        return _infer_signature_class_static(task_code)
    try:
        ns: dict[str, Any] = {}
        exec(task_code, ns)  # trusted dataset code; the evaluator execs it too
        get_inputs = ns.get("get_inputs")
        if get_inputs is None:
            return None
        sample = get_inputs()
        if not isinstance(sample, (list, tuple)):
            sample = [sample]
        count = sum(1 for x in sample if isinstance(x, torch.Tensor))
    except Exception:
        return _infer_signature_class_static(task_code)
    return {1: "E1", 2: "E2"}.get(count)


def _infer_signature_class_static(task_code: str) -> str | None:
    """Torch-free fallback: count elements of get_inputs()'s return literal."""
    try:
        tree = ast.parse(task_code)
    except SyntaxError:
        return None
    fn = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "get_inputs"
        ),
        None,
    )
    if fn is None:
        return None
    returns = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Return) and node.value is not None
    ]
    if not returns:
        return None
    value = returns[-1].value
    count = len(value.elts) if isinstance(value, (ast.List, ast.Tuple)) else 1
    return {1: "E1", 2: "E2"}.get(count)


def ops6k_contract_text(sig_class: str) -> str:
    """Single source of truth for the EXP-018a unified extern "C" contract."""
    if sig_class == "E2":
        sig_line = (
            'extern "C" void run_kernel(const float* x, const float* y, '
            "float* out, int n)"
        )
        arg_desc = (
            "  `x` and `y` are device pointers to the two input tensors flattened to 1D.\n"
            "  `out` is a pre-allocated device pointer for the output (same total element "
            "count as the reference output).\n"
            "  `n` is the total element count."
        )
    else:  # E1 default
        sig_line = 'extern "C" void run_kernel(const float* x, float* out, int n)'
        arg_desc = (
            "  `x` is a device pointer to the input tensor flattened to 1D.\n"
            "  `out` is a pre-allocated device pointer for the output (same total element "
            "count as the reference output).\n"
            "  `n` is the total element count."
        )
    return (
        "Evaluation contract (EXP-018a unified extern \"C\"):\n"
        "- Produce a single raw CUDA source file. Do NOT include "
        "`<torch/extension.h>`, `<ATen/...>`, or `<c10/...>`. Do NOT "
        "reference `torch::`, `at::`, or `c10::` in any form.\n"
        f"- Export `{sig_line}`.\n"
        f"{arg_desc}\n"
        "- The function must launch your `__global__` CUDA kernel and "
        "cudaDeviceSynchronize before returning.\n"
        "- Output values must match the reference `Model(*get_inputs())` "
        "result element-wise to within rtol=1e-3, atol=1e-3.\n"
        "- Return CUDA/C++ code only.\n"
        "- If extra nvcc flags are required, add a `// CU_FLAGS:` comment."
    )


def task_interface_contract(row: dict[str, Any]) -> str:
    """Return the evaluator contract that the model must satisfy."""
    backend = normalize_task_row(row)["evaluation_backend"]
    if backend == "wcc":
        return (
            "Evaluation contract:\n"
            '- Export `extern "C" void wcc_kernel(const int* row_ptr, const int* col_idx, '
            "int num_vertices, int* labels)`.\n"
            "- Fill `labels` in-place with component identifiers.\n"
            "- Return CUDA/C++ code only.\n"
            "- If extra nvcc flags are required, add a `// CU_FLAGS:` comment."
        )
    if backend == "ops6k":
        sig = (row.get("extern_c_signature") or {}).get("class")
        if not sig:
            # EXP-018d: the dataset no longer carries extern_c_signature
            # (dropped in 51eb7c9), so infer the class the evaluator itself
            # will infer from get_inputs(); E1 only as a last resort.
            sig = infer_signature_class(row.get("task_code") or "") or "E1"
        return ops6k_contract_text(sig)
    return (
        "This task currently has no live evaluator in KernelForge. "
        "Do not use it for RL training until a correctness/runtime harness exists."
    )


def build_generation_prompt(
    task_row: dict[str, Any],
    skill_context: str = "",
    topology_context: str = "",
) -> str:
    """Compose the actual prompt shown to the policy for a task row."""
    row = normalize_task_row(task_row)
    contract = task_interface_contract(row)
    if contract and contract in row["prompt"]:
        # EXP-018d dataset prompts already embed the contract — don't repeat it.
        contract = ""
    parts = [
        skill_context.strip(),
        row["prompt"],
        topology_context.strip(),
        contract,
    ]
    return "\n\n---\n\n".join(part for part in parts if part)


def build_modal_payload(
    cuda_code: str,
    task_row: dict[str, Any],
    baseline_orig_ms: float | None = None,
    baseline_dg_ms: float | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build the Modal function name + payload for a task row."""
    row = normalize_task_row(task_row)
    backend = row["evaluation_backend"]

    if backend == "ops6k":
        return (
            "evaluate_ops6k_kernel",
            {
                "cuda_code": cuda_code,
                "task_code": row.get("task_code") or "",
                "warmup_iters": 10,
                "benchmark_runs": 10,
                "evaluation_backend": backend,
                "extern_c_signature": row.get("extern_c_signature"),
            },
        )

    if backend == "wcc":
        return (
            "evaluate_kernel",
            {
                "cuda_code": cuda_code,
                "verify_graphs": 5,
                "warmup_iters": 50,
                "benchmark_runs": 30,
                "baseline_original_ms": baseline_orig_ms,
                "baseline_doublegraph_ms": baseline_dg_ms,
                "evaluation_backend": backend,
            },
        )

    raise ValueError(row["support_reason"])


def normalize_eval_result(result: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize evaluator output into the reward contract."""
    out = dict(result or {})
    out.setdefault("compiles", False)
    out.setdefault("correct", False)
    out.setdefault("speedup_vs_orig", 0.0)
    out.setdefault("speedup_vs_dg", 0.0)
    out.setdefault("runtime_ms", 0.0)
    out.setdefault("runtime_stats", {})
    out.setdefault("verifier_msg", "")
    out.setdefault("error", "")
    return out


def _detect_symbol_loaded(result: dict[str, Any]) -> bool:
    """Return False iff the verifier failed to dlsym the canonical entry symbol.

    The eval service does not expose a dedicated "symbol_missing" flag, so we
    substring-match the verifier message. Pattern is stable across the WCC
    verifier (`verification/pac_verify.py:198`): `"FFI verification failed"`
    AND `"undefined symbol"`. EXP-010 — gives v3-symbol-shaped reward a
    directional signal toward emitting `extern "C" void wcc_kernel(...)`.
    """
    if not result.get("compiles"):
        return True  # n/a when compile failed; v3 short-circuits on compiled first
    msg = str(result.get("verifier_msg", "") or "")
    if not msg:
        return True
    return not ("undefined symbol" in msg and "FFI verification failed" in msg)


def compute_task_reward(result: dict[str, Any] | None) -> float:
    """Compute the canonical reward from a normalized evaluator result."""
    from openenv_env.reward import compute_reward

    normalized = normalize_eval_result(result)
    return compute_reward(
        compiled=bool(normalized.get("compiles")),
        correct=bool(normalized.get("correct")),
        speedup_vs_eager=float(normalized.get("speedup_vs_orig") or 0.0),
        speedup_vs_compile=float(normalized.get("speedup_vs_dg") or 0.0),
        occupancy=normalized.get("occupancy"),
        mem_coalescing=normalized.get("mem_coalescing"),
        warp_efficiency=normalized.get("warp_efficiency"),
        symbol_loaded=_detect_symbol_loaded(normalized),
    )


def evaluate_code_remote(
    cuda_code: str,
    task_row: dict[str, Any],
    baseline_orig_ms: float | None = None,
    baseline_dg_ms: float | None = None,
) -> dict[str, Any]:
    """Dispatch a candidate kernel to the configured eval backend (CoreWeave or Modal)."""
    from openenv_env.eval_backend import dispatch_eval

    fn_name, payload = build_modal_payload(
        cuda_code,
        task_row,
        baseline_orig_ms=baseline_orig_ms,
        baseline_dg_ms=baseline_dg_ms,
    )
    result = normalize_eval_result(dispatch_eval(fn_name, payload))
    result["reward"] = compute_task_reward(result)
    result["evaluation_backend"] = normalize_task_row(task_row)["evaluation_backend"]
    return result


# Backward-compatible alias
def evaluate_code_on_modal(
    cuda_code: str,
    task_row: dict[str, Any],
    modal_app_name: str = "",
    baseline_orig_ms: float | None = None,
    baseline_dg_ms: float | None = None,
) -> dict[str, Any]:
    """Backward-compatible alias — dispatches via eval_backend (ignores modal_app_name)."""
    return evaluate_code_remote(
        cuda_code, task_row,
        baseline_orig_ms=baseline_orig_ms,
        baseline_dg_ms=baseline_dg_ms,
    )


def summarize_tasks(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Summarize task coverage for dataset and preflight reporting."""
    normalized = [normalize_task_row(row) for row in rows if row.get("prompt")]
    backend_counts = Counter(row["evaluation_backend"] for row in normalized)
    source_counts = Counter(str(row.get("data_source", "unknown")) for row in normalized)
    difficulty_counts = Counter(int(row.get("difficulty", 1)) for row in normalized)
    supported_counts = Counter("supported" if row["supports_evaluation"] else "unsupported" for row in normalized)
    return {
        "evaluation_backend": dict(sorted(backend_counts.items())),
        "data_source": dict(sorted(source_counts.items())),
        "difficulty": dict(sorted(difficulty_counts.items())),
        "support": dict(sorted(supported_counts.items())),
    }
