"""
Eval dispatch abstraction — routes to local in-process, CoreWeave (HTTP), or Modal.

Set KERNELFORGE_EVAL_BACKEND to control dispatch:
  - "local": import eval_service.eval_core and call in-process on the current GPU.
    Use this when training and eval share one slurm allocation on an HPC cluster
    (e.g. Northeastern Explorer). Zero external cost.
  - "coreweave" (default for legacy): HTTP POST to KERNELFORGE_EVAL_URL.
  - "modal": modal.Function.from_name().remote() (requires Modal auth + budget).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

EVAL_BACKEND = os.getenv("KERNELFORGE_EVAL_BACKEND", "coreweave")
EVAL_URL = os.getenv("KERNELFORGE_EVAL_URL", "")
MODAL_APP_NAME = os.getenv("KERNELFORGE_MODAL_APP", "kernelforge-a100")

# Subprocess timeout (seconds) per single eval call. Caps user-kernel
# infinite loops and runaway compiles from blocking the trainer.
EVAL_SUBPROCESS_TIMEOUT = int(os.getenv("KERNELFORGE_EVAL_SUBPROCESS_TIMEOUT", "180"))

# Functions whose execution may corrupt CUDA context (run arbitrary user
# kernels). Always dispatched through `eval_service/worker_main.py`
# subprocess so a CUDA fault kills only the worker, not the trainer.
# Diagnosed 2026-05-26 EXP-014-retry: in-process ops6k eval triggered
# `cudaErrorIllegalAddress` and killed training at step 4.
_ISOLATED_FNS = {"evaluate_kernel", "evaluate_ops6k_kernel", "evaluate_kernels_batch"}


def dispatch_eval(fn_name: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Dispatch an evaluation call to the configured backend.

    Args:
        fn_name: Evaluation function name (e.g. "evaluate_kernel", "profile_baselines").
        payload: JSON-serializable payload for the evaluation function.

    Returns:
        Evaluation result dict.
    """
    if EVAL_BACKEND == "local":
        return _dispatch_local(fn_name, payload)
    if EVAL_BACKEND == "modal":
        return _dispatch_modal(fn_name, payload)
    return _dispatch_http(fn_name, payload)


def _dispatch_http(fn_name: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    """Dispatch via HTTP POST to CoreWeave/Northflank eval service."""
    import httpx

    if not EVAL_URL:
        raise RuntimeError(
            "KERNELFORGE_EVAL_URL must be set when KERNELFORGE_EVAL_BACKEND=coreweave. "
            "Set it to the Northflank eval service URL (e.g. https://eval-kernelforge.northflank.app)."
        )
    url = f"{EVAL_URL.rstrip('/')}/{fn_name}"
    resp = httpx.post(url, json=payload or {}, timeout=300.0)
    resp.raise_for_status()
    return resp.json()


def _dispatch_modal(fn_name: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    """Dispatch via Modal serverless function."""
    import modal

    fn = modal.Function.from_name(MODAL_APP_NAME, fn_name)
    if payload is None:
        return fn.remote()
    return fn.remote(payload)


def _safe_local_failure(reason: str) -> dict[str, Any]:
    return {
        "compiles": False,
        "correct": False,
        "error": f"Local eval error: {reason[:1000]}",
        "runtime_ms": 0.0,
        "runtime_stats": {},
        "speedup_vs_orig": 0.0,
        "speedup_vs_dg": 0.0,
    }


def _dispatch_local_subprocess(
    fn_name: str, payload: dict[str, Any] | list[dict[str, Any]] | None
) -> dict[str, Any]:
    """Run a single eval call in an isolated subprocess via worker_main.

    If the subprocess crashes (CUDA fault, OOM, etc.) the parent reads
    the non-zero return code and returns a safe failure dict; training
    continues unaffected. Timeout caps runaway kernels.
    """
    req = json.dumps({"fn_name": fn_name, "payload": payload})
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "eval_service.worker_main"],
            input=req,
            capture_output=True,
            text=True,
            timeout=EVAL_SUBPROCESS_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return _safe_local_failure(
            f"subprocess timeout ({EVAL_SUBPROCESS_TIMEOUT}s)"
        )
    except FileNotFoundError as exc:
        return _safe_local_failure(f"subprocess spawn failed: {exc}")

    if proc.returncode != 0:
        # Subprocess died (likely CUDA fault) — stdout JSON may be missing.
        return _safe_local_failure(
            f"subprocess exit {proc.returncode}: "
            f"stderr={proc.stderr[:400]} stdout={proc.stdout[:200]}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return _safe_local_failure(
            f"subprocess produced non-JSON stdout: {proc.stdout[:500]}"
        )


def _dispatch_local(
    fn_name: str, payload: dict[str, Any] | list[dict[str, Any]] | None
) -> dict[str, Any]:
    """Dispatch in-process on the current GPU. Mirrors eval_service/app.py routing.

    This is the zero-external-cost path: training and eval run inside the same
    slurm allocation. CUDA-execution functions (`evaluate_kernel`,
    `evaluate_ops6k_kernel`, `evaluate_kernels_batch`) are routed through an
    isolated subprocess (see `eval_service/worker_main.py`) so user-kernel
    CUDA faults cannot corrupt the trainer's CUDA context. Diagnostic
    functions (`profile_baselines`, `test_gpu_features`) stay in-process —
    they only call internal CUDA paths we trust.

    Requires a CUDA-capable GPU on the current node (nvcc + torch.cuda.is_available()).
    """
    if fn_name in _ISOLATED_FNS:
        return _dispatch_local_subprocess(fn_name, payload)

    # In-process path for trusted diagnostic functions only.
    from eval_service.eval_core import (
        profile_baselines_impl,
        test_gpu_features_impl,
    )

    dispatch_table = {
        "profile_baselines": lambda _p: profile_baselines_impl(),
        "test_gpu_features": lambda _p: test_gpu_features_impl(),
    }
    if fn_name not in dispatch_table:
        raise ValueError(
            f"Unknown eval fn_name: {fn_name!r}. "
            f"Valid: {sorted(dispatch_table) + sorted(_ISOLATED_FNS)}"
        )

    try:
        return dispatch_table[fn_name](payload)
    except Exception as exc:
        # Mirror eval_service/app.py:global_exception_handler so caller code
        # behaves the same regardless of backend.
        return _safe_local_failure(str(exc))
