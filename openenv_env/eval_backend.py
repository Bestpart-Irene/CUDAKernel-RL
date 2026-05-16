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

import os
from typing import Any

EVAL_BACKEND = os.getenv("KERNELFORGE_EVAL_BACKEND", "coreweave")
EVAL_URL = os.getenv("KERNELFORGE_EVAL_URL", "")
MODAL_APP_NAME = os.getenv("KERNELFORGE_MODAL_APP", "kernelforge-a100")


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


def _dispatch_local(
    fn_name: str, payload: dict[str, Any] | list[dict[str, Any]] | None
) -> dict[str, Any]:
    """Dispatch in-process on the current GPU. Mirrors eval_service/app.py routing.

    This is the zero-external-cost path: training and eval run inside the same
    slurm allocation, so eval_core is imported and called directly. Requires
    a CUDA-capable GPU on the current node (nvcc + torch.cuda.is_available()).
    """
    from eval_service.eval_core import (
        evaluate_kernel_impl,
        evaluate_kernels_batch_impl,
        evaluate_ops6k_kernel_impl,
        profile_baselines_impl,
        test_gpu_features_impl,
    )

    dispatch_table = {
        "evaluate_kernel": lambda p: evaluate_kernel_impl(p or {}),
        "evaluate_ops6k_kernel": lambda p: evaluate_ops6k_kernel_impl(p or {}),
        "evaluate_kernels_batch": lambda p: evaluate_kernels_batch_impl(p or []),
        "profile_baselines": lambda _p: profile_baselines_impl(),
        "test_gpu_features": lambda _p: test_gpu_features_impl(),
    }
    if fn_name not in dispatch_table:
        raise ValueError(
            f"Unknown eval fn_name: {fn_name!r}. Valid: {sorted(dispatch_table)}"
        )

    try:
        return dispatch_table[fn_name](payload)
    except Exception as exc:
        # Mirror eval_service/app.py:global_exception_handler so caller code
        # behaves the same regardless of backend.
        return {
            "compiles": False,
            "correct": False,
            "error": f"Local eval error: {str(exc)[:1000]}",
            "runtime_ms": 0.0,
            "runtime_stats": {},
            "speedup_vs_orig": 0.0,
            "speedup_vs_dg": 0.0,
        }
