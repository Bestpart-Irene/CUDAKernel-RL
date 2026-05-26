"""Subprocess worker for eval calls — sandboxes user CUDA code.

When `KERNELFORGE_EVAL_BACKEND=local`, `openenv_env/eval_backend.py:_dispatch_local`
runs user-generated CUDA code in-process on the training GPU. If a user kernel
triggers an illegal memory access (out-of-bounds, race condition, NaN-driven
indexing, etc.) the CUDA context becomes globally corrupted: all subsequent
torch operations in the training process — including reward-tensor creation —
fail with `cudaErrorIllegalAddress` and the training job dies. Diagnosed
2026-05-26 on EXP-014-retry job 7026132 (62% compile_OK_wrong rollouts
finally executed user code, one of them killed the trainer at step 4).

This worker module runs ONE eval call per subprocess invocation, then exits.
A CUDA fault inside the worker kills only the worker; the parent trainer
continues. The parent reads the result via stdout JSON.

Used by `openenv_env/eval_backend.py:_dispatch_local_subprocess` for
fn_name in {"evaluate_kernel", "evaluate_ops6k_kernel"}.

Protocol:
  stdin:  {"fn_name": "evaluate_ops6k_kernel", "payload": {...}}
  stdout: {"compiles": ..., "correct": ..., "error": ..., ...}

On any exception, prints a safe failure JSON and exits 0 so the parent
can always parse stdout. Hard CUDA crashes will kill the worker before
the JSON is written — parent must handle this via subprocess return code.
"""
from __future__ import annotations

import json
import sys
import traceback


def _safe_failure(reason: str) -> dict:
    return {
        "compiles": False,
        "correct": False,
        "error": f"worker: {reason}"[:2000],
        "runtime_ms": 0.0,
        "runtime_stats": {},
        "speedup_vs_orig": 0.0,
        "speedup_vs_dg": 0.0,
    }


def main() -> None:
    try:
        req = json.loads(sys.stdin.read())
        fn_name = req["fn_name"]
        payload = req.get("payload")
    except Exception as exc:
        print(json.dumps(_safe_failure(f"invalid stdin: {exc}")), flush=True)
        return

    try:
        from eval_service.eval_core import (
            evaluate_kernel_impl,
            evaluate_kernels_batch_impl,
            evaluate_ops6k_kernel_impl,
        )
    except Exception as exc:
        print(json.dumps(_safe_failure(f"import eval_core failed: {exc}")), flush=True)
        return

    table = {
        "evaluate_kernel": evaluate_kernel_impl,
        "evaluate_ops6k_kernel": evaluate_ops6k_kernel_impl,
        "evaluate_kernels_batch": evaluate_kernels_batch_impl,
    }
    if fn_name not in table:
        print(json.dumps(_safe_failure(f"unknown fn_name: {fn_name!r}")), flush=True)
        return

    try:
        # evaluate_kernels_batch expects a list payload, others expect dict
        if fn_name == "evaluate_kernels_batch":
            result = table[fn_name](payload or [])
        else:
            result = table[fn_name](payload or {})
        print(json.dumps(result), flush=True)
    except Exception:
        print(
            json.dumps(_safe_failure(f"eval raised: {traceback.format_exc()[:1500]}")),
            flush=True,
        )


if __name__ == "__main__":
    main()
