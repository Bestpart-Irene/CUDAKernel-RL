"""Regression tests for the 2026-08-10 evaluator hardening (audit finding 2).

Covers the versioned-freeze / hack-channel fixes in eval_service/eval_core.py:
  - evaluator_sha: stable, well-formed, included in every eval result dict
  - speedup guard at source: non-finite or <=0 times emit None, never inf/NaN
  - B200 compute-capability parse ("100" -> major 10, not 1)
  - --use_fast_math dropped from default nvcc flags (opt-in via CU_FLAGS only)
  - anti-hack checks: an exception inside a check flags the candidate, never
    silently passes it (old `except Exception: pass` behavior)

Everything except the final integration test runs CPU-only without torch,
nvcc, or CUDA.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import types
from pathlib import Path

import pytest

from eval_service import eval_core


def _has_cuda_stack() -> bool:
    if shutil.which("nvcc") is None:
        return False
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


requires_cuda = pytest.mark.skipif(
    not _has_cuda_stack(), reason="needs torch + CUDA + nvcc"
)


# --- evaluator_sha: stability, format, definition ---------------------------

def test_evaluator_sha_stable_and_12_hex():
    s1 = eval_core.evaluator_sha()
    s2 = eval_core.evaluator_sha()
    assert s1 == s2
    assert len(s1) == 12
    int(s1, 16)  # must be valid hex


def test_evaluator_sha_matches_source_bytes():
    """The hash is sha256 over the raw bytes of eval_core.py + anti_hack.py."""
    import openenv_env.anti_hack as anti_hack

    digest = hashlib.sha256()
    for mod in (eval_core, anti_hack):
        digest.update(Path(mod.__file__).read_bytes())
    assert eval_core.evaluator_sha() == digest.hexdigest()[:12]


def test_evaluator_sha_in_wcc_result(monkeypatch):
    """WCC path: result dict carries evaluator_sha even on compile failure."""
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="fake nvcc error")

    stub = types.SimpleNamespace(run=fake_run, TimeoutExpired=subprocess.TimeoutExpired)
    monkeypatch.setattr(eval_core, "subprocess", stub)

    res = eval_core.evaluate_kernel_impl({"cuda_code": "__global__ void k() {}"})
    assert res["compiles"] is False
    assert res["evaluator_sha"] == eval_core.evaluator_sha()


def test_evaluator_sha_in_ops6k_result():
    """ops6k path: evaluator_sha + hack_suspected present from the first return."""
    res = eval_core.evaluate_ops6k_kernel_impl({})
    assert res["error"] == "Missing cuda_code or task_code"
    assert res["evaluator_sha"] == eval_core.evaluator_sha()
    assert res["hack_suspected"] is False


def test_evaluator_sha_in_batch_fallback(monkeypatch):
    """Batch path: the exception-fallback dict also carries evaluator_sha."""
    def boom(payload):
        raise RuntimeError("worker crashed")

    monkeypatch.setattr(eval_core, "evaluate_ops6k_kernel_impl", boom)
    results = eval_core.evaluate_kernels_batch_impl(
        [{"task_code": "x", "cuda_code": "y"}]
    )
    assert len(results) == 1
    assert results[0]["error"].startswith("Batch eval exception")
    assert results[0]["evaluator_sha"] == eval_core.evaluator_sha()


# --- speedup guard at source ------------------------------------------------

@pytest.mark.parametrize(
    "baseline,kernel",
    [
        (float("inf"), 1.0),
        (1.0, float("inf")),
        (float("nan"), 1.0),
        (1.0, float("nan")),
        (0.0, 1.0),
        (1.0, 0.0),
        (-1.0, 1.0),
        (1.0, -1.0),
        (None, 1.0),
        (1.0, None),
    ],
)
def test_guarded_speedup_invalid_times_emit_none(baseline, kernel):
    assert eval_core._guarded_speedup(baseline, kernel) is None


def test_guarded_speedup_valid_times():
    assert eval_core._guarded_speedup(10.0, 5.0) == pytest.approx(2.0)
    assert eval_core._guarded_speedup(1.0, 4.0) == pytest.approx(0.25)


# --- compute-capability parse (B200 fix) ------------------------------------

@pytest.mark.parametrize(
    "cc,expected",
    [
        ("80", (8, 0)),
        ("86", (8, 6)),
        ("90", (9, 0)),
        ("100", (10, 0)),  # B200 sm_100: old code parsed major=1
    ],
)
def test_parse_compute_capability(cc, expected):
    assert eval_core._parse_compute_capability(cc) == expected


# --- fast_math must be opt-in, never default --------------------------------

def test_nvcc_default_flags_have_no_fast_math():
    cmd = eval_core._nvcc_command("/tmp/a.cu", "/tmp/a.so", "__global__ void k() {}")
    assert "--use_fast_math" not in cmd
    # sanity: base flags survive
    assert "-O3" in cmd
    assert "--shared" in cmd


def test_nvcc_fast_math_still_available_via_cu_flags():
    code = "// CU_FLAGS: --use_fast_math\n__global__ void k() {}"
    cmd = eval_core._nvcc_command("/tmp/a.cu", "/tmp/a.so", code)
    assert "--use_fast_math" in cmd


# --- anti-hack: an exception inside a check is a failed check ---------------

def _fresh_result() -> dict:
    return {"correct": True, "hack_suspected": False, "error": "", "verifier_msg": ""}


def test_anti_hack_check_exception_flags_candidate(monkeypatch):
    """A crashing check must mark hack_suspected, never silently pass."""
    import openenv_env.anti_hack as anti_hack

    def raiser(*args, **kwargs):
        raise RuntimeError("kaboom in check")

    monkeypatch.setattr(anti_hack, "check_shapes_match", raiser)

    result = _fresh_result()
    ok = eval_core._run_anti_hack_checks(result, [object()], [object()], [[object()]])
    assert ok is False
    assert result["correct"] is False
    assert result["hack_suspected"] is True
    assert "kaboom in check" in result["error"]
    assert result["verifier_msg"] == result["error"]


def test_anti_hack_check_failure_flags_candidate(monkeypatch):
    """A check returning (False, reason) marks the candidate, with the reason."""
    import openenv_env.anti_hack as anti_hack

    monkeypatch.setattr(anti_hack, "check_shapes_match", lambda *a: (True, "ok"))
    monkeypatch.setattr(
        anti_hack,
        "check_output_not_constant",
        lambda *a: (False, "constant output detected"),
    )

    result = _fresh_result()
    ok = eval_core._run_anti_hack_checks(
        result, [object(), object()], [object()], [[object()]]
    )
    assert ok is False
    assert result["correct"] is False
    assert result["hack_suspected"] is True
    assert "constant output detected" in result["error"]


def test_anti_hack_all_checks_pass(monkeypatch):
    import openenv_env.anti_hack as anti_hack

    monkeypatch.setattr(anti_hack, "check_shapes_match", lambda *a: (True, "ok"))
    monkeypatch.setattr(anti_hack, "check_output_not_constant", lambda *a: (True, "ok"))
    monkeypatch.setattr(anti_hack, "check_not_passthrough", lambda *a: (True, "ok"))

    result = _fresh_result()
    ok = eval_core._run_anti_hack_checks(
        result, [object(), object()], [object()], [[object()]]
    )
    assert ok is True
    assert result["correct"] is True
    assert result["hack_suspected"] is False


# --- full-path integration (CUDA only) --------------------------------------

E1_RELU_TASK = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x):
        return torch.relu(x)

def get_inputs():
    return [torch.randn(4096)]

def get_init_inputs():
    return []
"""

E1_RELU_KERNEL = """
#include <cuda_runtime.h>

extern "C" __global__ void relu_kernel(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] > 0.0f ? in[i] : 0.0f;
}

extern "C" void run_kernel(const float* in, float* out, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(in, out, n);
    cudaDeviceSynchronize();
}
"""


@requires_cuda
def test_ops6k_full_path_honest_kernel():
    """Honest kernel survives isolation, re-verification, and the guards."""
    res = eval_core.evaluate_ops6k_kernel_impl(
        {
            "cuda_code": E1_RELU_KERNEL,
            "task_code": E1_RELU_TASK,
            "warmup_iters": 3,
            "benchmark_runs": 3,
        }
    )
    assert res["compiles"] is True
    assert res["correct"] is True, res["error"]
    assert res["hack_suspected"] is False
    assert res["evaluator_sha"] == eval_core.evaluator_sha()
    for key in ("speedup_vs_orig", "speedup_vs_dg"):
        val = res[key]
        if val is not None:
            assert val == val  # not NaN
            assert val != float("inf")
