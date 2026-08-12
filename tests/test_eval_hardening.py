"""Regression tests for the 2026-08-10 evaluator hardening (audit finding 2).

Covers the versioned-freeze / hack-channel fixes in eval_service/eval_core.py:
  - evaluator_sha: stable, well-formed, included in every eval result dict
  - speedup guard at source: non-finite or <=0 times emit None, never inf/NaN
  - B200 compute-capability parse ("100" -> major 10, not 1)
  - --use_fast_math dropped from default nvcc flags (opt-in via CU_FLAGS only)
  - anti-hack checks: an exception inside a check flags the candidate, never
    silently passes it (old `except Exception: pass` behavior)
  - constant-output probe (2026-08-11): when a task pins its RNG inside
    get_inputs(), the second invocation feeding check_output_not_constant
    must receive genuinely different inputs (E1 and E2; E2 order x,y,out,n)
    — the EXP-018c-p0 vector_add_e2 false-reject regression

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


# --- constant-output probe: E2 second-invocation regression ------------------
#
# EXP-018c-p0 evidence: 14/14 PERFECT two-input vector_add candidates were
# rejected with "Output is constant across different inputs". Root cause: the
# task's get_inputs() pins its own RNG (torch.manual_seed(0) inside), so the
# harness's manual_seed(42+seed) reseeding never varied the inputs — both
# anti-hack invocations saw bit-identical (x, y) and a correct kernel's
# bit-identical outputs tripped check_output_not_constant. These tests drive
# the fixed path (_apply_constant_probe_if_inputs_identical) with CPU
# numpy-backed fake tensors and mock run_kernel functions that read their
# argument buffers, asserting that DIFFERENT inputs actually reach the
# candidate's second invocation — for E1 and E2 — and that E2 arg order is
# (x, y, out, n). Genuine decoys must still be caught.

import ctypes as _ctypes

import numpy as np


class FakeTensor:
    """Minimal CPU stand-in for the tensor protocol eval_core relies on."""

    is_cuda = False

    def __init__(self, arr):
        self._arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))

    @property
    def shape(self):
        return tuple(self._arr.shape)

    @property
    def dtype(self):
        return self._arr.dtype

    def data_ptr(self):
        return self._arr.ctypes.data

    def numel(self):
        return int(self._arr.size)

    def contiguous(self):
        return self

    def detach(self):
        return self

    def clone(self):
        return FakeTensor(self._arr.copy())

    def cpu(self):
        return self

    def __add__(self, other):
        return FakeTensor(self._arr + np.float32(other))

    def __array__(self, dtype=None):
        return self._arr if dtype is None else self._arr.astype(dtype)


def _buf_at(ptr: int, n: int) -> np.ndarray:
    """View n float32s at a raw address (the args a real kernel would see)."""
    return np.ctypeslib.as_array((_ctypes.c_float * n).from_address(ptr))


def test_invoke_candidate_e1_arg_order():
    x = FakeTensor(np.arange(8))
    out = FakeTensor(np.zeros(8))
    calls = []
    eval_core._invoke_candidate(lambda *a: calls.append(a), "E1", [x], out)
    assert len(calls) == 1
    assert calls[0] == (x.data_ptr(), out.data_ptr(), 8)


def test_invoke_candidate_e2_arg_order():
    """E2 contract: run_kernel(x, y, out, n) — exactly this order."""
    x = FakeTensor(np.arange(8))
    y = FakeTensor(np.arange(8) * 2)
    out = FakeTensor(np.zeros(8))
    calls = []
    eval_core._invoke_candidate(lambda *a: calls.append(a), "E2", [x, y], out)
    assert len(calls) == 1
    assert calls[0] == (x.data_ptr(), y.data_ptr(), out.data_ptr(), 8)


def test_tensor_lists_identical():
    a = FakeTensor([1.0, 2.0, 3.0])
    same = FakeTensor([1.0, 2.0, 3.0])
    other = FakeTensor([1.0, 2.0, 4.0])
    assert eval_core._tensor_lists_identical([a], [same]) is True
    assert eval_core._tensor_lists_identical([a, a], [same, same]) is True
    assert eval_core._tensor_lists_identical([a], [other]) is False
    assert eval_core._tensor_lists_identical([a, a], [same, other]) is False
    assert eval_core._tensor_lists_identical([a], [a, a]) is False  # length
    assert eval_core._tensor_lists_identical([], []) is False  # nothing to probe


def _pinned_e2_setup():
    """Simulate a seed-pinning E2 task: both anti-hack samples saw the SAME
    (x, y), and the (honest) kernel produced identical x+y outputs."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(16).astype(np.float32)
    y = rng.standard_normal(16).astype(np.float32)
    out0 = x + y
    candidate_outputs = [FakeTensor(out0), FakeTensor(out0.copy())]
    inputs_list = [
        [FakeTensor(x), FakeTensor(y)],
        [FakeTensor(x.copy()), FakeTensor(y.copy())],
    ]
    base_gpu_inputs = [FakeTensor(x), FakeTensor(y)]
    out_template = FakeTensor(out0)
    return x, y, candidate_outputs, inputs_list, base_gpu_inputs, out_template


def test_constant_probe_e2_different_inputs_reach_candidate():
    """The probe's second invocation must hand the candidate GENUINELY
    different inputs, in (x, y, out, n) order — the EXP-018c-p0 regression."""
    x, y, candidate_outputs, inputs_list, base_inputs, out_template = _pinned_e2_setup()

    seen = {}

    def honest_add(x_ptr, y_ptr, out_ptr, n):
        px = _buf_at(x_ptr, n)
        py = _buf_at(y_ptr, n)
        seen["x"] = px.copy()
        seen["y"] = py.copy()
        seen["n"] = n
        seen["ptrs"] = (x_ptr, y_ptr, out_ptr)
        _buf_at(out_ptr, n)[:] = px + py

    probed = eval_core._apply_constant_probe_if_inputs_identical(
        honest_add, "E2", candidate_outputs, inputs_list, base_inputs, out_template
    )
    assert probed is True
    # Different inputs actually reached the candidate call:
    assert not np.array_equal(seen["x"], x)
    assert not np.array_equal(seen["y"], y)
    np.testing.assert_allclose(seen["x"], x + 1.0)
    np.testing.assert_allclose(seen["y"], y + 1.0)
    assert seen["n"] == 16
    # Arg order (x, y, out, n): out ptr is neither input ptr, and the probe
    # inputs the candidate saw are the ones recorded into inputs_list[1].
    assert seen["ptrs"][2] not in (seen["ptrs"][0], seen["ptrs"][1])
    assert seen["ptrs"][0] == inputs_list[1][0].data_ptr()
    assert seen["ptrs"][1] == inputs_list[1][1].data_ptr()
    # An honest kernel's probe output differs -> constant check now passes.
    assert not np.array_equal(
        np.asarray(candidate_outputs[0]), np.asarray(candidate_outputs[1])
    )
    np.testing.assert_allclose(np.asarray(candidate_outputs[1]), x + y + 2.0)


def test_constant_probe_e1_different_inputs_reach_candidate():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(12).astype(np.float32)
    out0 = np.maximum(x, 0.0)
    candidate_outputs = [FakeTensor(out0), FakeTensor(out0.copy())]
    inputs_list = [[FakeTensor(x)], [FakeTensor(x.copy())]]

    seen = {}

    def honest_relu(in_ptr, out_ptr, n):
        pin = _buf_at(in_ptr, n)
        seen["in"] = pin.copy()
        seen["n"] = n
        _buf_at(out_ptr, n)[:] = np.maximum(pin, 0.0)

    probed = eval_core._apply_constant_probe_if_inputs_identical(
        honest_relu, "E1", candidate_outputs, inputs_list,
        [FakeTensor(x)], FakeTensor(out0),
    )
    assert probed is True
    assert not np.array_equal(seen["in"], x)
    np.testing.assert_allclose(seen["in"], x + 1.0)
    assert seen["n"] == 12
    assert not np.array_equal(
        np.asarray(candidate_outputs[0]), np.asarray(candidate_outputs[1])
    )


def test_constant_probe_not_triggered_when_inputs_vary():
    """Normal tasks (get_inputs honors the harness seed) keep the old path:
    no extra invocation, outputs untouched."""
    rng = np.random.default_rng(2)
    x0 = rng.standard_normal(8).astype(np.float32)
    x1 = rng.standard_normal(8).astype(np.float32)
    candidate_outputs = [FakeTensor(x0 * 2), FakeTensor(x1 * 2)]
    inputs_list = [[FakeTensor(x0)], [FakeTensor(x1)]]
    before = [np.asarray(t).copy() for t in candidate_outputs]

    def must_not_run(*args):
        raise AssertionError("probe must not re-invoke when inputs differ")

    probed = eval_core._apply_constant_probe_if_inputs_identical(
        must_not_run, "E1", candidate_outputs, inputs_list,
        [FakeTensor(x0)], FakeTensor(x0 * 2),
    )
    assert probed is False
    for got, want in zip(candidate_outputs, before):
        assert np.array_equal(np.asarray(got), want)


def test_constant_probe_still_catches_constant_writer_decoy():
    """Fail-closed: a decoy that writes the same constant regardless of input
    produces an identical output pair, which check_output_not_constant flags."""
    _, _, candidate_outputs, inputs_list, base_inputs, _ = _pinned_e2_setup()
    const_out = np.full(16, 3.14, dtype=np.float32)
    candidate_outputs = [FakeTensor(const_out), FakeTensor(const_out.copy())]

    def decoy(x_ptr, y_ptr, out_ptr, n):
        _buf_at(out_ptr, n)[:] = 3.14

    probed = eval_core._apply_constant_probe_if_inputs_identical(
        decoy, "E2", candidate_outputs, inputs_list,
        base_inputs, FakeTensor(const_out),
    )
    assert probed is True
    assert np.array_equal(
        np.asarray(candidate_outputs[0]), np.asarray(candidate_outputs[1])
    )  # identical pair -> flagged as constant by check_output_not_constant


def test_constant_probe_still_catches_write_nothing_decoy():
    """A kernel that never writes its output leaves the probe buffer holding a
    clone of the first output -> identical pair -> flagged."""
    _, _, candidate_outputs, inputs_list, base_inputs, out_template = _pinned_e2_setup()

    def write_nothing(x_ptr, y_ptr, out_ptr, n):
        pass

    probed = eval_core._apply_constant_probe_if_inputs_identical(
        write_nothing, "E2", candidate_outputs, inputs_list, base_inputs, out_template
    )
    assert probed is True
    assert np.array_equal(
        np.asarray(candidate_outputs[0]), np.asarray(candidate_outputs[1])
    )


def test_constant_probe_failure_raises_for_fail_closed_handling():
    """Identical inputs but no captured seed-0 buffers -> the probe raises and
    the eval path converts that into hack_suspected (fail closed)."""
    _, _, candidate_outputs, inputs_list, _, _ = _pinned_e2_setup()
    with pytest.raises(RuntimeError, match="probe unavailable"):
        eval_core._apply_constant_probe_if_inputs_identical(
            lambda *a: None, "E2", candidate_outputs, inputs_list, [], None
        )


def test_constant_probe_full_chain_with_torch():
    """With real torch CPU tensors: pinned-input E2 + honest kernel passes
    check_output_not_constant after the probe; a decoy still fails it."""
    torch = pytest.importorskip("torch")
    from openenv_env.anti_hack import check_output_not_constant

    torch.manual_seed(0)
    x = torch.randn(32)
    y = torch.randn(32)
    out0 = (x + y).contiguous()
    inputs_list = [
        [x.clone(), y.clone()],
        [x.clone(), y.clone()],  # pinned task: second sample identical
    ]

    def honest_add(x_ptr, y_ptr, out_ptr, n):
        px = _buf_at(x_ptr, n)
        py = _buf_at(y_ptr, n)
        _buf_at(out_ptr, n)[:] = px + py

    candidate_outputs = [out0.clone(), out0.clone()]
    probed = eval_core._apply_constant_probe_if_inputs_identical(
        honest_add, "E2", candidate_outputs, inputs_list,
        [x.contiguous(), y.contiguous()], out0.clone(),
    )
    assert probed is True
    passed, reason = check_output_not_constant(candidate_outputs[0], candidate_outputs[1])
    assert passed is True, reason

    def decoy(x_ptr, y_ptr, out_ptr, n):
        _buf_at(out_ptr, n)[:] = 3.14

    const = torch.full((32,), 3.14)
    decoy_outputs = [const.clone(), const.clone()]
    inputs_list2 = [[x.clone(), y.clone()], [x.clone(), y.clone()]]
    probed = eval_core._apply_constant_probe_if_inputs_identical(
        decoy, "E2", decoy_outputs, inputs_list2,
        [x.contiguous(), y.contiguous()], const.clone(),
    )
    assert probed is True
    passed, reason = check_output_not_constant(decoy_outputs[0], decoy_outputs[1])
    assert passed is False
    assert "constant" in reason


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
