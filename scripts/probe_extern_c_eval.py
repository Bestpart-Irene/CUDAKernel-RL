#!/usr/bin/env python3
"""EXP-018a prototype harness.

Validates the unified extern "C" + dlsym eval contract end-to-end on a
single hand-written task (vector_add). Run on a CUDA-capable host
(NU Explorer H200 / A100). The point is to prove the contract is
implementable before touching production eval_core.py.

Flow per candidate kernel:
    1. Source-level forbidden-include scan
    2. nvcc -shared -fPIC -arch=<sm> kernel.cu -o kernel.so
    3. nm -D kernel.so | grep run_kernel  (symbol_loaded probe)
    4. dlopen + dlsym
    5. cudaMalloc + cudaMemcpy via ctypes (no torch in the eval call path)
    6. Compare host output to a PyTorch reference (PyTorch only used to
       *compute* expected output, not inside the eval call)
    7. anti-hack runtime checks

Usage:
    python scripts/probe_extern_c_eval.py
"""

from __future__ import annotations

import ctypes
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# ----------------------------------------------------------------------
# Hand-written reference: vector_add as a stateless single-op task
# ----------------------------------------------------------------------

REFERENCE_PY = """
import torch


class Model(torch.nn.Module):
    def forward(self, x, y):
        return x + y


def get_inputs():
    n = 1 << 20  # 1M floats
    torch.manual_seed(0)
    return [torch.randn(n), torch.randn(n)]


def get_init_inputs():
    return []
"""

# Hand-written correct candidate
CANDIDATE_CORRECT = r"""
extern "C" __global__ void va_kernel(const float* x, const float* y, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] + y[i];
}

extern "C" void run_kernel(const float* x, const float* y, float* out, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    va_kernel<<<grid, block>>>(x, y, out, n);
    cudaDeviceSynchronize();
}
"""

# Wrong candidate: writes zeros
CANDIDATE_WRONG = r"""
extern "C" __global__ void va_kernel(const float* x, const float* y, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 0.0f;
}

extern "C" void run_kernel(const float* x, const float* y, float* out, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    va_kernel<<<grid, block>>>(x, y, out, n);
    cudaDeviceSynchronize();
}
"""

# Hack attempt: include torch header — must be rejected at step 1
CANDIDATE_HACK_INCLUDE = r"""
#include <torch/extension.h>
extern "C" void run_kernel(const float* x, const float* y, float* out, int n) {
    auto xt = torch::from_blob((void*)x, {n});
    auto yt = torch::from_blob((void*)y, {n});
    auto ot = xt + yt;
    cudaMemcpy(out, ot.data_ptr(), n * sizeof(float), cudaMemcpyDeviceToDevice);
}
"""

# Hack attempt: try to call torch::add — must fail at step 2 (compile)
# because no libtorch is linked
CANDIDATE_HACK_TORCHCALL = r"""
namespace torch { float* add(const float*, const float*, int); }
extern "C" void run_kernel(const float* x, const float* y, float* out, int n) {
    float* r = torch::add(x, y, n);
    cudaMemcpy(out, r, n * sizeof(float), cudaMemcpyDeviceToDevice);
}
"""

# Passthrough hack: returns x unchanged
CANDIDATE_PASSTHROUGH = r"""
extern "C" void run_kernel(const float* x, const float* y, float* out, int n) {
    cudaMemcpy(out, x, n * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
}
"""


# ----------------------------------------------------------------------
# Harness
# ----------------------------------------------------------------------

FORBIDDEN_SOURCE_PATTERNS = [
    r'#\s*include\s*[<"]torch/',
    r'#\s*include\s*[<"]ATen/',
    r'#\s*include\s*[<"]c10/',
    r'\btorch::',
    r'\bat::',
    r'\bc10::',
]


@dataclass
class EvalResult:
    compiled: bool
    symbol_loaded: bool
    correct: bool
    runtime_ms: float
    error: str
    verifier_msg: str
    bucket: str  # one of: source_rejected, compile_failed, symbol_missing,
                 # wrong, hack_constant, hack_passthrough, correct


def scan_source(source: str) -> str | None:
    """Return rejection reason if source matches a forbidden pattern, else None."""
    for pat in FORBIDDEN_SOURCE_PATTERNS:
        m = re.search(pat, source)
        if m:
            return f"forbidden source pattern: {pat!r} matched {m.group(0)!r}"
    return None


def compile_kernel(source: str, sm_arch: str, workdir: Path) -> tuple[Path | None, str]:
    cu = workdir / "kernel.cu"
    so = workdir / "kernel.so"
    cu.write_text(source)
    cmd = [
        "nvcc", "-O3", f"-arch={sm_arch}",
        "-shared", "-Xcompiler", "-fPIC",
        str(cu), "-o", str(so),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        return None, proc.stderr[:1500]
    return so, ""


def symbol_present(so_path: Path, name: str) -> bool:
    proc = subprocess.run(["nm", "-D", str(so_path)], capture_output=True, text=True, timeout=10)
    return name in proc.stdout


def eval_candidate(source: str, sm_arch: str = "sm_90a") -> EvalResult:
    """Run one candidate through the full harness."""
    import torch  # only used to compute reference output and supply input data

    # Step 1: source-level scan
    reason = scan_source(source)
    if reason:
        return EvalResult(False, False, False, 0.0, reason, reason, "source_rejected")

    with tempfile.TemporaryDirectory() as td:
        workdir = Path(td)

        # Step 2: compile
        so_path, err = compile_kernel(source, sm_arch, workdir)
        if not so_path:
            return EvalResult(False, False, False, 0.0, err, err, "compile_failed")

        # Step 3: symbol probe
        if not symbol_present(so_path, "run_kernel"):
            msg = "Kernel FFI verification failed: undefined symbol: run_kernel"
            return EvalResult(True, False, False, 0.0, msg, msg, "symbol_missing")

        # Step 4: dlopen + dlsym
        lib = ctypes.CDLL(str(so_path))
        run_kernel = lib.run_kernel
        run_kernel.argtypes = [
            ctypes.c_void_p,  # const float* x
            ctypes.c_void_p,  # const float* y
            ctypes.c_void_p,  # float* out
            ctypes.c_int,     # int n
        ]
        run_kernel.restype = None

        # Step 5: compute reference via PyTorch (this stays in Python,
        # not in the eval call path)
        ns: dict = {}
        exec(REFERENCE_PY, ns)
        model = ns["Model"]().eval().cuda()
        inputs = [t.cuda().contiguous() for t in ns["get_inputs"]()]
        with torch.no_grad():
            ref_out = model(*inputs).contiguous()
        n = inputs[0].numel()

        # Step 6: invoke candidate via raw pointers
        cand_out = torch.empty_like(ref_out)
        run_kernel(inputs[0].data_ptr(), inputs[1].data_ptr(),
                   cand_out.data_ptr(), n)
        torch.cuda.synchronize()

        # Step 7: correctness via torch.testing (same tolerance as eval_core)
        try:
            torch.testing.assert_close(cand_out, ref_out, rtol=1e-3, atol=1e-3)
            correct = True
            err = ""
            vmsg = "Outputs matched reference"
        except AssertionError as e:
            correct = False
            err = f"Correctness check failed: {str(e)[:300]}"
            vmsg = err

        # Step 8: anti-hack runtime checks
        if correct:
            # not_passthrough vs x
            if torch.equal(cand_out, inputs[0]):
                correct = False
                err = "anti_hack: output identical to input x (passthrough)"
                vmsg = err
            # not_constant via second random input set
            torch.manual_seed(99)
            inputs2 = [torch.randn(n).cuda().contiguous() for _ in range(2)]
            cand_out2 = torch.empty_like(cand_out)
            run_kernel(inputs2[0].data_ptr(), inputs2[1].data_ptr(),
                       cand_out2.data_ptr(), n)
            torch.cuda.synchronize()
            if torch.equal(cand_out, cand_out2):
                correct = False
                err = "anti_hack: output constant across different inputs"
                vmsg = err

        # Step 9: timing (cudaEvent on dlsym'd call only)
        if correct:
            for _ in range(10):
                run_kernel(inputs[0].data_ptr(), inputs[1].data_ptr(),
                           cand_out.data_ptr(), n)
            torch.cuda.synchronize()
            times = []
            for _ in range(30):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                run_kernel(inputs[0].data_ptr(), inputs[1].data_ptr(),
                           cand_out.data_ptr(), n)
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end))
            runtime_ms = sorted(times)[len(times) // 2]
        else:
            runtime_ms = 0.0

        if correct:
            bucket = "correct"
        elif "passthrough" in err:
            bucket = "hack_passthrough"
        elif "constant" in err:
            bucket = "hack_constant"
        else:
            bucket = "wrong"

        return EvalResult(True, True, correct, runtime_ms, err, vmsg, bucket)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

CANDIDATES = [
    ("CORRECT", CANDIDATE_CORRECT, "correct"),
    ("WRONG (writes zeros)", CANDIDATE_WRONG, "wrong"),
    ("HACK: includes <torch/extension.h>", CANDIDATE_HACK_INCLUDE, "source_rejected"),
    ("HACK: torch:: call", CANDIDATE_HACK_TORCHCALL, "source_rejected"),
    ("PASSTHROUGH (returns x)", CANDIDATE_PASSTHROUGH, "hack_passthrough"),
]


def main() -> int:
    sm_arch = os.environ.get("EXTERN_C_SM_ARCH", "sm_90a")
    print(f"[probe_extern_c_eval] sm_arch={sm_arch}")
    print(f"[probe_extern_c_eval] task=vector_add (E2 signature, 1M floats)")
    print()

    all_pass = True
    for name, source, expected_bucket in CANDIDATES:
        print(f"=== {name} ===")
        result = eval_candidate(source, sm_arch)
        verdict = "PASS" if result.bucket == expected_bucket else "FAIL"
        if verdict == "FAIL":
            all_pass = False
        print(f"  compiled={result.compiled}")
        print(f"  symbol_loaded={result.symbol_loaded}")
        print(f"  correct={result.correct}")
        print(f"  bucket={result.bucket} (expected={expected_bucket}) -> {verdict}")
        if result.runtime_ms > 0:
            print(f"  runtime_ms={result.runtime_ms:.4f}")
        if result.error:
            print(f"  error={result.error[:200]}")
        print()

    print(f"[probe_extern_c_eval] OVERALL: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
