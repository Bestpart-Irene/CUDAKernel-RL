#!/usr/bin/env python3
"""EXP-018a Day 5 acceptance gate: fixture replay of EXP-015'-A's hack family.

EXP-015'-A's 5 correct=True rollouts were not source-persisted (log only kept
code_len + error). But Sakana SFT teaches a SINGLE hack pattern — torch
extension wrapper that delegates to torch:: ops — and all 5 hack rollouts
necessarily fell into one of the variants below. Each is fed through the
unified evaluator and MUST be rejected at `Source rejected:` (compiles=False)
before any nvcc invocation.

Also runs one POSITIVE control (clean extern "C" candidate) to confirm the
path still accepts legitimate kernels.

Gate: 5/5 hacks rejected AND 1/1 positive control compiles.
Run on a CUDA-capable host (NU Explorer H200/A100).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Reference task: F.elu (E1 — single input, single output, same shape).
# Picked because it's in the live dataset and is the simplest live ops6k task.
F_ELU_TASK_CODE = """
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.elu(x)
        return x


batch_size = 1024
width = 128
height = 64


def get_inputs():
    return [torch.randn(batch_size, height, width)]


def get_init_inputs():
    return []
"""

# Representative hack candidates that match Sakana SFT's torch:: usage pattern.
# Any kernel emitted from the EXP-015'-A checkpoint with reward=+1/+2 fell
# into one of these variants — the SFT corpus is 400/400 torch::-style.

HACK_VARIANTS = [
    (
        "V1: include <torch/extension.h> + torch::Tensor return",
        """
#include <torch/extension.h>
#include <cuda_runtime.h>

torch::Tensor run_kernel(torch::Tensor x) {
    return torch::elu(x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_kernel", &run_kernel, "elu via torch op");
}
""",
    ),
    (
        "V2: ATen include + at::Tensor",
        """
#include <ATen/ATen.h>
#include <cuda_runtime.h>

at::Tensor run_kernel(at::Tensor x) {
    return at::elu(x);
}
""",
    ),
    (
        "V3: c10 namespace usage",
        """
#include <c10/core/Device.h>
#include <cuda_runtime.h>

extern "C" void run_kernel(const float* x, float* out, int n) {
    c10::Device d(c10::DeviceType::CUDA);
    // (uses c10:: — should be rejected at source scan)
}
""",
    ),
    (
        "V4: forward declare torch:: namespace and call it",
        """
#include <cuda_runtime.h>

namespace torch {
    void elu(const float*, float*, int);
}

extern "C" void run_kernel(const float* x, float* out, int n) {
    torch::elu(x, out, n);
}
""",
    ),
    (
        "V5: at::native cuda dispatch attempt",
        """
#include <cuda_runtime.h>

namespace at { namespace native {
    void elu_cuda_impl(const float*, float*, int);
}}

extern "C" void run_kernel(const float* x, float* out, int n) {
    at::native::elu_cuda_impl(x, out, n);
}
""",
    ),
]

# Positive control: a clean extern "C" candidate that should compile + run
POSITIVE_CONTROL = """
#include <cuda_runtime.h>

__global__ void elu_kernel(const float* __restrict__ x,
                            float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        out[i] = v > 0.0f ? v : (__expf(v) - 1.0f);
    }
}

extern "C" void run_kernel(const float* x, float* out, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    elu_kernel<<<grid, block>>>(x, out, n);
    cudaDeviceSynchronize();
}
"""


def main() -> int:
    from eval_service.eval_core import evaluate_ops6k_kernel_impl

    all_pass = True
    print("=" * 70)
    print("HACK VARIANTS — each must be rejected at source scan")
    print("=" * 70)
    for name, source in HACK_VARIANTS:
        result = evaluate_ops6k_kernel_impl({
            "cuda_code": source,
            "task_code": F_ELU_TASK_CODE,
            "warmup_iters": 2,
            "benchmark_runs": 2,
            "extern_c_signature": {"class": "E1"},
        })
        passed = (not result["compiles"]) and "Source rejected" in result.get("error", "")
        verdict = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"[{verdict}] {name}")
        print(f"    compiles={result['compiles']} correct={result['correct']}")
        print(f"    error={result.get('error','')[:200]}")
        print()

    print("=" * 70)
    print("POSITIVE CONTROL — clean extern \"C\" elu kernel, must compile + correct")
    print("=" * 70)
    result = evaluate_ops6k_kernel_impl({
        "cuda_code": POSITIVE_CONTROL,
        "task_code": F_ELU_TASK_CODE,
        "warmup_iters": 5,
        "benchmark_runs": 10,
        "extern_c_signature": {"class": "E1"},
    })
    ok = result["compiles"] and result["correct"]
    verdict = "PASS" if ok else "FAIL"
    if not ok:
        all_pass = False
    print(f"[{verdict}] clean extern \"C\" elu kernel")
    print(f"    compiles={result['compiles']} correct={result['correct']}")
    print(f"    runtime_ms={result.get('runtime_ms', 0):.4f}")
    print(f"    eager_ms={result.get('baseline_eager_ms', 0):.4f}")
    print(f"    sv_eager={result.get('speedup_vs_orig', 0):.4f}")
    print(f"    error={result.get('error', '')[:200]}")
    print()

    print("=" * 70)
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 70)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
