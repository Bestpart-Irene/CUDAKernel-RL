"""Build a *mixed* WCC SFT corpus: keep 4 production doublegraph kernels
(rows 76-79) AND append 4 simple Claude-level variants (rows 192-195).

Rationale (EXP-012-A', 2026-05-21):

  EXP-011 produced 0/124 symbol_OK rollouts. Diagnosed: production-only SFT
  pushes the model's prior toward "complex but syntactically wrong"
  imitations (`__attribute__((global))`, `#define __launch_bounds__(256)`,
  `&false` lvalue errors, lambda capture omissions). The model copies
  advanced surface patterns it has never seen used incorrectly — without
  any "drill" that anchors the simple correct contract.

  EXP-010 used 4 simple Claude-level kernels and achieved 1/80 symbol_OK
  ("historic first" of any rollout reaching the symbol-OK bucket).
  Removing those rows in commit 17505f6 cost us that signal.

  EXP-012-A' restores the simple kernels BUT keeps the production kernel
  too — giving SFT a *mixed* prior over both styles. The intent (per the
  ability-decomposition framing the user articulated):

    rows 76-79 (production, identical variant-76 wrapper):
      teaches abilities 6 (performance patterns: __launch_bounds__,
      __restrict__, __ldg) in a *correct in-context use* example.
    rows 192-195 (simple, 4 distinct algorithms):
      teaches abilities 1-5 (contract, CUDA basics, mem lifecycle,
      atomics, WCC algorithm) cleanly with zero ability-6 noise.

  After SFT, the model's prior covers BOTH simple-correct and
  production-correct kernels. GRPO can sample from either mode; the
  expected effect is that the simple kernels lift P(symbol_OK) from ~0%
  back toward EXP-010's ~1.25% while preserving production patterns
  for later reward-tier climbing.

  Stage 2 MUST be retrained fresh (delete /scratch/.../checkpoints/stage2
  before sbatch) — ckpt from EXP-011 is on the broken (production-only)
  prior.

Run: `python scripts/build_wcc_sft_replacements.py`
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SFT_PATH = ROOT / "datasets" / "doublegraph_sft.jsonl"
WCC_INDICES_REPLACE = [76, 77, 78, 79]  # production kernel goes here


# EXP-013-B (2026-05-23): ABI-axis annotations.
#
# SYMPROBE diagnostic on EXP-012-A' rollouts:
#   - 80 SYMPROBE rollouts, 37 with compiles=True
#   - 26/37 (70%) had NO `extern "C"` -> C++ name mangling kills the symbol
#   - 14/37 (38%) wrote `__global__ void wcc_kernel(...)` -- a CUDA device
#     kernel masquerading as the entry symbol, NOT a host wrapper
#   -  4/37 (11%) had BOTH `__global__ wcc_kernel` AND `extern "C"` but
#     placement was wrong
#
# Conclusion: model conflates "the WCC kernel" (the device function) with
# "the wcc_kernel entry symbol" (the host wrapper). Diversity along the
# wrong axis. Inject explicit role-labelling comments at every device
# kernel + before the host wrapper so the two-layer concept is reinforced
# 8 x ~20 occurrences across the SFT corpus.

DEVICE_KERNEL_ANNOTATION = (
    "// DEVICE KERNEL -- runs on GPU, launched from host via <<<grid, block>>>.\n"
    "// NOT the entry symbol the verifier dlsym's. The 'extern \"C\" void wcc_kernel'\n"
    "// host wrapper below is what gets dlsym'd.\n"
)

HOST_WRAPPER_ANNOTATION = (
    "// ===== HOST WRAPPER =====\n"
    "// This is the entry symbol the verifier finds via dlsym(\"wcc_kernel\").\n"
    "// Without 'extern \"C\"', the C++ compiler mangles the name and dlsym fails\n"
    "// with `undefined symbol: wcc_kernel`. The four arguments are HOST POINTERS\n"
    "// (numpy arrays passed via ctypes); this wrapper allocates device memory,\n"
    "// copies host -> device, launches the __global__ kernels above, and copies\n"
    "// labels device -> host.\n"
)

# Match `__global__` optionally followed by `__launch_bounds__(...)`, then
# `void <name>(`. Capture the whole leading run so we can prepend cleanly.
_DEVICE_KERNEL_RE = re.compile(
    r'(__global__\s+(?:__launch_bounds__\([^)]*\)\s+)?void\s+\w+\s*\()',
)
_HOST_WRAPPER_RE = re.compile(r'(extern\s+"C"\s+void\s+wcc_kernel\s*\()')


def _inject_abi_annotations(code: str) -> str:
    """Inject explicit ABI-axis role comments before every __global__ device
    kernel and before the extern "C" host wrapper. Idempotent: re-running
    on already-annotated code is a no-op because the regex matches the
    declaration line itself, and the inserted annotation is placed on the
    line *immediately before* the declaration.

    To make it idempotent we explicitly skip insertion when the previous
    line already contains the annotation marker.
    """
    def _device_repl(m: re.Match) -> str:
        # Look back: if the chunk just before this match already contains
        # the DEVICE KERNEL marker on the immediately preceding line, skip.
        start = m.start()
        prev_window = code[max(0, start - 200):start]
        if "DEVICE KERNEL -- runs on GPU" in prev_window:
            return m.group(0)
        return DEVICE_KERNEL_ANNOTATION + m.group(0)

    def _host_repl(m: re.Match) -> str:
        start = m.start()
        prev_window = code[max(0, start - 400):start]
        if "===== HOST WRAPPER =====" in prev_window:
            return m.group(0)
        return HOST_WRAPPER_ANNOTATION + m.group(0)

    # NOTE: we need to do both substitutions on the same string. Because
    # the insertions shift offsets, just call .sub() which handles that.
    out = _DEVICE_KERNEL_RE.sub(_device_repl, code)
    out = _HOST_WRAPPER_RE.sub(_host_repl, out)
    return out


# Production doublegraph variant-76 wrapped with extern "C" entry.
# Byte-for-byte the same kernel that EXP-011 used at rows 76-79.
PRODUCTION_WCC_KERNEL = r"""// SPDX-License-Identifier: Apache-2.0
// Production WCC kernel adapted from doubleGraph A100 reference
// (rev 40dd553 datasets/doublegraph_sft.jsonl row 76). Original
// proprietary namespace and Cacheable struct removed; the canonical
// `extern "C" void wcc_kernel(...)` entry point is added at the bottom
// to satisfy verification/pac_verify.py:wcc_kernel dlsym contract.
//
// CU_FLAGS: --use_fast_math --extra-device-vectorization
#include <cuda_runtime.h>
#include <cstdint>

__global__ __launch_bounds__(256) void wcc_init(int* __restrict__ parent, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        parent[i] = i;
    }
}

__device__ __forceinline__ int find_root(int* __restrict__ parent, int v) {
    int p = parent[v];
    while (p != parent[p]) {
        int gp = parent[p];
        parent[v] = gp;  // path halving
        v = p;
        p = gp;
    }
    return p;
}

__device__ __forceinline__ void link(int* __restrict__ parent, int u, int v) {
    int ru = find_root(parent, u);
    int rv = find_root(parent, v);
    while (ru != rv) {
        int hi = ru > rv ? ru : rv;
        int lo = ru < rv ? ru : rv;
        int old = atomicCAS(&parent[hi], hi, lo);
        if (old == hi) break;
        ru = find_root(parent, u);
        rv = find_root(parent, v);
    }
}

__global__ __launch_bounds__(256) void wcc_hook_sample(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    int* __restrict__ parent,
    int n,
    int k
) {
    for (int u = blockIdx.x * blockDim.x + threadIdx.x; u < n; u += blockDim.x * gridDim.x) {
        int start = __ldg(&offsets[u]);
        int end = __ldg(&offsets[u + 1]);
        int degree = end - start;
        int samples = degree < k ? degree : k;
        for (int s = 0; s < samples; s++) {
            int v = __ldg(&indices[start + s]);
            link(parent, u, v);
        }
    }
}

__global__ __launch_bounds__(256) void wcc_compress(int* __restrict__ parent, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        int v = i;
        int p = parent[v];
        while (p != parent[p]) {
            int gp = parent[p];
            parent[v] = gp;
            v = p;
            p = gp;
        }
        parent[i] = p;
    }
}

__global__ __launch_bounds__(256) void wcc_hook_full(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    int* __restrict__ parent,
    int n,
    int skip_root
) {
    for (int u = blockIdx.x * blockDim.x + threadIdx.x; u < n; u += blockDim.x * gridDim.x) {
        if (find_root(parent, u) == skip_root) continue;
        int start = __ldg(&offsets[u]);
        int end = __ldg(&offsets[u + 1]);
        for (int e = start; e < end; e++) {
            int v = __ldg(&indices[e]);
            link(parent, u, v);
        }
    }
}

__global__ __launch_bounds__(256) void wcc_shortcut_check(
    int* __restrict__ parent,
    int n,
    int* __restrict__ not_converged
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        int p = parent[i];
        int gp = parent[p];
        if (p != gp) {
            parent[i] = gp;
            *not_converged = 1;
        }
    }
}

__global__ __launch_bounds__(256) void wcc_shortcut(int* __restrict__ parent, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        int p = parent[i];
        int gp = parent[p];
        if (p != gp) parent[i] = gp;
    }
}

extern "C" void wcc_kernel(const int* row_ptr,
                           const int* col_idx,
                           int num_vertices,
                           int* labels) {
    if (num_vertices <= 0) return;
    int nnz = row_ptr[num_vertices];

    int *d_row_ptr = nullptr, *d_col_idx = nullptr, *d_parent = nullptr;
    cudaMalloc(&d_row_ptr, (num_vertices + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, nnz * sizeof(int));
    cudaMalloc(&d_parent, num_vertices * sizeof(int));
    cudaMemcpy(d_row_ptr, row_ptr, (num_vertices + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice);

    int *h_flag = nullptr, *d_flag = nullptr;
    cudaHostAlloc(&h_flag, sizeof(int), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&d_flag, h_flag, 0);

    const int BLOCK = 256;
    int grid = (num_vertices + BLOCK - 1) / BLOCK;
    cudaStream_t stream = 0;

    wcc_init<<<grid, BLOCK, 0, stream>>>(d_parent, num_vertices);
    wcc_hook_sample<<<grid, BLOCK, 0, stream>>>(d_row_ptr, d_col_idx, d_parent, num_vertices, 2);
    wcc_compress<<<grid, BLOCK, 0, stream>>>(d_parent, num_vertices);
    wcc_hook_full<<<grid, BLOCK, 0, stream>>>(d_row_ptr, d_col_idx, d_parent, num_vertices, 0);
    wcc_compress<<<grid, BLOCK, 0, stream>>>(d_parent, num_vertices);

    *h_flag = 0;
    __sync_synchronize();
    wcc_shortcut_check<<<grid, BLOCK, 0, stream>>>(d_parent, num_vertices, d_flag);
    cudaStreamSynchronize(stream);

    if (*h_flag) {
        for (int iter = 0; iter < 100; iter++) {
            for (int j = 0; j < 5; j++) {
                wcc_shortcut<<<grid, BLOCK, 0, stream>>>(d_parent, num_vertices);
            }
            *h_flag = 0;
            __sync_synchronize();
            wcc_shortcut_check<<<grid, BLOCK, 0, stream>>>(d_parent, num_vertices, d_flag);
            cudaStreamSynchronize(stream);
            if (!(*h_flag)) break;
        }
    }

    cudaMemcpy(labels, d_parent, num_vertices * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_parent);
    cudaFreeHost(h_flag);
}
"""


# 4 simple Claude-level WCC variants. NO __launch_bounds__, NO __restrict__,
# NO __ldg, NO cudaHostAlloc. Pure ability 1-5 demonstrations. Each ~70
# lines. Different algorithms across the 4 so the model sees algorithmic
# diversity within the simple-style regime.

SIMPLE_V0_ATOMICCAS_UF = r"""// Simple WCC via atomicCAS union-find (no path compression).
// Demonstrates: extern C contract + cudaMalloc/Memcpy/Free + atomicCAS basics.
// CU_FLAGS: --extended-lambda
#include <cuda_runtime.h>
#include <cstdint>

__global__ void wcc_init(int* parent, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) parent[tid] = tid;
}

__device__ int find_root(const int* parent, int v) {
    while (parent[v] != v) v = parent[v];
    return v;
}

__global__ void wcc_union(const int* row_ptr, const int* col_idx, int* parent, int n) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n) return;
    for (int e = row_ptr[u]; e < row_ptr[u + 1]; ++e) {
        int v = col_idx[e];
        int ru = find_root(parent, u);
        int rv = find_root(parent, v);
        while (ru != rv) {
            int hi = ru > rv ? ru : rv;
            int lo = ru < rv ? ru : rv;
            int old = atomicCAS(&parent[hi], hi, lo);
            if (old == hi) break;
            ru = find_root(parent, u);
            rv = find_root(parent, v);
        }
    }
}

__global__ void wcc_flatten(int* parent, int* labels, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) labels[tid] = find_root(parent, tid);
}

extern "C" void wcc_kernel(const int* row_ptr, const int* col_idx, int num_vertices, int* labels) {
    int nnz = row_ptr[num_vertices];
    int *d_row_ptr, *d_col_idx, *d_parent, *d_labels;
    cudaMalloc(&d_row_ptr, (num_vertices + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, nnz * sizeof(int));
    cudaMalloc(&d_parent, num_vertices * sizeof(int));
    cudaMalloc(&d_labels, num_vertices * sizeof(int));
    cudaMemcpy(d_row_ptr, row_ptr, (num_vertices + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice);

    const int BLOCK = 256;
    int grid = (num_vertices + BLOCK - 1) / BLOCK;
    wcc_init<<<grid, BLOCK>>>(d_parent, num_vertices);
    wcc_union<<<grid, BLOCK>>>(d_row_ptr, d_col_idx, d_parent, num_vertices);
    wcc_flatten<<<grid, BLOCK>>>(d_parent, d_labels, num_vertices);
    cudaMemcpy(labels, d_labels, num_vertices * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_row_ptr); cudaFree(d_col_idx); cudaFree(d_parent); cudaFree(d_labels);
}
"""

SIMPLE_V1_PATH_HALVING = r"""// Simple WCC via union-find with path-halving find().
// Demonstrates: same as v0 plus iterative path compression idiom.
// CU_FLAGS: --extended-lambda
#include <cuda_runtime.h>
#include <cstdint>

__global__ void wcc_init(int* parent, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) parent[tid] = tid;
}

__device__ int find_compress(int* parent, int v) {
    int p = parent[v];
    while (p != parent[p]) {
        int gp = parent[p];
        parent[v] = gp;
        v = p;
        p = gp;
    }
    return p;
}

__global__ void wcc_link(const int* row_ptr, const int* col_idx, int* parent, int n) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n) return;
    for (int e = row_ptr[u]; e < row_ptr[u + 1]; ++e) {
        int v = col_idx[e];
        int ru = find_compress(parent, u);
        int rv = find_compress(parent, v);
        while (ru != rv) {
            int hi = ru > rv ? ru : rv;
            int lo = ru < rv ? ru : rv;
            int old = atomicCAS(&parent[hi], hi, lo);
            if (old == hi) break;
            ru = find_compress(parent, u);
            rv = find_compress(parent, v);
        }
    }
}

__global__ void wcc_flatten(int* parent, int* labels, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) labels[tid] = find_compress(parent, tid);
}

extern "C" void wcc_kernel(const int* row_ptr, const int* col_idx, int num_vertices, int* labels) {
    int nnz = row_ptr[num_vertices];
    int *d_row_ptr, *d_col_idx, *d_parent, *d_labels;
    cudaMalloc(&d_row_ptr, (num_vertices + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, nnz * sizeof(int));
    cudaMalloc(&d_parent, num_vertices * sizeof(int));
    cudaMalloc(&d_labels, num_vertices * sizeof(int));
    cudaMemcpy(d_row_ptr, row_ptr, (num_vertices + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice);

    const int BLOCK = 256;
    int grid = (num_vertices + BLOCK - 1) / BLOCK;
    wcc_init<<<grid, BLOCK>>>(d_parent, num_vertices);
    wcc_link<<<grid, BLOCK>>>(d_row_ptr, d_col_idx, d_parent, num_vertices);
    wcc_flatten<<<grid, BLOCK>>>(d_parent, d_labels, num_vertices);
    cudaMemcpy(labels, d_labels, num_vertices * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_row_ptr); cudaFree(d_col_idx); cudaFree(d_parent); cudaFree(d_labels);
}
"""

SIMPLE_V2_LABEL_PROP = r"""// Simple WCC via label propagation. Each vertex takes min(labels[neighbors]),
// iterated until no changes. No union-find at all.
// Demonstrates: extern C + iterative kernel pattern + atomicExch.
// CU_FLAGS: --extended-lambda
#include <cuda_runtime.h>
#include <cstdint>

__global__ void lp_init(int* labels, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) labels[tid] = tid;
}

__global__ void lp_iterate(const int* row_ptr, const int* col_idx, int* labels, int* changed, int n) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n) return;
    int best = labels[u];
    for (int e = row_ptr[u]; e < row_ptr[u + 1]; ++e) {
        int v = col_idx[e];
        int lv = labels[v];
        if (lv < best) best = lv;
    }
    if (best < labels[u]) {
        labels[u] = best;
        atomicExch(changed, 1);
    }
}

extern "C" void wcc_kernel(const int* row_ptr, const int* col_idx, int num_vertices, int* labels) {
    int nnz = row_ptr[num_vertices];
    int *d_row_ptr, *d_col_idx, *d_labels, *d_changed;
    cudaMalloc(&d_row_ptr, (num_vertices + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, nnz * sizeof(int));
    cudaMalloc(&d_labels, num_vertices * sizeof(int));
    cudaMalloc(&d_changed, sizeof(int));
    cudaMemcpy(d_row_ptr, row_ptr, (num_vertices + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice);

    const int BLOCK = 256;
    int grid = (num_vertices + BLOCK - 1) / BLOCK;
    lp_init<<<grid, BLOCK>>>(d_labels, num_vertices);

    int h_changed = 1;
    int iter = 0;
    while (h_changed && iter < num_vertices) {
        h_changed = 0;
        cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice);
        lp_iterate<<<grid, BLOCK>>>(d_row_ptr, d_col_idx, d_labels, d_changed, num_vertices);
        cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
        iter++;
    }
    cudaMemcpy(labels, d_labels, num_vertices * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_row_ptr); cudaFree(d_col_idx); cudaFree(d_labels); cudaFree(d_changed);
}
"""

SIMPLE_V3_HOOK_JUMP = r"""// Simple WCC via Soman-style hooking + pointer-jumping. Two-pass per round.
// Demonstrates: extern C + atomicMin + multi-kernel orchestration with a
// host-side convergence loop.
// CU_FLAGS: --extended-lambda
#include <cuda_runtime.h>
#include <cstdint>

__global__ void hk_init(int* parent, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) parent[tid] = tid;
}

__global__ void hk_hook(const int* row_ptr, const int* col_idx, int* parent, int* changed, int n) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n) return;
    int pu = parent[u];
    for (int e = row_ptr[u]; e < row_ptr[u + 1]; ++e) {
        int v = col_idx[e];
        int pv = parent[v];
        if (pu == pv) continue;
        int hi = pu > pv ? pu : pv;
        int lo = pu < pv ? pu : pv;
        if (atomicMin(&parent[hi], lo) > lo) {
            atomicExch(changed, 1);
        }
    }
}

__global__ void hk_jump(int* parent, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int p = parent[tid];
        int gp = parent[p];
        if (p != gp) parent[tid] = gp;
    }
}

__global__ void hk_flatten(const int* parent, int* labels, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int v = tid;
        while (parent[v] != v) v = parent[v];
        labels[tid] = v;
    }
}

extern "C" void wcc_kernel(const int* row_ptr, const int* col_idx, int num_vertices, int* labels) {
    int nnz = row_ptr[num_vertices];
    int *d_row_ptr, *d_col_idx, *d_parent, *d_labels, *d_changed;
    cudaMalloc(&d_row_ptr, (num_vertices + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, nnz * sizeof(int));
    cudaMalloc(&d_parent, num_vertices * sizeof(int));
    cudaMalloc(&d_labels, num_vertices * sizeof(int));
    cudaMalloc(&d_changed, sizeof(int));
    cudaMemcpy(d_row_ptr, row_ptr, (num_vertices + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice);

    const int BLOCK = 256;
    int grid = (num_vertices + BLOCK - 1) / BLOCK;
    hk_init<<<grid, BLOCK>>>(d_parent, num_vertices);

    int h_changed = 1;
    int iter = 0;
    while (h_changed && iter < num_vertices) {
        h_changed = 0;
        cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice);
        hk_hook<<<grid, BLOCK>>>(d_row_ptr, d_col_idx, d_parent, d_changed, num_vertices);
        hk_jump<<<grid, BLOCK>>>(d_parent, num_vertices);
        cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
        iter++;
    }
    hk_flatten<<<grid, BLOCK>>>(d_parent, d_labels, num_vertices);
    cudaMemcpy(labels, d_labels, num_vertices * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_row_ptr); cudaFree(d_col_idx); cudaFree(d_parent); cudaFree(d_labels); cudaFree(d_changed);
}
"""

SIMPLE_VARIANTS = [
    SIMPLE_V0_ATOMICCAS_UF,
    SIMPLE_V1_PATH_HALVING,
    SIMPLE_V2_LABEL_PROP,
    SIMPLE_V3_HOOK_JUMP,
]


def main() -> None:
    rows = []
    with SFT_PATH.open() as fh:
        for line in fh:
            rows.append(json.loads(line))

    # EXP-013-B: inject ABI-axis annotations into both production and simple
    # kernels before writing them out.
    annotated_production = _inject_abi_annotations(PRODUCTION_WCC_KERNEL)
    annotated_simple = [_inject_abi_annotations(v) for v in SIMPLE_VARIANTS]

    # Sanity: each annotated kernel must contain both marker strings.
    assert "DEVICE KERNEL -- runs on GPU" in annotated_production
    assert "===== HOST WRAPPER =====" in annotated_production
    for i, v in enumerate(annotated_simple):
        assert "DEVICE KERNEL -- runs on GPU" in v, f"simple v{i} missing device annotation"
        assert "===== HOST WRAPPER =====" in v, f"simple v{i} missing host annotation"

    # Replace rows 76-79 with the production kernel (already there from EXP-011
    # but re-write to be sure / idempotent).
    for idx in WCC_INDICES_REPLACE:
        row = rows[idx]
        assert "Weakly Connected Components (WCC)" in row["messages"][1]["content"], (
            f"row {idx} is not a WCC entry"
        )
        row["messages"][2]["content"] = "```cuda\n" + annotated_production + "```"

    # Append 4 simple variants. Reuse the existing 4 WCC prompts (one per
    # variant) so the model sees 2 valid completions per WCC prompt during
    # SFT — one production, one simple.
    new_rows = []
    for variant_idx, variant_code in enumerate(annotated_simple):
        prompt_idx = WCC_INDICES_REPLACE[variant_idx]  # 76, 77, 78, 79
        src_row = rows[prompt_idx]
        new_row = {
            "messages": [
                {
                    "role": "system",
                    "content": src_row["messages"][0]["content"],
                },
                {
                    "role": "user",
                    "content": src_row["messages"][1]["content"],
                },
                {
                    "role": "assistant",
                    "content": "```cuda\n" + variant_code + "```",
                },
            ]
        }
        new_rows.append(new_row)
    rows.extend(new_rows)

    SFT_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    )
    print(f"Total rows after rewrite: {len(rows)} (expected 196)")

    # Self-check.
    with SFT_PATH.open() as fh:
        check_rows = [json.loads(l) for l in fh]
    assert len(check_rows) == 196

    n_prod = 0
    n_simple = 0
    n_wcc_rows = 0
    for i, r in enumerate(check_rows):
        c = r["messages"][2]["content"]
        if "Weakly Connected Components (WCC)" not in r["messages"][1]["content"]:
            continue
        n_wcc_rows += 1
        has_extern_c = 'extern "C" void wcc_kernel(' in c
        has_launch_bounds = "__launch_bounds__(256)" in c
        has_ldg = "__ldg(" in c
        has_device_marker = "DEVICE KERNEL -- runs on GPU" in c
        has_host_marker = "===== HOST WRAPPER =====" in c
        if has_launch_bounds and has_ldg:
            n_prod += 1
            tag = "PRODUCTION"
        elif not has_launch_bounds and not has_ldg:
            n_simple += 1
            tag = "simple"
        else:
            tag = "MIXED?"
        assert has_extern_c, f"row {i} missing extern C entry"
        assert has_device_marker, (
            f"row {i} ({tag}) missing DEVICE KERNEL annotation -- "
            f"EXP-013-B annotation injection failed"
        )
        assert has_host_marker, (
            f"row {i} ({tag}) missing HOST WRAPPER annotation -- "
            f"EXP-013-B annotation injection failed"
        )
        # Count DEVICE KERNEL annotations vs ACTUAL __global__ declarations
        # (not just any mention of __global__ — the annotation comment itself
        # contains the string "__global__" and would double-count).
        n_device_markers = c.count("DEVICE KERNEL -- runs on GPU")
        # An actual declaration matches `__global__ ... void name(` with no
        # leading `//` comment on the same line.
        decl_re = re.compile(
            r'^[^/\n]*?__global__\s+(?:__launch_bounds__\([^)]*\)\s+)?void\s+\w+\s*\(',
            re.MULTILINE,
        )
        n_global_decls = len(decl_re.findall(c))
        print(
            f"  row {i}: {tag} (len {len(c)} chars, "
            f"{n_device_markers} device markers / {n_global_decls} __global__ decls)"
        )
        assert n_device_markers == n_global_decls, (
            f"row {i}: device marker count {n_device_markers} != "
            f"__global__ declaration count {n_global_decls}"
        )

    print(f"\nSelf-check: {n_prod} production rows, {n_simple} simple rows.")
    assert n_wcc_rows == 8, f"expected exactly 8 WCC rows, got {n_wcc_rows}"
    assert n_prod == 4, f"expected 4 production WCC rows, got {n_prod}"
    assert n_simple == 4, f"expected 4 simple WCC rows, got {n_simple}"
    print(
        "OK: 4 production + 4 simple WCC rows, all expose extern C contract "
        "AND carry EXP-013-B ABI-axis annotations on every __global__ kernel "
        "+ host wrapper."
    )


if __name__ == "__main__":
    main()
