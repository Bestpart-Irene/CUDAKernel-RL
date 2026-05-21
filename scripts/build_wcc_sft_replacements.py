"""Wrap the original doublegraph production WCC kernel with an extern "C"
contract entry point and use it for all 4 WCC SFT corpus entries (indices
76-79 in datasets/doublegraph_sft.jsonl).

Rationale (EXP-011, 2026-05-20):

  EXP-009/010 demonstrated the SFT corpus + GRPO pipeline works end-to-end
  but the model's WCC kernel quality is capped by what we hand-wrote in
  the previous version of this script (commit 3c48b0f). After 40 steps of
  Stage 1 GRPO on top of Stage 2 ckpt-144, only 1 out of 80 rollouts
  reached "symbol OK + numerically wrong" — the rest were either compile
  failures (missing helpers, lambda capture issues) or symbol missing.

  The doubleGraph production kernels (visible in
  `git show 40dd553:datasets/doublegraph_sft.jsonl` rows 76-79) use real
  A100 optimization patterns: __launch_bounds__(256), __restrict__, __ldg,
  zero-copy host-pinned convergence flag, sample-then-full hooking, atomic
  pointer-jumping shortcut, adaptive Afforest, warp-level union, etc.
  These were lost when commit 3c48b0f replaced rows 76-79 with simpler
  hand-written kernels. The current SFT data caps the model at
  "Claude-level mediocre kernels" — net loss vs the original corpus
  except that the original couldn't satisfy the verifier contract.

  This script restores the production patterns of variant 76 (the only
  self-contained one — variants 77/78/79 depend on edge_mask and
  segment_offsets which the verifier does NOT supply) and wraps it with
  the canonical extern "C" entry point the verifier dlsym's. All 4 SFT
  rows are populated with the same wrapped kernel; SFT does not need
  variant diversity in the entry point — it needs quality production
  patterns.

  After this script runs, Stage 2 must be retrained from scratch (not
  resumed) because ckpt-144 was SFT'd on the previous mediocre kernels.

Run: `python scripts/build_wcc_sft_replacements.py`
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SFT_PATH = ROOT / "datasets" / "doublegraph_sft.jsonl"
WCC_INDICES = [76, 77, 78, 79]


# This is the production doublegraph WCC kernel (variant 76 in the
# pre-3c48b0f corpus). All __global__ and __device__ bodies are byte-for-byte
# preserved; only the proprietary wrapper has been replaced:
#   - `#include <cugraph/aai/algorithms.hpp>` -> stripped
#   - `namespace aai { namespace { ... }}` -> stripped (kernels at file scope)
#   - `struct Cache : Cacheable {}` -> stripped (inline cudaHostAlloc in wrapper)
#   - `weakly_connected_components(const graph32_t&, int32_t*)` host entry ->
#     replaced with `extern "C" void wcc_kernel(const int*, const int*, int, int*)`
#     that allocates device buffers, copies host CSR in, runs launch_wcc, and
#     copies labels back out.
#   - All optimization patterns retained: __launch_bounds__(256), __restrict__,
#     __ldg, zero-copy host-pinned flag, sample-then-full hooking, atomic
#     pointer-jumping shortcut convergence loop.
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

// Sample-then-full hooking: hit the first `k` neighbors per vertex with a
// cheap pass before paying for the full neighbor list. Skips most of the
// useless union work on dense vertices.
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

    // Zero-copy convergence flag (pinned + mapped) so the device kernel
    // can write `*d_flag = 1` and the host loop reads `*h_flag` without
    // an explicit cudaMemcpy each iteration.
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

    // labels[i] = final root of vertex i
    cudaMemcpy(labels, d_parent, num_vertices * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_parent);
    cudaFreeHost(h_flag);
}
"""


def main() -> None:
    rows = []
    with SFT_PATH.open() as fh:
        for line in fh:
            rows.append(json.loads(line))

    assert len(rows) == 192, f"expected 192 rows, got {len(rows)}"
    for idx in WCC_INDICES:
        row = rows[idx]
        user_content = row["messages"][1]["content"]
        assert "Weakly Connected Components (WCC)" in user_content, (
            f"row {idx} is not a WCC entry"
        )
        row["messages"][2]["content"] = "```cuda\n" + PRODUCTION_WCC_KERNEL + "```"

    SFT_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    )
    print(f"Rewrote {len(WCC_INDICES)} WCC rows in {SFT_PATH}")

    # Self-check: confirm canonical entry exists in every WCC row.
    n_ok = 0
    n_launch = 0
    with SFT_PATH.open() as fh:
        for i, line in enumerate(fh):
            r = json.loads(line)
            if i in WCC_INDICES:
                content = r["messages"][2]["content"]
                assert 'extern "C" void wcc_kernel(' in content, (
                    f"row {i} missing canonical entry"
                )
                assert "__launch_bounds__(256)" in content, (
                    f"row {i} missing __launch_bounds__ — production patterns lost"
                )
                assert "__ldg(" in content, (
                    f"row {i} missing __ldg — production patterns lost"
                )
                n_ok += 1
                n_launch += 1
    assert n_ok == len(WCC_INDICES)
    print(
        f"Self-check OK: {n_ok}/{len(WCC_INDICES)} WCC rows have "
        f"extern \"C\" wcc_kernel + __launch_bounds__(256) + __ldg."
    )


if __name__ == "__main__":
    main()
