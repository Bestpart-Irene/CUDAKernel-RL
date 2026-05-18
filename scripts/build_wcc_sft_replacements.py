"""Hand-written WCC reference kernels for SFT corpus replacement.

Diagnosed 2026-05-18 (job 6888876): the 4 WCC entries at indices 76-79 in
datasets/doublegraph_sft.jsonl are upstream doubleGraph production code
that lives in `namespace aai`, depends on `cugraph/aai/algorithms.hpp`,
and never exposes a `wcc_kernel(...)` C entry point. The verifier
(verification/pac_verify.py:192) calls
`lib.wcc_kernel(row_ptr_c, col_idx_c, num_vertices, labels_c)` via ctypes
on the resulting .so, so the contract is broken before any model output
can be evaluated.

This builder replaces those 4 rows with 4 algorithmically distinct,
self-contained WCC kernels that all expose the exact contract
`extern "C" void wcc_kernel(const int* row_ptr, const int* col_idx,
int num_vertices, int* labels)`, where row_ptr/col_idx/labels are HOST
(numpy) pointers per the verifier convention.

Run: `python scripts/build_wcc_sft_replacements.py`
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SFT_PATH = ROOT / "datasets" / "doublegraph_sft.jsonl"
WCC_INDICES = [76, 77, 78, 79]


# Each kernel is a complete standalone .cu file. The contract:
#   extern "C" void wcc_kernel(const int* row_ptr, const int* col_idx,
#                              int num_vertices, int* labels)
# row_ptr / col_idx / labels are HOST pointers (numpy). Kernel owns
# cudaMalloc / cudaMemcpy / cudaFree.

WCC_VARIANT_0 = r"""// Variant 0: atomicCAS union-find (links every undirected edge once).
// No path compression — minimal, easy-to-audit baseline.
// CU_FLAGS: --extended-lambda
#include <cuda_runtime.h>
#include <cstdint>

__global__ void wcc_init(int* parent, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) parent[tid] = tid;
}

__device__ __forceinline__ int find_root(const int* parent, int v) {
    while (parent[v] != v) v = parent[v];
    return v;
}

__global__ void wcc_union(const int* __restrict__ row_ptr,
                          const int* __restrict__ col_idx,
                          int* __restrict__ parent,
                          int n) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n) return;
    int start = row_ptr[u];
    int end = row_ptr[u + 1];
    for (int e = start; e < end; ++e) {
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

extern "C" void wcc_kernel(const int* row_ptr,
                           const int* col_idx,
                           int num_vertices,
                           int* labels) {
    int nnz = row_ptr[num_vertices];
    int *d_row_ptr = nullptr, *d_col_idx = nullptr, *d_parent = nullptr, *d_labels = nullptr;
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

    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_parent);
    cudaFree(d_labels);
}
"""

WCC_VARIANT_1 = r"""// Variant 1: union-find with iterative path compression in find().
// Better tree-depth amortization on islands-heavy topologies.
// CU_FLAGS: --extended-lambda
#include <cuda_runtime.h>
#include <cstdint>

__global__ void wcc_init(int* parent, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) parent[tid] = tid;
}

__device__ __forceinline__ int find_compress(int* parent, int v) {
    int p = parent[v];
    while (p != parent[p]) {
        int gp = parent[p];
        parent[v] = gp;  // path-halving
        v = p;
        p = gp;
    }
    return p;
}

__global__ void wcc_link(const int* __restrict__ row_ptr,
                         const int* __restrict__ col_idx,
                         int* __restrict__ parent,
                         int n) {
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

extern "C" void wcc_kernel(const int* row_ptr,
                           const int* col_idx,
                           int num_vertices,
                           int* labels) {
    int nnz = row_ptr[num_vertices];
    int *d_row_ptr = nullptr, *d_col_idx = nullptr, *d_parent = nullptr, *d_labels = nullptr;
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

WCC_VARIANT_2 = r"""// Variant 2: label-propagation. parent[v] = min(parent[neighbor]) iterated
// until no changes. Naturally segment-aware (each component converges
// to its smallest vertex id).
// CU_FLAGS: --extended-lambda
#include <cuda_runtime.h>
#include <cstdint>

__global__ void lp_init(int* labels, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) labels[tid] = tid;
}

__global__ void lp_iterate(const int* __restrict__ row_ptr,
                           const int* __restrict__ col_idx,
                           int* __restrict__ labels,
                           int* __restrict__ changed,
                           int n) {
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

extern "C" void wcc_kernel(const int* row_ptr,
                           const int* col_idx,
                           int num_vertices,
                           int* labels) {
    int nnz = row_ptr[num_vertices];
    int *d_row_ptr = nullptr, *d_col_idx = nullptr, *d_labels = nullptr, *d_changed = nullptr;
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

WCC_VARIANT_3 = r"""// Variant 3: hooking + multi-jump pointer-jumping union-find (Soman et al.).
// Two-pass per round: hook smaller-onto-larger, then pointer-jump.
// CU_FLAGS: --extended-lambda
#include <cuda_runtime.h>
#include <cstdint>

__global__ void hk_init(int* parent, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) parent[tid] = tid;
}

__global__ void hk_hook(const int* __restrict__ row_ptr,
                        const int* __restrict__ col_idx,
                        int* __restrict__ parent,
                        int* __restrict__ changed,
                        int n) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n) return;
    int pu = parent[u];
    for (int e = row_ptr[u]; e < row_ptr[u + 1]; ++e) {
        int v = col_idx[e];
        int pv = parent[v];
        if (pu == pv) continue;
        int hi = pu > pv ? pu : pv;
        int lo = pu < pv ? pu : pv;
        // Hook the higher root onto the lower one.
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

extern "C" void wcc_kernel(const int* row_ptr,
                           const int* col_idx,
                           int num_vertices,
                           int* labels) {
    int nnz = row_ptr[num_vertices];
    int *d_row_ptr = nullptr, *d_col_idx = nullptr, *d_parent = nullptr, *d_labels = nullptr, *d_changed = nullptr;
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

VARIANTS = [WCC_VARIANT_0, WCC_VARIANT_1, WCC_VARIANT_2, WCC_VARIANT_3]


def main() -> None:
    rows = []
    with SFT_PATH.open() as fh:
        for line in fh:
            rows.append(json.loads(line))

    assert len(rows) == 192, f"expected 192 rows, got {len(rows)}"
    for idx, variant in zip(WCC_INDICES, VARIANTS, strict=True):
        row = rows[idx]
        # Confirm we're rewriting a WCC row, not some unrelated kernel.
        user_content = row["messages"][1]["content"]
        assert "Weakly Connected Components (WCC)" in user_content, (
            f"row {idx} is not a WCC entry"
        )
        row["messages"][2]["content"] = "```cuda\n" + variant + "```"

    SFT_PATH.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"Rewrote {len(WCC_INDICES)} WCC rows in {SFT_PATH}")

    # Self-check: confirm every WCC row now has the canonical entry symbol.
    n_ok = 0
    with SFT_PATH.open() as fh:
        for i, line in enumerate(fh):
            r = json.loads(line)
            if i in WCC_INDICES:
                content = r["messages"][2]["content"]
                assert 'extern "C" void wcc_kernel(' in content, (
                    f"row {i} missing canonical entry"
                )
                n_ok += 1
    assert n_ok == len(WCC_INDICES)
    print(f"Self-check OK: {n_ok}/{len(WCC_INDICES)} WCC rows expose `extern \"C\" void wcc_kernel(`.")


if __name__ == "__main__":
    main()
