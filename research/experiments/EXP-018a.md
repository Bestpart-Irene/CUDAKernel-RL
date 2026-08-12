# EXP-018a — Unify ops6k eval onto WCC-style extern "C" + dlsym contract

Part of [CAMPAIGN-018-unify-contract](../campaigns/CAMPAIGN-018-unify-contract.md).

## Spec

- **campaign**: CAMPAIGN-018-unify-contract
- **hypothesis**: rewriting `evaluate_ops6k_kernel_impl` to use raw nvcc
  + `dlopen` + `dlsym` of `extern "C" void run_kernel(...)` — mirroring the
  WCC path — closes the torch-extension reward-hack channel structurally
  and makes the user's design innovations I1 (v3-symbol-shaped reward),
  I2 (ABI annotation SFT), I4 (anti-hack stack) apply uniformly to ops6k
  tasks.
- **parent master hash**: null (master rolled back 2026-05-28)
- **variable changed**: this is a multi-file structural change, not a
  single-variable run. It is the prerequisite infrastructure step for
  the campaign and is NOT promotion-eligible. The file list:
  - `eval_service/eval_core.py` — replace `evaluate_ops6k_kernel_impl`
    body with extern-"C" eval (mirroring `evaluate_kernel_impl` WCC
    path); remove `cpp_extension.load` linkage entirely from this
    function. Keep the WCC function unchanged.
  - `openenv_env/anti_hack.py` — `scan_forbidden_symbols` is now meaningful
    on this path. No new code, just gets called from the unified eval.
  - `tasks/build_task_pool.py` — add per-task `extern_c_signature` field
    that the evaluator uses to know how to invoke the candidate.
  - `datasets/combined_kernelforge.jsonl` — add `extern_c_signature` to
    each ops_local_fallback row.
  - `training/task_support.py::build_modal_payload` / `normalize_task_row`
    — propagate `extern_c_signature` through the payload.
- **runner**: explorer-h200 (local smoke; this is infra, not training)
- **expected upside**: precondition for EXP-018b, 018c, 018d. No
  scientific result of its own — value is "the next experiment is now
  interpretable."
- **duplicate check**: `do-not-repeat.md` 2026-05-28 "Sakana SFT +
  ops6k torch-extension eval path is a reward-hack channel" lists
  "raw `extern "C"` contract" as option (2) of the two known
  structural fixes. This experiment implements that option.
- **touches reward**: yes (changes the eval that produces reward inputs).
  Flag `kernelforge-reward-design` skill discipline. Required tests:
  fixture replay through `reward_from_env` on at least 3 hand-crafted
  candidates (correct / wrong / hack-attempt) confirming the reward
  bucket is what we expect.

## Unified contract design

### `extern "C"` signature classes

Five canonical signatures cover the elementary ops we will use for
EXP-018c verification:

| Class | Signature | Use |
|---|---|---|
| **E1** unary elementwise (shape-preserving) | `extern "C" void run_kernel(const float* x, float* out, int n);` | F.elu, F.softplus, F.relu, F.gelu, vector_add-with-scalar |
| **E2** binary elementwise (shape-preserving) | `extern "C" void run_kernel(const float* x, const float* y, float* out, int n);` | vector_add, vector_mul, hadamard |
| **R1** unary reduction-to-scalar | `extern "C" void run_kernel(const float* x, float* out_scalar, int n);` | torch.sum, torch.mean (full reduction) |
| **R2** unary reduction-along-axis | `extern "C" void run_kernel(const float* x, float* out, int n_outer, int n_inner);` | torch.sum(dim=-1), torch.mean(dim=0) |
| **M1** matmul-like | `extern "C" void run_kernel(const float* a, const float* b, float* out, int m, int n, int k);` | torch.matmul (2D simplified) |

`n` is `total_elements`. The harness flattens torch tensors to 1D before
passing the `data_ptr`. Shape information is reconstructed Python-side
from the reference output's shape.

### Eval invocation flow (per rollout, mirrors WCC path)

```
1. extract_cuda_code(completion) -> .cu source
2. scan_forbidden_symbols (source-level pre-scan):
   - reject if source contains `#include <torch/`, `#include <ATen/`,
     `#include <c10/`, or any `torch::`, `at::`, `c10::` substring
   - this is enforced BEFORE compile (no false positives because
     no libtorch is linked anyway)
3. nvcc -O3 -arch=sm_90a -shared -Xcompiler -fPIC kernel.cu -o kernel.so
   (raw nvcc, no torch/extension.h, no libtorch linkage)
4. nm -D kernel.so | grep run_kernel -> verify symbol present
   (this is the v3-symbol-shaped reward's symbol_loaded probe)
5. dlopen(kernel.so) + dlsym("run_kernel")
6. For each (input_tensors, expected_output) pair from ref_model:
   a. cudaMalloc + cudaMemcpy(input.data_ptr -> device)
   b. cudaMalloc(output_buf, sizeof(float) * n_out)
   c. invoke via ctypes per signature class
   d. cudaMemcpy(device output -> host)
   e. torch.testing.assert_close(host_output, expected_output,
      rtol=1e-3, atol=1e-3)
   f. anti_hack runtime checks: not_constant, not_passthrough,
      not_noop, shapes_match
7. Timing: 10 warmup + 30 timed via cudaEvent on the dlsym'd function
   only (no PyTorch in the loop)
8. Compare to eager / compile baseline (run reference Python in
   separate context for timing reference)
```

### Per-task field in `combined_kernelforge.jsonl`

Add the following field to each ops_local_fallback row:

```json
"extern_c_signature": {
  "class": "E1",
  "input_dtypes": ["float32"],
  "output_dtype": "float32",
  "input_shapes_from_ref": true,
  "output_shape_from_ref": true,
  "extra_args": []
}
```

For tasks whose reference operator has stateful init params (e.g.
`F.rrelu(lower=0.1, upper=0.3)`), `extra_args` carries the constants
the candidate `run_kernel` must consume:

```json
"extra_args": [
  {"name": "lower", "ctype": "float", "value": 0.1},
  {"name": "upper", "ctype": "float", "value": 0.3}
]
```

The harness appends these to the dlsym call in declaration order:
`run_kernel(x, out, n, lower, upper)`.

## Concrete 3-task spike pool for EXP-018c

| ops_id | Reference (Python) | Signature class | Full extern "C" form |
|---|---|---|---|
| **vector_add** (hand-crafted, not in current dataset) | `x + y` | E2 | `extern "C" void run_kernel(const float* x, const float* y, float* out, int n);` |
| **F.elu** (already in dataset) | `F.elu(x)` (alpha=1.0 default) | E1 | `extern "C" void run_kernel(const float* x, float* out, int n);` |
| **F.softplus** (already in dataset) | `F.softplus(x)` (beta=1.0 threshold=20 default) | E1 | `extern "C" void run_kernel(const float* x, float* out, int n);` |

These three are:
- pure elementwise → simplest possible kernel a 30B can write
- single-input or two-input → minimal ctypes complexity
- shape-preserving → expected output shape comes free from ref
- already exist in `combined_kernelforge.jsonl` (rows 192+, except
  vector_add which we add as `vector_add_e2`)

vector_add is included as a sanity floor: if a base model can't
emit a working `out[i] = x[i] + y[i]` kernel, the entire pipeline
is broken regardless of any other change. EXP-018c's pass criterion
(3) — "≥ 1 correct=True under post-hoc anti-hack scan" — must
include at least one correct rollout on vector_add as a sanity
check that the eval harness itself works.

## Engineering plan (~1 week, 5-7 working days)

| Day | Work |
|---|---|
| 1 | Reference Python harness (`scripts/probe_extern_c_eval.py`): given a Python `Model` + `extern_c_signature` + hand-written `.cu`, perform end-to-end eval. Test on hand-crafted vector_add candidate. **This is the EXP-018a prototype = step C of the current plan; sets the contract empirically.** |
| 2 | Refactor `eval_core.py::evaluate_ops6k_kernel_impl` to call into the new harness. Keep `evaluate_kernel_impl` (WCC) unchanged — the two paths now do essentially the same thing for different tasks. |
| 3 | Update `tasks/build_task_pool.py` to emit `extern_c_signature`. Regenerate `combined_kernelforge.jsonl` for the 32 ops_local_fallback rows (auto-classify by `ops` field; manual review). |
| 4 | Add `vector_add_e2` row to the dataset. Update `training/task_support.py` to propagate the new field. |
| 5 | Fixture replay tests: replay EXP-015'-A's 5 known-hacked completions through the new evaluator. ALL 5 must now report `compiles=False` (source rejected at step 2) — this is the campaign-level acceptance test. |
| 6 | Single-rollout smoke on H200: generate 1 completion per task from current Sakana-SFT checkpoint, run through new evaluator, confirm bucket distribution is plausible. |
| 7 | Buffer / cleanup / write `## Run` section. |

## Run

- job id: local (darwin host, negative half only) — 2026-08-10
- log path: n/a (inline; see research/audits/2026-08-10-README.md context)
- fixture replay verdict: **5/5 hack variants rejected at `Source rejected:`**
  (V1 torch/extension.h include, V2 ATen include, V3 c10 namespace,
  V4 torch:: forward-decl, V5 at::native dispatch), executed via
  `scripts/fixture_replay_018a.py` fixtures against the 2026-08-10-hardened
  evaluator, `evaluator_sha=b182df91c482`. Source rejection fires before any
  nvcc invocation, so the negative half is valid on a CUDA-less host.
- positive control: **PASSED 2026-08-11** — slurm 9080628 (Explorer sharing
  A100, 30s): clean extern "C" elu kernel compiles=True correct=True,
  runtime 0.0696ms, sv_eager 0.97; 5/5 hack rejections reproduced on the
  CUDA host in the same job. OVERALL: PASS. Log:
  `logs/kf_fixture_018a_9080628.out` (cluster). **Acceptance gate fully
  closed** — both halves recorded.
- single-rollout smoke verdict: superseded by the EXP-018c-p0 probe
  (32 samples/task × 3 tasks — a strict superset of the Day-6 smoke).
- promote: false (verification-only, infra step)

## Interpretation

(populated after run)
