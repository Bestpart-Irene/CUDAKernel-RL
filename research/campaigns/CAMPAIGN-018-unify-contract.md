# CAMPAIGN-018 — Unify WCC and ops6k onto a single extern "C" eval contract so the user's design innovations actually fire on the experimental task pool

## Goal

The user's design (distinct from PRD verbatim) makes 9 innovations, of which 6
have internal contradictions identified on 2026-05-30. The three "MUST" ones
are A, B, D in `research/notes.md` 2026-05-30 audit [2026-08-10 note: this
2026-05-30 audit was never ledgered in notes.md — the citation is preserved
for history but has no ledger anchor; see
research/audits/2026-08-10-planner-plan-coherence.md §4.], all rooted in the same
structural split: novel reward / SFT / anti-hack mechanisms are
WCC-extern-"C"-centric, but the live experimental task pool is ops6k-cpp_extension-
centric. The two paths never reach each other.

This campaign exists to **collapse that split** so the user's stated
innovations (v3-symbol-shaped reward, ABI annotation SFT, 5-check anti-hack
stack, three-prior fusion) can be evaluated on a single coherent eval
path — at which point the design either produces a real RL signal or
fails on the science rather than on infra mismatch.

Success criterion for the whole campaign: under the unified contract, run
one 50-step Stage-1 GRPO that produces a `mean_reward` curve whose last
10 steps' mean is strictly greater than its first 10 steps' mean, with
≥ 1 `correct=True` rollout whose `.cu` source passes a post-hoc deep
anti-hack scan. If that gate passes → continue user's full pipeline
(Stage-2 RFT → Stage-3 GRPO + curriculum) per their own design. If it
fails → the failure is now scientifically interpretable, not an infra
artifact.

This is the prerequisite for any further investment. Until it lands, no
ablation on G / β / loss_type / max_completion / reward shape is worth
running — they will all be on the same broken substrate that produced
EXP-001 through EXP-015'-A.

## Design innovations the campaign is validating (the user's, not PRD's)

- **I1**: v3-symbol-shaped reward (-0.5 bucket for compiled-but-symbol-missing)
- **I2**: ABI-axis annotation injection in SFT (teach contract via in-body comment)
- **I4**: 5-check Dr. Kernel-style anti-hack runtime stack
- **I5**: three-prior fusion (doubleGraph patterns + skills.md heuristics + Ops-6K tasks)
- **I6**: Sakana 400 SFT corpus (the user's chosen alternative to PRD's doubleGraph SFT)

Innovations I3 (TRLOO N/(N-1)), I7 (curriculum), I8 (OpenEnv wrapper),
I9 (SkyDiscover hedge) are out of scope for this campaign — they either
already work in isolation (I3, I8) or presuppose this campaign succeeds
(I7, I9).

## Hypotheses queued

### EXP-018a — Unify all task evaluation onto the WCC-style extern "C" contract

- **hypothesis**: rewriting `evaluate_ops6k_kernel_impl` to use the same
  raw-nvcc + dlsym pattern as `evaluate_kernel_impl` (WCC path) closes
  the torch-extension hack channel structurally and makes innovations
  I1, I2, I4 actually applicable to ops6k tasks.
- **variable changed**: `eval_service/eval_core.py` — replace
  `evaluate_ops6k_kernel_impl` (L588-877) with an extern-"C"-based eval
  that:
  - requires candidate to expose `extern "C" void run_kernel(const float* in,
    float* out, int n, ...)` (per-task signature TBD by `tasks/build_task_pool.py`)
  - compiles with raw nvcc, no `torch/extension.h` linkage
  - runs candidate via `dlopen` + `dlsym` (mirrors WCC path)
  - allocates I/O with `cudaMalloc`/`cudaMemcpy`, compares against
    reference output computed on CPU side via the Python `Model.forward()`
  - calls `scan_forbidden_symbols` on the produced `.so` (now meaningful
    because no libtorch is linked)
- **duplicate check**: not in `do-not-repeat.md`. The closest entry is
  2026-05-28 "Sakana SFT + ops6k torch-extension eval path is a
  reward-hack channel" — this experiment is the structural fix that
  entry's "conditions under which it could be revisited" calls for
  (option 2: raw `extern "C"` contract).
- **status**: spec only — needs `eval_service/eval_core.py` and
  `tasks/build_task_pool.py` rewrite, plus per-task signature schema in
  `combined_kernelforge.jsonl`. Engineering: ~1 week.
- **touches reward**: no, but touches the evaluator that produces
  reward inputs — flag for `kernelforge-reward-design` skill review
  per integrity discipline.

### EXP-018b — Rebuild SFT corpus to match the unified extern "C" contract

- **hypothesis**: with ops6k evaluator on extern "C", Sakana 400 corpus
  (100% torch:: signature) becomes unusable. Replace with a rebuilt
  corpus where each row's assistant body emits `extern "C" void
  run_kernel(...)` raw-CUDA, matching the unified contract.
- **variable changed**: `datasets/build_sakana_sft.py` — add a transform
  pass that rewrites each Sakana row's `torch::Tensor run_kernel(torch::Tensor)`
  body into `extern "C" void run_kernel(const float*, float*, int)` body,
  extracting raw pointer args and inlining the original `__global__` kernel
  launch. Sakana boilerplate is structurally regular (`#include
  <torch/extension.h>` + `CHECK_CUDA` + `auto output = torch::empty_like(x)`
  + launch + `PYBIND11_MODULE`) — a libclang-or-regex rewriter handles
  ~80% of rows automatically, drop the residual 20%. Target output:
  ~300-320 rebuilt rows, file `datasets/sakana_sft_externc.jsonl`.
- **duplicate check**: not in `do-not-repeat.md`. Related to 2026-05-17
  "doubleGraph WCC SFT for ops6k Tensor-return tasks" (ruled out
  *that* corpus / *that* contract mismatch); this experiment goes the
  *other* direction (rewrite SFT to match contract).
- **status**: spec only; depends on EXP-018a landing first (contract must be
  defined to know what to rewrite to). Engineering: ~3-5 days.
- **touches reward**: no.

### EXP-018c — First clean verification run under unified contract

- **hypothesis**: with EXP-018a (unified evaluator) + EXP-018b
  (contract-aligned SFT) in place, a 50-step Stage-1 GRPO with G=2,
  β=0, max_turns=1, max_completion=2048, v3-symbol-shaped reward,
  on a 3-5 task subset of the unified task pool, produces a
  `mean_reward` curve whose last 10 steps' mean is strictly greater
  than its first 10 steps' mean.
- **variable changed**: this is NOT a single-change experiment; it
  is a verification run on top of a composite infrastructure change
  (EXP-018a + EXP-018b). The "single change" framing does not apply
  to verification spikes — flag explicitly in run notes.
- **duplicate check**: superficially resembles EXP-016p-spike
  (`research/experiments/EXP-016p-spike.md`), which was the prior
  attempt at this verification gate. Differences:
  - EXP-016p used regex blocklist as a stopgap anti-hack; this run
    uses the *structural* extern "C" contract.
  - EXP-016p kept the Sakana 400 torch:: SFT; this run uses the
    rebuilt extern "C" SFT.
  - EXP-016p targeted `sv_compile > 1.05`; this run targets the
    weaker but realistic `mean_reward last10 > first10` gate.
  EXP-016p is **superseded** by EXP-018c — mark its status accordingly.
- **pass criteria** (mechanically computable, no judgment calls):
  1. `mean(reward[40:50]) - mean(reward[0:10]) > 0.15` (covers
     transition from compile-fail dominated to compile-OK dominated)
  2. `scipy.stats.linregress(range(50), per_step_mean_reward).slope > 0`
     with `p_value < 0.05`
  3. ≥ 1 unique rollout with `correct=True` AND its `.cu` source
     passes a post-hoc deep anti-hack scan (script TBD as part of
     EXP-018a deliverables — must include source-level header
     blocklist, torch:: API call grep, AND `nm -D` symbol scan;
     specify the script's path in run notes)
- **fail criteria**: any of (1), (2), (3) does not hold
- **status**: blocked on EXP-018a + EXP-018b. Run estimate: 16h
  H200 (2 × 8h slurm with auto-resume), eval_backend=local
  (verification-phase, not promotion-eligible).
- **touches reward**: no.

### EXP-018e — Promotion-eligible repeat on Northflank A100 (renumbered from 018d on 2026-08-10)

Note (2026-08-10): the EXP-018d identifier was consumed on 2026-08-03 by the unrelated prompt-migration bugfix (research/experiments/EXP-018d-contract-migration.md, commit fc6c8f1); this promotion rerun is renumbered EXP-018e to keep ledger references unambiguous.

- **hypothesis**: the EXP-018c result reproduces on Northflank-managed
  CoreWeave A100 (not H200 local subprocess) under identical config,
  with `eval_backend=coreweave` and identical task pool.
- **variable changed**: `KERNELFORGE_EVAL_BACKEND=coreweave` instead of
  `local`. All other config identical to EXP-018c.
- **duplicate check**: docs/CLAUDE.md "What Still Needs Real Runtime
  Validation" lists this as a never-validated requirement. PRD §0
  Locked Decisions: "any reward involving speedup/runtime MUST execute
  on A100" — this is the first time the project actually enforces
  that constraint on a non-broken eval path.
- **status**: blocked on EXP-018c PASSING. If 018c FAILS, this
  experiment is moot. Run estimate: 8h H200 train + ~$300 A100 eval
  cash. Hard prerequisite: Northflank service deployment has been
  validated end-to-end at least once before this run launches.
- **touches reward**: no (reward function unchanged from EXP-018c).

## Hypotheses ruled out (link to do-not-repeat.md entries)

- **2026-05-28 — Sakana SFT + ops6k torch-extension eval path is a
  reward-hack channel via torch:: C++ API** — this campaign is the
  structural fix path that entry calls for. Cannot be re-litigated.
- **2026-05-28 — Project history has NEVER produced a multi-step
  monotone mean_reward training curve** — this campaign is the first
  serious attempt to settle that question on a non-broken substrate.
- **2026-05-17 — doubleGraph WCC SFT for ops6k Tensor-return tasks**
  — this campaign goes the *opposite* direction (rewrite SFT to match
  contract), so does not re-attempt that path.
- **2026-05-18 — Stage 1 / Stage 3 with
  `KERNELFORGE_STAGE1_MAX_COMPLETION_LENGTH <= 1024`** — campaign
  uses 2048 default.
- **2026-05-18 — Probes that rely on `[ROLLOUT]` debug from
  `multi_turn_rollout.py`** — campaign uses only `[ROLLOUT_INLINE]`
  and `[VERIFIER_INLINE]` tags on the active `reward_from_env` path.

## Current leader within campaign

None — campaign is in spec phase. First leader will emerge from
EXP-018c run.

## What success at the campaign level looks like

If EXP-018a → 018b → 018c PASSES → 018e PASSES, the user's stated
design innovations (I1, I2, I4, I5, I6 in their extern-"C" form) have
been demonstrated to produce a non-flat training curve on real A100
timing under a closed-channel evaluator. This is the first
empirically defensible "走通" of the user's design in project history.
At that point, the user's broader design plan (Stage-2 RFT + Stage-3
GRPO + curriculum + SkyDiscover hedge) is unblocked and individual
ablations can begin under the standard single-change gate.

## What failure at the campaign level means

If EXP-018c FAILS under all reasonable sub-variations (3-5 task subset,
50 steps, G=2, β=0, v3-symbol-shaped reward, Sakana-rewritten SFT,
extern "C" contract), then the user's design's central claim — "the
three-prior + symbol-shaped + anti-hack stack produces an RL signal
the policy can follow" — is empirically falsified at the structure
level (no infra excuse remaining). At that point the user must
either modify the design's claim or pivot to a different setup.
Failure is therefore informative, not wasted.
