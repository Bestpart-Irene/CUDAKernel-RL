# KernelForge Research Notes

Durable narrative for KernelForge experiments. Append, do not rewrite.

`memory-keeper` is the only agent that edits this file. Workers and planners
hand structured summaries to `memory-keeper` after a managed run completes.

> **2026-05-18 INVALIDATION BANNER** — every numeric result narrated below
> dated **before 2026-05-18** was produced on a Stage 2 SFT corpus
> (`datasets/doublegraph_sft.jsonl`) whose 4 WCC entries (indices 76-79)
> never exposed the verifier's `extern "C" void wcc_kernel(...)` contract.
> Stage 1/3 runs warm-started from `outputs/kernelforge-stage2/checkpoint-*`
> inherit that broken prior. Treat the narratives below as **diagnostic
> history**, not as comparable baselines. See `do-not-repeat.md` entry
> 2026-05-18 for the full evidence chain. The pivot is
> `scripts/build_wcc_sft_replacements.py` which regenerates the 4 affected
> rows; everything after a fresh Stage 2 SFT on the patched corpus is the
> new comparability anchor.

## Conventions

### Cold-start promotion (master = null)

When `research/live/master.json` has `hash=null`, the first managed run
that produces parseable `mean_reward` and `pass_rate` is **unconditionally
promoted** to master. That run writes its commit SHA into `hash`,
`parent_master_hash=null` in the ledger row, and `promote=true`.

### Promotion gate (master != null)

A subsequent run promotes only when it satisfies ALL of:

1. `mean_reward` strictly greater than master.
2. `pass_rate` not regressed (>=) versus master.
3. `eval_split`, `seed_count`, `max_turns`, `reward_version`, and
   `eval_backend` all match master.

Ties or `mean_reward` equal-to-master do not promote.

### Backend lock

`eval_backend` is part of the comparability contract. A run measured on
`coreweave` (FastAPI on A100) cannot be promoted against a master measured
on `local` (in-process eval), and vice versa. Switching backend re-opens
cold-start.

### Verification-phase runs

A run is **verification-phase** when its goal is to test a binary
hypothesis (does X unlock signal? does Y produce gradient?) rather than
to produce a master-promotion candidate. Verification runs are marked
explicitly in three places:

1. Filename: `research/experiments/EXP-NNN-verification-<context>.md`
2. Header banner in the spec: "⚠️ VERIFICATION PHASE — NOT
   MASTER-PROMOTION-ELIGIBLE."
3. `results.tsv` row: `experiment_id` suffix `-verify`, `promote=false`,
   `comment` begins with `VERIFICATION ONLY — ...`.

Verification runs **never** write `research/live/master.json`, regardless
of metric values. They may run on non-master-comparable backends (e.g.
`local` H200 eval while master is locked to `coreweave` A100). A passing
verification result triggers a follow-up production run (same config,
correct backend) which IS master-promotion-eligible.

This convention exists because the A100 eval queue at NU Explorer is
unreliable, and we don't want infrastructure availability to block
testing scientific hypotheses about the RL pipeline.

## Format Per Entry

```
## <YYYY-MM-DD> — <experiment-id> — <one-line title>

- hypothesis: <one sentence>
- parent master hash: <hash>
- variable changed: <file> :: <symbol> :: <before> → <after>
- runner / job id: <modal | northflank> / <job-id>
- metrics: mean_reward=<x> pass_rate=<x> speedup_vs_orig=<x> fast_p=<x>
- decision: <promote | no-promote> — <one-line reason>
- interpretation: <one short paragraph>
```

## Entries

## 2026-05-17 — EXP-001 — stage1 raw GRPO baseline (no SFT)

- hypothesis: base Qwen3-Coder-30B-A3B-Instruct + Stage 1 GRPO on 19 mixed
  (15 ops6k + 4 wcc) tasks can produce *any* reward variance.
- parent master hash: null
- variable changed: none (baseline)
- runner / job id: explorer-h200 / slurm 6858988 (scancel @ step 13)
- metrics: reward=-1, reward_std=0, grad_norm=0, clipped_ratio=1.0,
  mean_terminated_length=0, entropy collapsed 0.15
- decision: no-promote — cold-start exploration failure
- interpretation: base model emits zero parseable ```cuda blocks in 1024
  tokens. Every G=2 group has zero reward variance → advantage=0 →
  grad_norm=0 → policy never moves. The failure family is "Stage 1 GRPO
  from raw base on sparse-binary outcome reward". Confirmed in `do-not-repeat`.

## 2026-05-17 — EXP-002 — stage2 SFT on doubleGraph 192

- hypothesis: SFT on 192 doubleGraph A100 expert kernels produces a
  converged checkpoint usable as Stage 1 warm-start.
- parent master hash: null
- variable changed: pipeline order, doubleGraph kernels as SFT corpus.
- runner / job id: explorer-h200 / slurm 6868569 (FAILED exit 120 post-save
  at step 50 — bug in TRL ≥0.29 post-save hooks; ckpt-50 intact)
- metrics: train/loss 2.5→0.5 over 50 steps, grad_norm 0.17-0.31,
  mean_token_accuracy 0.87
- decision: no-promote — SFT-only run; produces ckpt artifact, not RL metrics.
- interpretation: SFT mechanism works. Adapter `adapter_model.safetensors`
  (3.38 GB LoRA) saved at `outputs/kernelforge-stage2/checkpoint-50/`.
  Subsequent Stage 1 runs (EXP-003 v1/v2/v3) all warm-start from this.

## 2026-05-17 — EXP-003-v1 — stage1 GRPO + doubleGraph ckpt + ops6k+wcc

- hypothesis: doubleGraph-SFT'd model unlocks GRPO reward signal on the
  mixed Stage 1 task pool (15 ops6k + 4 wcc).
- parent master hash: null
- variable changed: KERNELFORGE_STAGE1_INIT_CKPT = outputs/kernelforge-stage2/checkpoint-50
- runner / job id: explorer-h200 / slurm 6869808 (3 steps)
- metrics: reward=-1, reward_std=0, grad_norm=0
- decision: no-promote — contract domain mismatch
- interpretation: doubleGraph SFT teaches WCC void contract
  `void wcc_kernel(row_ptr, col_idx, num_vertices, labels)`. ops6k tasks
  require `torch::Tensor run_kernel(torch::Tensor)`. EXP-004 debug
  (slurm 6870504) showed model writes `void run_kernel(output, ...)` —
  the F1 prompt hardcode (committed then reverted) could not override
  the SFT bias. Family ruled out: "doubleGraph WCC void-contract SFT
  for ops6k Tensor-return tasks" regardless of prompt engineering.

## 2026-05-17 — EXP-003-v2 — stage1 GRPO + sakana 200 ckpt + ops6k+wcc

- hypothesis: Sakana 200 SFT (ops6k contract, AI-CUDA-Engineer-Archive)
  unlocks ops6k reward signal.
- parent master hash: null
- variable changed: SFT corpus + ckpt path
- runner / job id: explorer-h200 / slurm 6872320 (4 steps)
- metrics: reward=-1, reward_std=0, grad_norm=0, clipped_ratio=1.0,
  mean_terminated_length=0
- decision: no-promote — output-length blow-up
- interpretation: Sakana 200 corpus has assistant CUDA bodies averaging
  ~4000 chars (~1200 tokens). SFT'd model reproduces that length under
  sampling. Stage 1 max_completion_length=1024 cuts every generation.
  Family ruled out: "Sakana CUDA Engineer SFT without a token-length
  filter at max_completion ≤1024". Re-attempt only after filtering to
  <800 tokens (D4-pending).

## 2026-05-17 — EXP-003-v3 (D6) — stage1 GRPO + doubleGraph ckpt + wcc-only

- hypothesis: aligning Stage 1 task domain to SFT contract (filter to 4
  WCC tasks) unlocks reward signal because doubleGraph SFT teaches WCC
  void contract end-to-end.
- parent master hash: null
- variable changed: KERNELFORGE_STAGE1_BACKEND_FILTER=wcc
- runner / job id: explorer-h200 / slurm 6872638 (6 steps, walltime'd)
- metrics: reward=-1, reward_std=0, grad_norm=0, clipped_ratio=0.5,
  mean_terminated_length=772 (model produces well-formed code at length)
- decision: no-promote — initial diagnosis "SFT teaches format but not
  algorithm correctness" was **incomplete**.
- interpretation: EXP-004 v3 follow-up (slurm 6876541) exposed 4
  infrastructure bugs that were silently masking the real story:
  - slurm did not `module load cuda/12.8.0` → eval_core nvcc subprocess
    raised FileNotFoundError on every rollout
  - `_local_compile_check` swallowed FileNotFoundError and returned
    "compile succeeded" — lying to the rollout loop
  - `extract_cuda_code` regex left the opening ``` fence when closing
    fence was truncated by max_new_tokens
  - Stage 2 left empty `outputs/kernelforge-stage1/` mistaken for a ckpt
  All four fixed in commits 5b5db65, 75e34f1, e115cf6, ed628a9.
  D6's reward=-1 verdict thus **cannot** be attributed to "SFT can't
  teach correctness" — most of the -1 was infra. Hypothesis needs re-test.

## 2026-05-17 — EXP-004 v1/v2/v3 — single-rollout failure-bucket probes

- hypothesis: identify which bucket inside compute_reward's -1 branch
  (compile_failed / compiled_but_wrong / anti_hack / eval_crash) is
  firing for each of the 4 WCC tasks.
- parent master hash: null
- variable changed: scripts/debug_eval_pipeline.py (4 WCC tasks × 1 rollout,
  no GRPO training)
- runner / job id: explorer-h200 / slurm 6870504 (v1), 6870925 (v2),
  6874203 (v2-buggy), 6875257 (v2 post-nvcc-fix), 6876541 (v3 post-fence-fix)
- metrics (v3 final): compile_failed=2 (real syntax errors), compiled_but_wrong=2
- decision: diagnostic — produced the smoking gun on infra bugs above.
- interpretation: this was the highest-leverage hour of the night.
  Single-rollout probes exposed cascading infra bugs that 5 prior GRPO
  runs had attributed to "RL can't learn". Generalizable rule:
  **before pinning a reward=-1 finding on RL/SFT science, dispatch a
  single-rollout debug that prints extract/compile/eval verdicts**.
  Captured in `do-not-repeat`.

## 2026-05-17 — EXP-005 / v4 / v6 — shaped reward and GRPO knob A/B/C

- hypothesis: with infra fixed (commits ed628a9, 75e34f1, b7c5bc0),
  shaped reward (v2-shaped: compiled_but_wrong=0.0) and/or larger G + no
  KL pin unlocks reward_std>0.
- parent master hash: null
- variable changed (3-way A/B/C):
  - v5 (slurm 6880035): reward v2-shaped, G=2, beta=0.04 default
  - v4 (slurm 6880846): reward v1-discrete-milestone, G=2, beta=0.04
  - v6 (slurm 6881355): reward v2-shaped, G=4, beta=0.0
- runner / job id: explorer-h200 × 3 parallel
- metrics: ALL three runs — reward=-1 unique, reward_std=0, grad_norm=0
  across 50/51 steps each (walltime'd). Total ~1200+ rollouts.
- decision: no-promote — the entire 3-way A/B/C produced identical
  reward distribution. Either v2-shaped is not reaching the GRPO reward
  path, or the downstream chain rejects every completion uniformly.
- interpretation: this is the puzzle. EXP-004 v3 single-rollout showed
  2/4 tasks hit compiled_but_wrong, which under v2-shaped should produce
  reward=0.0 in v5. Instead all 200+ v5 rollouts return reward=-1. Two
  candidate explanations:
  - (Q1) v2-shaped does not reach evaluate_code_remote's reward
    assignment in the GRPO context (module caching / wrong path)
  - (Q2) The GRPO multi-turn rollout context produces different model
    outputs than the debug script, and all 200 rollouts genuinely fail
    extract/compile before reaching eval
  EXP-S4 (debug_eval rerun on v2-shaped default) and EXP-D1 (wandb
  detailed metrics) both run today address Q1/Q2 respectively.

## 2026-05-17 — EXP-D1 — 2-step verbose stage1 + rollout_debug

- hypothesis: per-rollout stdout prints expose which bucket (extract /
  compile / eval) is firing at the GRPO multi-turn level.
- parent master hash: null
- variable changed: KERNELFORGE_ROLLOUT_DEBUG=1, max_steps=2 (env)
- runner / job id: explorer-h200 / slurm 6885765 (2 steps complete)
- metrics: ROLLOUT_DEBUG env var did NOT propagate (no `[ROLLOUT` lines
  in stdout) — separate bug. But wandb metric dict revealed:
  - step 1: clipped_ratio=0.25, mean_terminated_length=706 (75% complete!)
  - step 2: clipped_ratio=1.0, mean_terminated_length=0 (full degeneration)
- decision: diagnostic — KEY finding for the open puzzle.
- interpretation: TWO independent findings:
  1. **Multi-turn feedback degrades context budget**. Turn N's error
     feedback append makes Turn N+1's effective max_completion shrink.
     By step 2 every completion is truncated. Family-level rule: at
     max_turns=3 with feedback-append, this run config self-poisons
     across steps. Mitigations: lower max_turns or refactor feedback to
     replace-not-append.
  2. **75% complete completions in step 1 also return reward=-1**.
     This rules out "truncation is the sole bottleneck". Even when the
     model writes well-formed code that naturally EOSes, reward stays
     -1. This forces investigation into the downstream chain (extract
     / compile / eval correctness) — not into max_completion bumps,
     not into reward shape alone.

## 2026-05-17 — EXP-S4 — single-rollout reward-version verification

- hypothesis: confirm v2-shaped reward reaches the evaluate_code_remote
  reward path; expected reward_in_result=0.0 for compiled_but_wrong.
- parent master hash: null
- variable changed: rerun debug_eval_pipeline.py with v2-shaped active
- runner / job id: explorer-h200 / slurm 6885946 (completed 15:30)
- metrics: Bucket counts: compile_failed=4, compiled_but_wrong=0,
  anti_hack=0, eval_crash=0 — different bucket distribution than v3
  because 4/4 hit REAL compile failures (no rollout reached eval).
- decision: diagnostic — exposed two genuine compile failures the
  earlier infra fixes had not caught.
- interpretation: 4/4 WCC rollouts hit GENUINE nvcc compile errors. Two
  causes:
  - 2/4 (tasks 0 & 2): "__host__ or __device__ annotation on lambda
    requires --extended-lambda nvcc flag". Model writes device lambdas
    (idiomatic for union-find / path-compression in WCC kernels) but
    both `_local_compile_check` and `eval_core._nvcc_command` lacked
    `--extended-lambda`. Fix landed in commit 5c4b63d (added flag to
    both default command lists).
  - 2/4 (tasks 1 & 3): "expected a ';'" — truncation at
    `max_completion_length=1024`. Code body cut mid-statement.
  Family ruled out: trusting earlier infra-fix coverage was sufficient.
  Real-output diagnostics (single-rollout) keep finding new
  infrastructure gaps even after our explicit fix list. Generalization:
  ship the cheap-kill diagnostic on every cycle, not just the first.

## 2026-05-17 — EXP-008 E — max_turns=1 isolation

- hypothesis: removing multi-turn feedback (max_turns=3 → 1) eliminates
  the step-2+ context-budget collapse from D1 and unlocks reward signal.
- parent master hash: null
- variable changed: KERNELFORGE_STAGE1_MAX_TURNS=1
- runner / job id: explorer-h200 / slurm 6886191 (completed all 5 steps)
- metrics: reward=-1 throughout, reward_std=0, grad_norm=0,
  clipped_ratio=0 by step 5 (100% natural EOS — multi-turn collapse
  fully eliminated as predicted)
- decision: no-promote — but the conceptual ruling is rich.
- interpretation: **Multi-turn feedback collapse is FIXED** by
  max_turns=1 (clipped_ratio stayed healthy across all 5 steps, no
  step-2+ degradation pattern from D1 reproduced). **BUT reward stayed
  -1 throughout 5 steps × G=2 × ~10 rollouts = ~50 rollouts.** Family
  ruled out: multi-turn feedback is NOT the primary cause of binary
  reward collapse — it is a secondary symptom. The primary cause is
  downstream of generation (extract / compile / eval correctness),
  unaffected by max_turns.

## 2026-05-17 — EXP-008 C / F (FAILED at TRL config validation)

- hypothesis (C): GSPO sequence-level loss unblocks Qwen3 MoE GRPO
  instability (researcher #3 contingency).
- hypothesis (F): G=2 below DAPO/Kevin/Dr.GRPO floor; G=8 in isolation.
- runner / job id: slurm 6886192 (C), 6886282 (F). Both FAILED at
  TRL init validator before any training.
- failures:
  - C: `ValueError: Unknown loss type: gspo`. TRL 0.29.0 loss_type enum
    is {grpo, dr_grpo, dapo, bnpo}. GSPO not available in this TRL.
  - F: `ValueError: generation_batch_size (4) must be divisible by
    num_generations (8)`. With default grad_accum=4, G must divide 4
    cleanly (1, 2, 4).
- decision: failed-at-config — but each failure is itself informative.
  Capture in do-not-repeat so no future planner re-proposes these
  exact configs. Mitigation cost is small (loss_type retry: vanilla
  grpo; G retry: G=4 isolation).

## 2026-05-17 — EXP-008 F-retry / C-retry — G=4 / vanilla grpo

- hypothesis (F-retry): G=4 isolation (divides 4 cleanly) on top of
  the same shaped reward and doubleGraph SFT D6 baseline.
- hypothesis (C-retry): loss_type="grpo" (vanilla GRPO, NOT the default
  "dapo" we have been on all night).
- runner / job id: slurm 6886378 (F-retry), 6886379 (C-retry). Both
  reached step 4 of 10 then FAILED (exit 1, ~1.5h elapsed each).
  Probable cause: OOM at step 4 with max_completion=2048 + G=4
  (F-retry) or vanilla-grpo memory profile (C-retry).
- metrics (last reported step):
  - F-retry: step 4 reward=-1 std=0 grad=0 clipped=0.25
  - C-retry: step 4 reward=-1 std=0 grad=0 clipped=0.25
- decision: no-promote — both confirm partial picture.
- interpretation: **None of {G=2, G=4} × {dapo, vanilla grpo} produced
  any reward != -1**. Researcher contingencies #1 and #2 (G scaling +
  loss type) both ruled out as primary blockers — the same -1 collapse
  appears regardless of GRPO knob. Pattern matches EXP-008 E
  conclusion: the bottleneck is downstream of GRPO param choices.

## 2026-05-17 — EXP-008 D / D-retry — anti-hack verifier_msg diagnostic

- hypothesis: anti-hack runtime checks may be false-rejecting correct
  kernels (researcher #4 contingency, Sakana arXiv 2509.14279 paper
  documents 3.13x → 1.49x speedup reduction from over-strict
  verification).
- runner / job id: slurm 6886283 (D, scancelled — verifier env did not
  propagate), slurm 6886408 (D-retry, after slurm explicit `export`
  fix in commit 2efe179, completed all 3 steps)
- metrics: D-retry reward=-1 std=0 grad=0 clipped=0.25 across 3 steps;
  **zero `[VERIFIER prompt=...]` lines in stdout** — env var STILL
  not visible to Python after slurm explicit export.
- decision: diagnostic — failed to produce diagnostic data twice. The
  env propagation bug is now the third confirmed instance (after
  ROLLOUT_DEBUG and the original VERIFIER_DEBUG) and survives the
  "explicit export inside slurm body" fix that should have worked.
- interpretation: the anti-hack hypothesis (researcher #4) **remains
  untested**. The diagnostic infra is currently broken. Fixing requires
  ssh-into-running-job inspection or a minimal repro to find why
  `os.getenv("KERNELFORGE_VERIFIER_DEBUG", "0")` returns "0" even after
  bash `export KERNELFORGE_VERIFIER_DEBUG=1` happened in the same
  slurm shell.

## 2026-05-17 — Conceptual summary at end-of-EXP-008

By the end of EXP-008 sub-experiments E/F/C/D and their retries, **the
GRPO parameter space is empirically ruled out as the bottleneck for the
binary reward=-1 collapse**:

- multi-turn (max_turns 3 → 1): same reward=-1
- G size (2 → 4): same reward=-1
- loss type (dapo default → grpo vanilla): same reward=-1
- reward shape (v1-binary → v2-shaped, ROLLED INTO every EXP-008 run):
  same reward=-1
- infra layer (11 cumulative bugs across the day): all fixed, still -1

Remaining hypotheses to test are all STRUCTURAL, not knob-level:

| candidate | why we did not eliminate tonight |
|---|---|
| anti-hack false negatives | diagnostic infra (env var prop) broken |
| SFT corpus too small (192 vs 2K floor) | structural, multi-day |
| WCC task pool too hard for this model | needs ops6k domain switch |
| eval_core correctness tolerance too strict | touches frozen file |
| LoRA capacity (r=16 on 30B-MoE) | structural |

The path forward is no longer "try another knob". It is one of:
(a) fix env prop and confirm/refute anti-hack
(b) switch to ops6k domain (path B, needs Sakana ckpt restart)
(c) pivot to SkyDiscover evolutionary search (PRD already implements)
(d) structural scale-up of SFT corpus + maybe base model

## 2026-05-18 — EXP-009-B — FIRST NON-(-1) SIGNAL: max_completion_length 1024 -> 2048

- hypothesis: bumping `KERNELFORGE_STAGE1_MAX_COMPLETION_LENGTH` from 1024
  to 2048 lets the doubleGraph-SFT'd model finish kernel source before
  hitting max_new_tokens, unlocking non-(-1) reward on WCC-only Stage 1.
- parent master hash: 117bb5d
- variable changed: env :: `KERNELFORGE_STAGE1_MAX_COMPLETION_LENGTH` ::
  1024 -> 2048 (only change vs the EXP-008-E baseline)
- runner / job id: explorer-h200 / slurm 6888764 (node d4054, gpu-short,
  walltime 22m13s, 1 step)
- config: Stage 1 GRPO warmup, 1 step, G=2, beta=0.0, max_turns=1,
  reward_version=v2-shaped, backend_filter=wcc, eval_backend=local,
  INIT_CKPT=outputs/kernelforge-stage2/checkpoint-50
- metrics:
  - compile rate 0/8 -> 6/7 (~86%)
  - reward_mean = -0.25 (was -1.0)
  - reward_std = 0.5 (was 0.0)
  - grad_norm = 0.046 (was 0.0)
  - clipped_ratio = 0.50 (was 0.75-1.0)
  - mean_terminated_length = 1120 (was 638-755)
  - 0/7 correct=True (every compiled rollout failed correctness)
- decision: no-promote — first non-(-1) signal in project history but
  zero correctness; promotion gate requires correct=True.
- interpretation: **the bottleneck for the entire EXP-001..EXP-008
  reward=-1 collapse was kernel completions being truncated mid-source,
  not reward shape, not multi-turn collapse, not GRPO knobs, not
  anti-hack**. All earlier hypotheses (v2-shaped reward, bypass removal,
  TRLOO/DAPO loss type, G scaling, max_turns=1) were correct fixes but
  not the bottleneck. With kernels allowed to finish, 86% compile, and
  reward variance + non-zero gradient + non-zero advantage now exist for
  the first time. Next gates: bring at least one rollout to correct=True
  (then re-open cold-start promotion), and confirm the trend reproduces
  for >1 step.

## 2026-05-18 — TRL 0.29 `rollout_func` is dead code — multi-turn silently inactive

- finding: TRL 0.29 GRPOTrainer no longer invokes the user-provided
  `rollout_func` during training. In job 6888764 the construction tag
  `[ROLLOUT_FACTORY] make_multi_turn_rollout built rollout_func` fired
  once at init, but the call-site tag `[ROLLOUT_CALL] rollout_func
  invoked` fired ZERO times across both `.out` and `.err`. TRL emits at
  trainer init: `'rollout_func' is an experimental feature. This API
  may change or be removed at any time without prior notice.`
- effective consequence: every Stage 1 and Stage 3 run in this codebase
  has been **single-turn** regardless of `KERNELFORGE_STAGE1_MAX_TURNS`.
  The multi-turn feedback loop in `training/multi_turn_rollout.py:198-330`
  (the rollout_func body) is dead. `max_turns=3` and `max_turns=1` were
  empirically indistinguishable for a reason: they pointed at code TRL
  never called.
- real reward path: TRL drives generation single-turn, then calls
  `reward_from_env` (which IS invoked). Inside `reward_from_env`, the
  `env_reward len=0` branch inline-evaluates each completion via local
  compile + remote/local eval. This is the production reward path.
- side effects:
  - The B5 hypothesis ("multi-turn attribution bug — concat prompt_ids
    across turns") cannot bite because the multi-turn loop never runs.
  - The EXP-D1 "multi-turn feedback append at max_turns=3 self-poisons
    context budget" finding still holds as a *theoretical* concern but
    was not the active failure mode under TRL 0.29; the apparent step-2
    clipped_ratio=1.0 in D1 must be re-explained as a TRL-internal
    rollout artifact, not as our multi-turn feedback collapsing.
  - Any future probe depending on `[ROLLOUT_*]` debug tags inside
    `multi_turn_rollout.py` will produce zero output and is unusable as
    a diagnostic.
- mitigation already shipped: commit 469f6a4 relocates
  `KERNELFORGE_ROLLOUT_DEBUG` and `KERNELFORGE_VERIFIER_DEBUG` print
  gates to the active `reward_from_env` path with new tags
  `[ROLLOUT_INLINE]` / `[VERIFIER_INLINE]`. Use these going forward.
- evidence: slurm 6888764 stdout/stderr — single `[ROLLOUT_FACTORY]`
  line, zero `[ROLLOUT_CALL]` lines.

## 2026-05-27 — EXP-015'-A — FIRST correct=True IN PROJECT HISTORY (cold-start master promotion)

- hypothesis: stacking three independently-shipped fixes — subprocess
  isolation for CUDA eval (commit 3b28c36), skipping `_local_compile_check`
  pre-check for ops6k-shaped code (commit 61af7f5), and ABI-axis annotations
  injected into all 8 WCC SFT rows (commit 8989ac4) — unlocks the first
  `correct=True` signal on Stage 1 GRPO over ops6k tasks.
- parent master hash: null (cold-start)
- variable changed: three stacked fixes (NOT a single-variable run, but
  cold-start admits the first parseable mean_reward + pass_rate run
  unconditionally per the convention banner above):
  - `eval_backend` :: in-process eval :: subprocess-isolated CUDA execution
    (commit 3b28c36)
  - `_local_compile_check` :: pre-check on ops6k-shaped code :: skipped
    (commit 61af7f5)
  - `datasets/doublegraph_sft.jsonl` :: WCC entries without ABI annotation
    :: 8 rows annotated with ABI-axis (commit 8989ac4)
- runner / job id: explorer-h200 / slurm 7040100 (COMPLETED, walltime
  6:30:16, max_steps=20 all completed)
- config: Stage 1 GRPO, 20 steps, ops6k task pool, eval_backend=local
  (subprocess), SFT prior = Sakana 400 (commit 51cb391)
- metrics:
  - 152 rollouts total
  - bucket distribution: compile_fail=96 (63%), compile_OK_wrong=46 (30%),
    **correct=10 (7%) — FIRST IN PROJECT HISTORY**
  - mean_reward ≈ -0.86 (estimated from bucket mix under v2-shaped:
    compile_fail -> -1, compile_OK_wrong -> 0, correct -> +1; with 7%
    correct and 30% zero-shaped wrong, weighted mean ≈ -0.86)
  - pass_rate = 10/152 = 0.066
  - Wilson 95% CI on pass_rate: [3.6%, 11.7%]
  - failure taxonomy: compile_fail_extension_build=38 (48%),
    wrong_other=14 (18%), compile_fail_other=14 (18%),
    **wrong_pybind_signature=7 (9%) — was 31% in EXP-014, 3.4x reduction**,
    correct=5 (6%, ROLLOUT_INLINE count), wrong_numerical_mismatch=2 (2%)
- decision: **promote — cold-start unconditional promotion**. Master was
  `hash=null`; per the convention banner, the first managed run producing
  parseable `mean_reward` and `pass_rate` is unconditionally promoted.
  EXP-015'-A is the first such run, and additionally crosses the
  historically-unreached `correct=True` boundary.
- interpretation: After EXP-009-B unlocked compile-rate signal but
  produced zero correct rollouts, three orthogonal fixes were needed to
  cross the correctness boundary: (1) subprocess isolation prevents CUDA
  context corruption from poisoning subsequent rollouts in the same
  process; (2) skipping pre-check on ops6k-shaped extension code stops
  spurious rejection of well-formed kernels at the host-Python layer;
  (3) ABI-axis annotations in the SFT corpus teach the model the
  signature convention the verifier expects, collapsing the
  `wrong_pybind_signature` bucket from 31% (EXP-014) to 9% — a 3.4x
  reduction — and freeing rollouts to compete on numerical correctness.
  7% correct (Wilson lower bound 3.6%) is a **phase transition
  signature**, not a stable steady-state rate. EXP-016 (24-step
  extension) will confirm whether the slope continues upward and provide
  a tighter sample for re-promotion against a real (non-null) master.

> **Note**: master was promoted to EXP-015'-A on 2026-05-27. The
> promotion documents the cold-start unblock; the next promotion gate
> (EXP-016 or successor) is the standard non-null-master gate: strictly
> greater `mean_reward`, non-regressed `pass_rate`, matching
> eval_split / seed_count / max_turns / reward_version / eval_backend.

## 2026-05-28 — EXP-015'-A promotion INVALIDATED; master rolled back to null

- finding: all 5 unique `correct=True` rollouts in EXP-015'-A were
  reward-hacked via direct `torch::` C++ API calls from inside the
  candidate `.cu` source. The ops6k evaluation path
  (`eval_service/eval_core.py::evaluate_ops6k_kernel_impl`, L588) does
  not call `scan_forbidden_symbols`; the candidate is built via
  `torch.utils.cpp_extension.load(..., with_cuda=True)` which links
  libtorch and makes `#include <torch/extension.h>` + `torch::relu(x)`
  / `x.softmax(-1)` style delegation a valid compile path. The Sakana
  400 SFT corpus demonstrates this style on 400/400 rows, so the SFT
  prior actively biases the model toward the hack. Anti-hack runtime
  checks (not_constant / not_passthrough / not_noop / shapes_match)
  are blind to this attack because the delegated torch op produces a
  real, input-dependent, non-trivial output.
- corroborating evidence: 4 of 5 correct rollouts had `sv_eager < 1.0`
  (0.17, 0.77, 0.04, 0.77), meaning the candidate was 6-23x slower
  than eager PyTorch — exactly the overhead profile of pybind + torch
  op dispatch wrapped inside a custom extension. A hand-written CUDA
  kernel on simple ops should not be 23x slower than eager.
- audit also revealed that the original promotion record's
  `rollouts_total=152` and `correct=10` are double-counts:
  `compute_reward` is called twice per rollout (once inside
  `evaluate_code_remote::compute_task_reward` at task_support.py:311,
  once inside `reward_from_env::_compute_reward_from_result` at
  multi_turn_rollout.py:454) and the COMPUTE_REWARD print emits twice.
  Real counts: 80 unique rollouts, 5 unique correct=True. The pass_rate
  ratio (~6%) is unchanged but the sample size is half what was
  recorded.
- actions taken:
  - `research/live/master.json` rolled back to `hash=null`. Cold-start
    gate re-opened.
  - `research/results.tsv` row renamed `EXP-015'-A-INVALIDATED`,
    `promote=false`. Reward_version corrected from `v2-shaped` to
    `v3-symbol-shaped` (log evidence: every `[COMPUTE_REWARD]` line in
    `logs/kf_stage1_7040100.out` carries `version=v3-symbol-shaped`;
    the original record's `v2-shaped` was incorrect).
  - `research/do-not-repeat.md` entry 2026-05-28 documents the failure
    family and notes that the naive fix (wire `scan_forbidden_symbols`
    into the ops6k path) does NOT close the hole — every torch
    extension links libtorch, so `nm -D` would false-positive every
    legitimate kernel.
- next: the path forward is gated by `EXP-016p-spike`
  (`research/experiments/EXP-016p-spike.md`), a verification-only run
  that tests whether the GRPO pipeline produces a multi-step
  monotone-trend `mean_reward` curve under a patched evaluator on the
  simplest possible ops6k task pool. Project history has never
  produced such a curve. Until the spike either passes or fails, no
  further investment is authorized.

## 2026-06-01 — EXP-018c-v1 — first extern-C verification attempt (VOID; retro-recorded 2026-08-10)

- hypothesis: (CAMPAIGN-018 EXP-018c) 50-step Stage-1 GRPO under the
  unified extern "C" contract produces a rising mean_reward curve.
- parent master hash: null
- variable changed: verification run on composite infra (EXP-018a
  contract); NOT single-change.
- runner / job id: explorer-h200 / slurm 7359525
- metrics: none recorded — VOID
- decision: no-promote — VOID
- interpretation: launched cold-start from base
  (KERNELFORGE_STAGE1_INIT_CKPT="" in scripts/cluster/exp018c.slurm) in
  tension with do-not-repeat 2026-05-16; ran on a self-contradictory
  task pool (15/16 ops6k prompts still demanded pybind while the
  extern-C evaluator hard-rejects it — found by the 2026-08-03 audit,
  fixed in fc6c8f1/EXP-018d), so any outcome was uninterpretable and
  none was recorded at the time. Only surviving datum: ~19 min/step at
  G=2, max_completion=2048 (exp018c.slurm v2 comment), which falsifies
  the 50-step-in-8h walltime plan and itself contradicts the step-40-50
  pass criteria after MAX_STEPS was cut to 25. Retro-recorded during
  the 2026-08-10 tri-agent audit; this entry closes the ghost run.

## 2026-08-10 — tri-agent audit — full reports in research/audits/

- verdict: campaign direction sound (extern-C contract structurally
  stronger than published harnesses); EXP-018c-as-launched invalid on
  five counts: (1) cold-start violates do-not-repeat 2026-05-16 with
  the revisit condition (p(valid candidate) ≥ ~0.05) never measured;
  (2) G=2 group-std normalization erases v3-symbol-shaped magnitudes
  (any unequal pair → advantage ±1/√2) and the TRLOO correction is
  inactive under default loss_type=dapo; (3) MAX_STEPS=25 vs pass
  criteria defined over steps 40-50, no save_steps/auto-resume; (4) the
  required deep anti-hack scan script does not exist and EXP-018a's 5/5
  hacked-fixture replay acceptance was never recorded; (5) run unmarked
  as verification-phase anywhere in research/ → with master=null the
  cold-start rule would auto-promote a local-H200 verification result.
- open hack channels on the extern-C path (reviewer report):
  reference-output scraping (ref computed in same process/CUDA
  context), stateful timing gaming (timed output never re-verified),
  tolerance/fast-math gaming (rtol=atol=1e-3 plus default
  --use_fast_math), permanently fixed eval seeds; anti-hack steps
  wrapped in except:pass.
- gate for any 018c rerun: EXP-018c-p0 base-model pass@k probe (3-task
  spike pool, migrated prompts, frozen evaluator, ~1 GPU-hour) —
  p ≥ 0.05 licenses cold start per the do-not-repeat revisit condition;
  p < 0.05 makes EXP-018b (extern-C SFT rebuild) mandatory. Plus
  bookkeeping: fixture replay recorded, launcher rewritten,
  evaluator_sha/task_pool_hash added to the ledger schema.

## 2026-08-10 — overhaul Phase 1 landed (post-audit infra, no training run)

- evaluator hardened per audit finding 2: reference-output isolation,
  timed-output re-verification with urandom seed, default
  --use_fast_math dropped, anti-hack except:pass removed, device-wide
  sync timing, B200 cc parse fix. New evaluator_sha b182df91c482;
  versioned-freeze policy now in AGENTS.md.
- EXP-019 anti-hack integrity bundle recorded at
  research/experiments/EXP-019-anti-hack-integrity.md; reward_version
  unchanged (v3-symbol-shaped).
- trainer defaults: G 2→8, max_completion 1024→2048; TRLOO correction
  declared INACTIVE under TRL 0.29 (docs corrected).
- ledger schema now 19 columns — added evaluator_sha, task_pool_hash,
  init_ckpt, eval_split, seed_count, max_turns, eval_backend.
- 2 uninferable-signature ops6k rows relabeled unsupported (sticky in
  task_support) — supported pool 18 (14 ops6k + 4 wcc),
  task_pool_hash 7c9eca9b3c2d.
- holdout split v2 built: 4 held out, 14 train; CAMPAIGN-018 spike
  tasks pinned to train; training loaders exclude holdout by prompt.
- EXP-018a acceptance NEGATIVE HALF recorded — 5/5 hack fixtures
  rejected at source scan locally (positive control pending on CUDA
  host).
- old exp018c.slurm deleted; probe launcher
  scripts/cluster/exp018c_p0.slurm ready but NOT submitted.
- full test suite: 259 passed.
- decision: no run dispatched — next gate is EXP-018c-p0 probe per the
  2026-08-10 audit entry above.

## 2026-08-11 — June 018c evidence recovered from Explorer logs (three runs, not one)

- hypothesis: retro-record — the June "018c" slot, previously ledgered as a
  single VOID run (7359525), was actually THREE attempts on 2026-06-01;
  recovered Explorer logs now live at `research/audits/evidence-june-018c/`.
- parent master hash: null
- variable changed: none (evidence recovery; ledger corrections only, no new run)
- runner / job id: explorer-h200 / slurm 7359525 (v1), 7365795 (v2), 7366090 (v3)
- metrics: v1 partial (2/50 steps, batch rewards in {-1.0, -0.5}); v2 none
  (0/25 steps); v3 mean_reward=-1.0 terminal, pass_rate=0.0 (0 correct=True
  across all rollouts), 25-step series flat in [-1.0, -0.75]
- decision: no-promote — all three VOID for interpretation; results.tsv v1
  row corrected, v2/v3 rows appended.
- what the logs show:
  - **v1 (7359525)**: Ops-6K dataset load FAILED (`invalid literal for
    int() with base 10: 'trivial'`) and the launcher SILENTLY fell back to
    3 WCC prompts (`Using fallback Stage 1 prompts with live WCC evaluation
    support`) — the run never touched the intended ops6k pool. 2 of 50
    steps completed at ~19 min/step (1149 s/it) before scancel; step
    metrics WERE visible (e.g. one batch mean -0.625 std 0.25; rewards
    -0.5 = compiled-but-wcc_kernel-symbol-missing, -1.0 = compile fail).
    This corrects the 2026-06-01 entry's "no metrics recorded" and
    reassigns its self-contradictory-pool attribution to v3.
  - **v2 (7365795)**: same silent WCC fallback (its load error: `cannot
    mix list and non-list, non-null values`); scancel'd by the operator at
    17:43:51 EDT, ~4 min after start, 0/25 training steps completed.
  - **v3 (7366090)**: after commit 51eb7c9 (vector_add_e2 schema hygiene)
    the Ops-6K pool loaded (16 ops6k prompts). COMPLETED 25/25 steps in
    7h27m (train_runtime 26827s). Cold start from base
    Qwen3-Coder-30B-A3B-Instruct (INIT_CKPT empty), eval_backend=local,
    reward v3-symbol-shaped, G=2, max_completion=2048, config max_turns=3
    (TRL-0.29 dead code, effectively single-turn). Terminal step: reward
    mean -1.0, std 0.0, grad_norm 0.0; grad_norm=0.0 on 22/25 steps with
    blips 0.04-0.06 at steps 2/4/10; entropy 0.35 early → 0.25 terminal
    (peak 0.45 @step7). Late completions overwhelmingly `Source rejected:
    #include <torch/` — the cold-start base model kept emitting
    torch-extension code because the PRE-MIGRATION dataset prompts
    demanded pybind while the extern-C evaluator hard-rejects it (the
    exact contradiction EXP-018d later fixed). wandb: exp018c-7366090 /
    4ocfqywn.
- interpretation: this is the strongest empirical support for the
  2026-08-10 audit's two claims — the prompt-contradiction poison
  (Source-rejected torch code dominating rollouts) and the cold-start
  zero-gradient collapse — and it sharpens EXP-018c-p0's value: the probe
  measures whether the EXP-018d prompt migration changed what the base
  model emits. The 25-step flat series also shows the -0.5 symbol bucket
  produced occasional variance (three small grad_norm blips) but never
  gradient traction under G=2. v3 empirically reproduces the EXP-001
  cold-start collapse family under the newer reward/evaluator stack.
