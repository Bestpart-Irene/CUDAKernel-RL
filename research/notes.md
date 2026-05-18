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
