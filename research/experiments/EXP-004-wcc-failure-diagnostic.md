# EXP-004 — VERIFICATION PHASE: WCC failure-bucket diagnostic (1 rollout × 4 tasks, H200 local eval)

> **⚠️ VERIFICATION PHASE — NOT MASTER-PROMOTION-ELIGIBLE.**
> This is a diagnostic probe, not a training run. It executes one rollout
> against each of the 4 WCC-filtered Stage 1 tasks (the same subset that
> EXP-003-v3 D6 trained on) and classifies the failure mode of each.
> No GRPO step is taken, no checkpoint is saved, no metric here is
> A100-comparable. `parent_master_hash = null`.

## Background — what motivated this diagnostic

EXP-003-v3 D6 (slurm 6872638, wandb stage1-6872638) ran 6 GRPO steps with
the doubleGraph SFT ckpt-50 warm-start and
`KERNELFORGE_STAGE1_BACKEND_FILTER=wcc`. Generation health was good:

- `clipped_ratio = 0.5` (model emits tokens, not pad spam)
- `mean_terminated_length = 772` (EOS reached cleanly)
- structurally valid completions

But all 144 rollouts (6 steps × 24/step) returned `reward = -1`,
`reward_std = 0`, `grad_norm = 0`. The `compute_reward(...)` function in
`openenv_env/reward.py` collapses several distinct failure modes into a
single -1.0 return:

1. **compile_failed** — `compiles=False`. Could be signature mismatch with
   the WCC ABI, real nvcc syntax error in the generated code, or
   extraction failure (no fenced block).
2. **compiled_but_wrong** — `compiles=True, correct=False`. The kernel
   built but produced incorrect output on PAC verification graphs.
3. **anti_hack** — Dr. Kernel-style tripwire from
   `openenv_env/anti_hack.py` (passthrough / constant output / no-op
   runtime / shape mismatch). Surfaces as `error` prefix `"Anti-hack: ..."`.
4. **eval_crash** — `evaluate_code_remote` raised, or eval returned an
   error that is neither a compile message nor an anti-hack message
   (e.g., harness import error, OOM, malformed payload).

Without distinguishing these, EXP-005 might shape the wrong reward (e.g.,
reward partial-credit on compile when the actual bottleneck is correctness,
or grow the SFT corpus when the actual problem is anti-hack triggering).

## Parent master

`null`. Diagnostic only; does not produce metrics for master.json.

## Hypothesis

≥1 of the 4 WCC rollouts shows `compiles=True, correct=False`, identifying
**correctness-not-compile** as the dominant cold-start failure bucket. The
alternative (4/4 compile failures) would point at SFT contract / ABI
mismatch and require a different remediation.

## Variable changed (one logical diagnostic, three file edits)

The single logical change is "instrument the EXP-003 reward chain to
attribute -1 rewards into 4 buckets across all 4 WCC tasks." Operationally:

1. **`scripts/debug_eval_pipeline.py`** — rewritten to (a) honor
   `KERNELFORGE_STAGE1_BACKEND_FILTER` defensively, (b) iterate over all
   filtered rows instead of `rows[0]` only, (c) write per-task structured
   JSON to `outputs/exp004_debug/<idx>_<op>.json`, (d) classify each
   rollout into {compile_failed, compiled_but_wrong, anti_hack,
   eval_crash, ok_correct}, (e) print an aggregate bucket-count line.
   Generation params match Stage 1: do_sample=True, temperature=1.0,
   top_p=0.95, top_k=50, max_new_tokens=1024, G=1 per task.
2. **`scripts/cluster/debug_eval.slurm`** — exports
   `KERNELFORGE_STAGE1_BACKEND_FILTER=wcc` and
   `KERNELFORGE_EXP004_OUTDIR=${REPO_ROOT}/outputs/exp004_debug`, and
   creates that directory before launching python so the per-task JSON
   dumps survive the job.
3. **`research/experiments/EXP-004-wcc-failure-diagnostic.md`** — this
   spec.

## Configuration

| Field | Value | Source |
|---|---|---|
| model | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | `training/model_loader.py` |
| adapter | `outputs/kernelforge-stage2/checkpoint-50` | hardcoded in debug script |
| training GPU | H200 80GB × 1 | `debug_eval.slurm` `--gres=gpu:h200:1` |
| eval backend | `local` | `KERNELFORGE_EVAL_BACKEND=local` |
| dataset filter | `wcc` | `KERNELFORGE_STAGE1_BACKEND_FILTER=wcc` |
| expected #tasks | 4 (the EXP-003 D6 subset) | matches stage1_warmup.py filter |
| G per task | 1 | hardcoded |
| max_new_tokens | 1024 | matches `KERNELFORGE_STAGE1_MAX_COMPLETION_LENGTH` |
| temperature / top_p / top_k | 1.0 / 0.95 / 50 | matches Stage 1 |
| reward_version | `v1-discrete-milestone` (frozen) | `openenv_env/reward.py` |

## Managed runner

NU Explorer cluster, `gpu` partition, H200 × 1, 1h walltime.
Submit with: `sbatch scripts/cluster/debug_eval.slurm`.

## Decision rule

This experiment's whole purpose is to produce a 4-way bucket count. The
table below maps the observed distribution to the next informational
experiment idea (NOT committed; the user decides at review time):

| Observed bucket counts | Verdict | Next experiment idea (informational) |
|---|---|---|
| 4/4 compiles=True, correct=False | correctness-bottleneck | Shaped reward (Kevin recipe; bump `reward_version=v2-shaped`) |
| 4/4 compiles=False AND signature error | contract mismatch | More SFT data with correct WCC contract — NOT prompt hardcode (see do-not-repeat) |
| 4/4 compiles=False AND real nvcc error | SFT producing junk code | Grow SFT corpus to ≥2K (literature floor) |
| ≥2 anti_hack tripwires | reward harness working | Audit SFT data for hackable patterns |
| Mixed | report distribution | Choose dominant bucket as primary failure mode |

## Metrics to record (informational, NOT for master.json)

| Field | Value |
|---|---|
| experiment_id | `EXP-004-wcc-diagnostic` |
| parent_master_hash | `null` |
| runner | `explorer-h200-local` |
| job_id | slurm `kf_debug_eval` jobid |
| bucket_compile_failed | int |
| bucket_compiled_but_wrong | int |
| bucket_anti_hack | int |
| bucket_eval_crash | int |
| bucket_ok_correct | int (should be 0; if not, EXP-003-v3 conclusion was wrong) |
| promote | `false` (diagnostic, by construction) |
| comment | `EXP-004 diagnostic — bucket counts above; H200-local eval, NOT master-eligible` |

`research/live/master.json` is NOT modified by this run.

## Cost ceiling

1h H200 wall. Single rollout × 4 tasks at ≤2 min each + ~5 min model load
fits comfortably under the 1h walltime. If any single rollout exceeds 10
min the job should be aborted and the model-load path investigated
(likely OOM / cold MoE init).

## Abort conditions

- Model load > 15 min → likely OOM with bf16 + adapter; switch to
  KERNELFORGE_QUANT_BITS=4 in a follow-up.
- Per-task generation > 5 min → eos not reached at temperature=1.0;
  unrelated to the bucketing question, abort and investigate separately.
- `nvcc` not found on the compute node → blocks compile-bucket
  classification; fall back to extraction-only diagnostic.
- 0 rows after backend filter → debug script reports and exits; check
  `KERNELFORGE_STAGE1_BACKEND_FILTER` propagation and the dataset
  fallback path in `load_stage1_dataset()`.

## Must NOT do (frozen surfaces)

- No edits to `openenv_env/reward.py` (reward_version frozen).
- No edits to `openenv_env/anti_hack.py` (anti-hack rules frozen).
- No edits to `eval_service/eval_core.py`, `evaluation/verifier.py`,
  `evaluation/compiler.py`, `evaluation/ablation.py`.
- No edits to `training/stage1_warmup.py` or `training/model_loader.py`
  (out of scope; we're diagnosing, not changing GRPO setup).
- No `git add` / `commit` / `push` / `sbatch` / `scancel` issued by this
  prep agent. The user dispatches.

## Open follow-ups

1. If bucket = correctness-bottleneck, draft EXP-005 around shaped reward
   (e.g., `reward(compile)=0`, `reward(correct,slow)=1`, etc.) and decide
   whether to invoke `kernelforge-reward-design` skill for the change to
   `openenv_env/reward.py`.
2. If bucket = compile/contract mismatch, draft EXP-005 around growing
   the doubleGraph SFT corpus or adding contract-anchored prompts (NOT
   hardcoded examples — that's in do-not-repeat).
3. If bucket = mostly anti_hack, the diagnostic itself is the result;
   document patterns in SFT corpus that trigger Dr. Kernel checks.

## Status

**READY TO DISPATCH (PENDING USER REVIEW)** — H200-only, 1h ceiling, no
A100 capacity required.
