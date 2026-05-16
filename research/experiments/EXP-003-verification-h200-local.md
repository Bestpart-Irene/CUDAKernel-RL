# EXP-003 — VERIFICATION PHASE: Stage 1 GRPO warm-start from EXP-002 SFT ckpt (H200 local eval)

> **⚠️ VERIFICATION PHASE — NOT MASTER-PROMOTION-ELIGIBLE.**
> This run uses `KERNELFORGE_EVAL_BACKEND=local` on the H200 training node,
> NOT the A100 FastAPI eval service. Reward / speedup / pass_rate / fast_p
> numbers measured here are H200-relative and **not comparable** to A100
> baselines. They answer two diagnostic questions ONLY:
>   1. Does the EXP-002 SFT prior give GRPO non-zero reward variance? (`reward_std > 0`)
>   2. Does GRPO produce non-zero policy gradients on this stack? (`grad_norm > 0`)
>
> If both pass, a follow-up run (EXP-003-prod) re-runs the same
> configuration with `KERNELFORGE_EVAL_BACKEND=coreweave` on A100 — that
> is the actual cold-start candidate for `research/live/master.json`.
> Do NOT update master.json from this run.

## Background

EXP-001 (slurm 6858988) demonstrated cold-start exploration failure: base
Qwen3-Coder-30B-A3B-Instruct emits ~0 parseable ```cuda blocks per
rollout, so GRPO group `reward_std = 0`, `grad_norm = 0`, and 13 training
steps moved no weights.

EXP-002 (slurm 6868569) ran SFT on 192 doubleGraph A100 expert kernels
through step 50 of 144 before crashing on a post-save exit-120 anomaly.
The checkpoint at `outputs/kernelforge-stage2/checkpoint-50/` is complete
(3.38 GB LoRA adapter + optimizer + scheduler) with healthy SFT learning
curve (loss 2.5→0.5, grad_norm 0.17–0.31, token_acc 87–89%).

This run tests whether the EXP-002 SFT prior unlocks a learnable GRPO
signal — the central RL hypothesis of the KernelForge pipeline.

## Hypothesis

Loading the Stage 2 SFT LoRA adapter from `outputs/kernelforge-stage2/checkpoint-50/`
into the Stage 1 GRPO trainer produces a policy whose first 30 multi-turn
rollouts yield at least one non-`-1` reward, observable as `train/reward_std > 0`
and `train/grad_norm > 0` within 30 GRPO steps.

## Why H200 local eval (and why that's OK for verification)

- NU Explorer A100 capacity is unreliable; eval_server jobs sit
  PENDING(Priority) for hours.
- The verification questions are **binary** (variance / no variance,
  gradient / no gradient) — they do not depend on the absolute speedup
  scale.
- Reward = 1 ("compiles + correct") is GPU-agnostic — it answers the
  format / extraction question.
- The numerical reward values produced here will systematically differ
  from A100 numbers because:
  - H200 (sm_90) ≠ A100 (sm_80); kernel performance scales differently
  - doubleGraph expert kernels were tuned for sm_80
  - PyTorch eager/compile is also faster on H200, shifting speedup ratios
- Hence `mean_reward`, `pass_rate`, `speedup_vs_orig`, `fast_p`
  produced by this run **must not** be written into master.json.

## Parent master

`null`. This is a verification run; it does not create master, and the
subsequent A100-eval EXP-003-prod will be the actual cold-start candidate.

## Variable changed

**One logical variable: warm-start Stage 1 from the EXP-002 SFT adapter
and switch eval backend to local for verification.** Operationally three
coupled writes implementing the single logical change:

1. **`training/stage1_warmup.py`** — read `KERNELFORGE_STAGE1_INIT_CKPT`
   env var; if set, pass it to `load_model_and_tokenizer(checkpoint_path=...)`.
2. **`training/model_loader.py:_load_from_checkpoint`** — detect
   PEFT-only checkpoint directories (have `adapter_config.json` but no
   `config.json`), load `PRIMARY_MODEL` as the base and apply the adapter
   with `is_trainable=True`.
3. **`scripts/cluster/kf_stage1.slurm`** — default
   `KERNELFORGE_STAGE1_INIT_CKPT=outputs/kernelforge-stage2/checkpoint-50`
   AND default `KERNELFORGE_EVAL_BACKEND=local` with a VERIFICATION
   PHASE comment block, AND skip the eval server wait when backend=local.

## Configuration held constant vs EXP-001

| Field | Value | Source |
|---|---|---|
| model | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | `training/model_loader.py` `PRIMARY_MODEL` |
| quant | bf16 | `kf_stage1.slurm` `KERNELFORGE_QUANT_BITS=0` |
| stage entrypoint | `training.grpo_train --stage stage1` | |
| training GPU | H200 80GB × 1 | `kf_stage1.slurm` `--gres=gpu:h200:1` |
| steps | 100 | `training/stage1_warmup.py` |
| LR | 2e-6 | `training/CLAUDE.md` |
| temperature | 1.0 | `training/CLAUDE.md` |
| G (generations) | 2 | `training/CLAUDE.md` |
| max_turns | 3 | `KERNELFORGE_MAX_TURNS=3` default |
| max_completion_length | 1024 | `KERNELFORGE_STAGE1_MAX_COMPLETION_LENGTH` default |
| reward_version | v1-discrete-milestone | `openenv_env/reward.py` (frozen) |
| eval split | heldout (`evaluation/eval_model.py::_load_eval_tasks`) | (frozen) |

**Intentionally changed vs EXP-001:**
- `KERNELFORGE_STAGE1_INIT_CKPT` from `(unset, base model)` →
  `outputs/kernelforge-stage2/checkpoint-50` (warm-start)
- `KERNELFORGE_EVAL_BACKEND` from `coreweave` → `local` (verification only)

## Managed runner

Explorer cluster, `gpu` partition, H200 × 1, 8h walltime.
No eval server job — eval runs in-process on the training H200.
Submit with: `sbatch scripts/cluster/kf_stage1.slurm`.

## Metrics to record

Append to `research/results.tsv` with a comment flagging verification:

| Column | Value |
|---|---|
| experiment_id | `EXP-003-verify` |
| parent_master_hash | `null` |
| runner | `explorer-h200-local` |
| job_id | slurm `kf_stage1` jobid |
| mean_reward | (H200-local, **NOT comparable to A100**) |
| pass_rate | (H200-local, **NOT comparable to A100**) |
| speedup_vs_orig | (H200-local, **NOT comparable to A100**) |
| fast_p | (H200-local, **NOT comparable to A100**) |
| reward_version | `v1-discrete-milestone` |
| promote | `false` (verification-only, by construction) |
| comment | `VERIFICATION ONLY — eval_backend=local-h200, NOT A100-comparable. reward_std@step30=<x>, grad_norm@step30=<x>` |

`research/live/master.json` is NOT modified by this run.

## Decision rule (verification verdict)

- **PASS verification**: within the first 30 GRPO steps, `train/reward_std > 0`
  for ≥ 3 separate steps AND `train/grad_norm > 0` consistently. Triggers
  EXP-003-prod (same config + `KERNELFORGE_EVAL_BACKEND=coreweave` once
  A100 is available) as the actual cold-start candidate.
- **FAIL verification**: `reward_std = 0` for 30+ steps OR `grad_norm = 0`
  consistently. SFT prior is insufficient. Add a do-not-repeat entry and
  consider deeper interventions (larger SFT corpus, prompt engineering,
  shaped reward, etc.) before another GRPO attempt.

## Abort conditions

- OOM on H200 with both training model + local eval execute paths
  sharing the GPU → abort, record memory profile, follow-up experiment
  needs CPU-side eval scheduling or batched eval.
- `reward_std = 0` for 30+ consecutive steps → FAIL verification per
  decision rule (not a crash, an experiment outcome).
- `KERNELFORGE_STAGE1_INIT_CKPT` not observed in stdout log → abort,
  env var did not propagate; relaunch.
- Walltime SIGTERM → graceful; partial progress survives if `save_steps`
  is configured.

## Open follow-ups

1. **EXP-003-prod** — once verification passes AND A100 is available,
   rerun this exact config with `KERNELFORGE_EVAL_BACKEND=coreweave` to
   produce master.json cold-start with A100-comparable metrics.
2. EXP-002 post-save exit-120 still uninvestigated; ckpt-50 was saved
   before the anomaly so it does not block EXP-003. Track separately.
3. vLLM colocate (from the demoted spec at
   `.claude/worktrees/agent-a1a80c219a07b0e87/`) is now EXP-004 or
   later — only valuable after EXP-003-prod establishes master.

## Status

**READY TO DISPATCH (PENDING USER REVIEW)** — H200-only, no A100 needed.
