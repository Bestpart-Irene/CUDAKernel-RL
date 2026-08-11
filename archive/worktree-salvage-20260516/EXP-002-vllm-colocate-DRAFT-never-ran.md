# EXP-002 — switch rollout backend to vLLM colocate

Second managed run. Replaces the slow vanilla-Transformers + PEFT
`.generate()` rollout path (the only path available after Unsloth's fast
path broke upstream on Qwen3 MoE — see researcher notes below) with vLLM
running in `colocate` mode on the same H200 as the trainer.

## Hypothesis

Enabling vLLM colocate (`KERNELFORGE_USE_VLLM=1`,
`vllm_gpu_memory_utilization=0.18`) cuts wall-clock per training step
by at least 2x vs the EXP-001 vanilla `.generate()` path, without
regressing `mean_reward` or `pass_rate` on the heldout split.

Secondary: confirm that on H200 80GB the 0.18 utilization budget is
sufficient for `max_completion_length=1024` with `G=2` rollouts and does
not OOM at vLLM engine init.

## Parent master

If EXP-001 (slurm job 6858988) completes and cold-start-promotes before
this is dispatched: parent = that master SHA (fill in at dispatch).

If EXP-001 fails or has not produced a parseable metric row by
dispatch time: parent = `null`. This is a **cold-start continuation**
under the same "first parseable run becomes master" rule, because the
rollout backend change is so fundamental that there is no meaningful
A/B against an unfinished EXP-001 anyway.

## Variable changed

One logical variable: **rollout backend = vLLM colocate**. Operationally
this is two coupled env-var exports in `scripts/cluster/kf_stage1.slurm`
— `KERNELFORGE_USE_VLLM` (the switch) and
`KERNELFORGE_VLLM_GPU_MEMORY_UTILIZATION` (the backend parameter that
must be set together with the switch on H200 80GB; the default 0.6 OOMs
because it does not leave room for the 61GB bf16 30B-A3B actor plus
LoRA grads and activations). Treating the util value as an independent
variable would be misleading — at H200's memory budget, 0.6 is not a
valid setting for vLLM colocate at all, so flipping the switch alone
would not produce a runnable experiment.

### Exact before/after in `scripts/cluster/kf_stage1.slurm`

Before (EXP-001):

```
# H200 141GB fits Qwen3-Coder-30B-A3B at bf16 with vLLM/GRPO headroom.
export KERNELFORGE_QUANT_BITS="${KERNELFORGE_QUANT_BITS:-0}"

# Override cluster_paths.sh's default (local) — we are using remote A100 eval.
export KERNELFORGE_EVAL_BACKEND=coreweave
```

After (EXP-002):

```
# H200 141GB fits Qwen3-Coder-30B-A3B at bf16 with vLLM/GRPO headroom.
export KERNELFORGE_QUANT_BITS="${KERNELFORGE_QUANT_BITS:-0}"

# EXP-002: switch rollout backend from HF .generate() to vLLM colocate.
# colocate mode is mandatory — server mode breaks multi-turn GRPO importance
# sampling (TRL #4543). util=0.18 is the budgeted vLLM share after bf16
# 30B-A3B actor (~61GB) + LoRA grads (~2GB) + activations on H200 80GB.
export KERNELFORGE_USE_VLLM=1
export KERNELFORGE_VLLM_GPU_MEMORY_UTILIZATION=0.18

# Override cluster_paths.sh's default (local) — we are using remote A100 eval.
export KERNELFORGE_EVAL_BACKEND=coreweave
```

No `.py` file is touched. `training/stage1_warmup.py:39-40` already
reads both env vars.

## Configuration held constant vs EXP-001

| Field | Value | Source |
|---|---|---|
| model | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | `training/model_loader.py` `PRIMARY_MODEL` |
| quant | bf16 (no quantization) | `kf_stage1.slurm` `KERNELFORGE_QUANT_BITS=0` |
| stage entrypoint | `training.grpo_train --stage stage1` | |
| training GPU | H200 80GB × 1 | `kf_stage1.slurm` `--gres=gpu:h200:1` |
| eval backend | `coreweave` (FastAPI on A100) | `kf_stage1.slurm` `KERNELFORGE_EVAL_BACKEND=coreweave` |
| eval GPU | A100 80GB × 1 | `kf_eval_server.slurm` `--gres=gpu:a100:1` |
| steps | 100 | `training/stage1_warmup.py` |
| LR | 2e-6 | `training/CLAUDE.md` |
| temperature | 1.0 | `training/CLAUDE.md` |
| G (generations) | 2 | `training/CLAUDE.md` |
| max_turns | 3 | `KERNELFORGE_MAX_TURNS=3` default |
| max_completion_length | 1024 | `KERNELFORGE_STAGE1_MAX_COMPLETION_LENGTH=1024` default |
| reward_version | v1-discrete-milestone | `openenv_env/reward.py` (frozen) |
| eval split | heldout (`evaluation/eval_model.py::_load_eval_tasks`) | (frozen) |
| seed count | 5 (master.json default) | |

Comparability key (must match parent master if non-null): `model`,
`quant`, `reward_version`, `eval split`, `eval_backend`, `steps`, `G`,
`max_turns`, `max_completion_length`, `temperature`, `seed count`.

## Researcher citations

- **TRL #4543** — In multi-turn GRPO, `vllm_mode="server"` drops per-turn
  prefixes and breaks importance sampling. **`colocate` is mandatory.**
  We MUST NOT set `vllm_mode="server"` regardless of any throughput
  advertisement.
- **Unsloth GH #3807 and #3422** — Unsloth's fast-path patches for
  Qwen3 MoE (`Qwen3MoeForCausalLM`) are currently broken upstream. This
  is *why* EXP-001 is running the slow vanilla Transformers + PEFT
  path, and why a vLLM rollout switch is the highest-leverage one-file
  change available right now. Do not waste planner slots retrying
  Unsloth on Qwen3 MoE until Datta0 ships a fix.
- **HF blog: "vLLM colocate for GRPO" (huggingface.co/blog/vllm-colocate)**
  — On H200 80GB, a bf16 30B-A3B actor (~61GB) plus LoRA grads (~2GB)
  plus activations leaves roughly 12–15GB for vLLM. That maps to
  `vllm_gpu_memory_utilization ≈ 0.18`, not the library default 0.6.
- **HF blog: "The async-rl landscape" (huggingface.co/blog/...async-rl...)**
  — AsyncGRPOTrainer's "Keep Routing" stabilizer is not implemented
  for MoE actors. We are NOT enabling async pipelining in this run;
  rollout and policy step remain synchronous. Asynchronous H200↔A100
  is parked as a future experiment, see `do-not-repeat.md` entry C.

## Managed runner

Northeastern Explorer cluster (same as EXP-001).

- Eval server: `sbatch scripts/cluster/kf_eval_server.slurm` (already
  running from EXP-001 → can be reused; if cancelled, relaunch).
- Training: `sbatch scripts/cluster/kf_stage1.slurm` (H200 × 1).
- Coordinator: `bash scripts/cluster/launch_run.sh stage1`.

## Execution checklist

```
# On laptop, after user merges this worktree back to main
bash scripts/cluster/sync_to_discovery.sh

# On Explorer
ssh xie.yiyi@login.explorer.northeastern.edu
cd ~/CUDAKernel-RL

# Decide EXP-001's fate first:
#   - if 6858988 is still progressing usefully, let it finish and record EXP-001 row
#   - if it's stuck on step 1 with no sign of progress, scancel 6858988
#     (and note that in research/notes.md before relaunch)

# Launch EXP-002
bash scripts/cluster/launch_run.sh stage1

# Monitor
squeue -u $USER -o '%.10i %.20j %.8T %.10M %R'
tail -f logs/kf_stage1_*.out
```

## Metrics to record

From the run log, parse and append one row to `research/results.tsv`:

| Column | Source |
|---|---|
| timestamp | wall-clock of run completion |
| experiment_id | `EXP-002` |
| parent_master_hash | EXP-001 SHA (or `null` if EXP-001 didn't promote) |
| runner | `explorer-h200-multigpu` |
| job_id | slurm `kf_stage1` jobid for EXP-002 |
| mean_reward | final eval mean over heldout |
| pass_rate | final eval pass rate |
| speedup_vs_orig | mean speedup over reference |
| fast_p | fast_p metric |
| reward_version | `v1-discrete-milestone` |
| promote | per promotion rule below |
| comment | record `sec_per_step` here (median of trainer step time once vLLM is warm — this is the primary signal for the wall-clock-2x hypothesis even when reward gates are tied) |

Then update `research/live/master.json` if promoted, plus the commit
SHA at the time of dispatch.

## Promotion rule

- If `parent_master_hash != null`: **strict gate** — promote iff
  `mean_reward(EXP-002) > mean_reward(parent)` AND
  `pass_rate(EXP-002) >= pass_rate(parent)` AND comparability key
  matches the parent's master.json entry. `sec_per_step` is the
  speedup motivation, but the reward gate still governs promotion —
  we don't trade reward for wall-clock.
- If `parent_master_hash == null`: **cold-start unconditional** —
  any parseable metric row promotes.

## Abort conditions (run-time)

- **OOM at vLLM engine init** → abort. Do NOT in-place lower
  `vllm_gpu_memory_utilization` and restart; that becomes a new
  hypothesis (EXP-003: "is util=0.12 enough on H200 80GB?"). Cancel,
  record the failure in `do-not-repeat.md` with the actual GPU
  memory print-out, and let `planner` decide the next util target.
- **reward_std = 0 after 30 steps** → abort, model collapsed (same
  rule as EXP-001 / `training/CLAUDE.md`).
- **vLLM "server" mode appears in any log line** → abort
  immediately. We never enable server mode (TRL #4543).
- **`KERNELFORGE_USE_VLLM` not observed at `1` in preflight log
  output** → abort. The env-var did not propagate; relaunch.

## Open follow-ups (do not block this run)

1. Once vLLM lands in master, retry the `B` ladder (`beta`,
   `max_prompt_length`, `max_completion_length`) as three separate
   single-variable experiments. See `do-not-repeat.md` entry B.
2. Once rollout is no longer H200-`.generate()`-bound, revisit async
   H200↔A100 pipelining. See `do-not-repeat.md` entry C.
3. Add a preflight assertion that prints `(USE_VLLM,
   VLLM_GPU_MEMORY_UTILIZATION)` so post-mortem of any future OOM is
   one grep away.

## Status

Status: **READY TO DISPATCH (PENDING USER REVIEW)** — worktree only.
No commit, no push, no sbatch.
