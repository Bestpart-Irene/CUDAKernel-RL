# EXP-001 — stage1 GRPO warm-up baseline

Cold-start. First managed run on Explorer cluster. Establishes the first
entry in `research/live/master.json` and the first row in
`research/results.tsv`.

## Hypothesis

100-step GRPO warm-up on `Qwen3-Coder-30B-A3B-Instruct` with the
v1-discrete-milestone reward produces a non-degenerate policy and a
parseable set of metrics (`mean_reward`, `pass_rate`, `speedup_vs_orig`,
`fast_p`) that can anchor the promotion gate for all subsequent runs.

## Parent master

`null` — establishing first master.

## Variable changed

None. This is a baseline; no source-file change. The experiment is purely
a scheduled launch of `python -m training.grpo_train --stage stage1`.

## Configuration locked by this run

| Field | Value | Source |
|---|---|---|
| model | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | `training/model_loader.py` `PRIMARY_MODEL` |
| quant | bf16 (no quantization) | `kf_stage1.slurm` `KERNELFORGE_QUANT_BITS=0` |
| stage entrypoint | `training.grpo_train --stage stage1` | |
| training GPU | H200 80GB × 1 | `kf_stage1.slurm` `--gres=gpu:h200:1` |
| eval backend | `coreweave` (FastAPI on A100) | `kf_stage1.slurm` exports `KERNELFORGE_EVAL_BACKEND=coreweave` |
| eval GPU | A100 80GB × 1 | `kf_eval_server.slurm` `--gres=gpu:a100:1` |
| steps | 100 | `training/stage1_warmup.py` |
| LR | 2e-6 | `training/CLAUDE.md` |
| temperature | 1.0 | `training/CLAUDE.md` |
| G (generations) | 2 | `training/CLAUDE.md` |
| max_turns | 3 | `KERNELFORGE_MAX_TURNS=3` default |
| reward_version | v1-discrete-milestone | `openenv_env/reward.py` |
| eval split | heldout (`evaluation/eval_model.py::_load_eval_tasks`) | |
| seed count | 5 (master.json default) | |

## Managed runner

Northeastern Explorer cluster.

- Eval server: `sbatch scripts/cluster/kf_eval_server.slurm` (A100 × 1,
  `multigpu` partition, 24h).
- Training:   `sbatch scripts/cluster/kf_stage1.slurm` (H200 × 1,
  `multigpu` partition, 24h).
- Coordinator:`bash scripts/cluster/launch_run.sh stage1` submits both in
  the right order.

## Execution checklist

```
# On laptop
bash scripts/cluster/sync_to_discovery.sh

# On Explorer
ssh xie.yiyi@login.explorer.northeastern.edu
cd ~/CUDAKernel-RL

# First time only
sbatch scripts/cluster/setup_env.slurm
tail -f logs/kf_setup_*.out

# Smoke (optional, 30 min)
sbatch scripts/cluster/smoke.slurm

# The actual baseline run
bash scripts/cluster/launch_run.sh stage1

# Monitor
squeue -u $USER -o '%.10i %.20j %.8T %.10M %R'
tail -f logs/kf_stage1_*.out

# When stage1 finishes, free the A100
scancel <eval_server_jobid>
```

## Metrics to record

From the run log, parse and append one row to `research/results.tsv`:

| Column | Source |
|---|---|
| timestamp | wall-clock of run completion |
| experiment_id | `EXP-001` |
| parent_master_hash | `null` |
| runner | `explorer-h200-multigpu` |
| job_id | slurm `kf_stage1` jobid |
| mean_reward | final eval mean over heldout |
| pass_rate | final eval pass rate |
| speedup_vs_orig | mean speedup over reference |
| fast_p | fast_p metric |
| reward_version | `v1-discrete-milestone` |
| promote | `true` (cold-start unconditional) |
| comment | `cold-start baseline` |

Then write `research/live/master.json` with the same metrics, plus the
commit SHA at the time of dispatch.

## Promotion decision

Cold-start unconditional promote: if the run completes with parseable
metrics, it becomes master. See the "Cold-start promotion convention"
preamble in `research/notes.md`.

## Open follow-ups (do not block this run)

1. **Eval-split overlap assertion** — `reviewer` flagged that we lack
   hard proof that `training/dataset_loader.py` `stage1` task IDs do not
   overlap with `evaluation/eval_model.py::_load_eval_tasks`. Next
   experiment (`EXP-002`) should add a preflight assertion that prints
   the set diff and aborts on overlap. **Do not edit
   `evaluation/eval_model.py`** (frozen) — add the check in
   `training/grpo_train.py::preflight`.
2. **Seed-variance anchor** — after this run, schedule a 3-seed re-eval
   on the produced checkpoint to record CI bounds in `master.json`
   alongside the point estimate.
3. **`eval_backend` field in `master.json`** — added in this run to lock
   future runs to the same A100 HTTP eval path. If anyone ever wants to
   re-baseline on `local` in-process, that becomes a new master, not a
   comparable continuation.

## Status

Status: **READY TO DISPATCH** — pending user confirmation.
