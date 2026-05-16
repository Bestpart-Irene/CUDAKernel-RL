# EXP-002 — SFT-first reorder: Stage 2 RFT on doubleGraph expert demos

Cold-start recovery from EXP-001. EXP-001 (slurm 6858988, wandb
`xxiellan-northeastern-university/kernelforge/qll2wvnl`) cancelled at step
13 of 100 after `train/reward` collapsed to a constant -1, `reward_std=0`,
`frac_reward_zero_std=1.0`, `grad_norm=0`, `clipped_ratio=1.0`,
`mean_terminated_length=0`. Diagnosis: base `Qwen3-Coder-30B-A3B-Instruct`
emits ~zero parseable ```cuda blocks within 1024 tokens, so GRPO sees no
reward signal and weights never move. Standard fix is an SFT pass on the
192 doubleGraph A100 expert kernels before any GRPO.

This experiment reorders the pipeline: run Stage 2 (SFT on doubleGraph
expert demos) BEFORE Stage 1 (GRPO warm-up). The Stage 2 checkpoint
becomes the warm-start for EXP-003 (Stage 1 GRPO resumed from this SFT
ckpt), which will be the actual cold-start of `research/live/master.json`.

## Hypothesis

SFT on the 192 doubleGraph A100 expert kernels for 3 epochs produces a
converged checkpoint at `outputs/kernelforge-stage2/` with `train/loss`
strictly decreasing from step 1 to last step, and a final value below
initial. This checkpoint will anchor EXP-003 (Stage 1 GRPO from this SFT
checkpoint), which is the actual test of the SFT-first hypothesis.

## Parent master

`null`. EXP-001 was cancelled at step 13 before producing parseable
metrics — no GRPO column populated in `research/results.tsv`. EXP-002
itself produces SFT loss, NOT GRPO metrics: `mean_reward`, `pass_rate`,
`speedup_vs_orig`, `fast_p` are all undefined for an SFT run. Therefore
**EXP-002 cannot promote `master.json`**. EXP-003 (GRPO from this SFT
ckpt) will be the actual cold-start of master.

## Variable changed

**One logical variable: skip the broken RFT trajectory collection path,
train SFT directly on doubleGraph expert demos.**

Operationally this requires two coupled writes implementing the single
logical change (same justification used by the EXP-002-vllm-colocate spec
for the coupled `USE_VLLM` + `gpu_memory_utilization` pair):

1. **`training/stage2_rft.py`** — add module-level env-var
   `SKIP_RFT_COLLECTION = os.getenv("KERNELFORGE_SKIP_RFT_COLLECTION", "0") == "1"`
   and short-circuit the `TrajectoryCollector` → `collect_trajectories` →
   `filter_trajectories` → early-return path when set. When skipped,
   `rft_rows = []` and `combined_sft_rows = dg_sft_rows`. Default behavior
   (env-var=0 or unset) is byte-identical to today.
2. **`scripts/cluster/kf_stage2.slurm`** — `export
   KERNELFORGE_SKIP_RFT_COLLECTION="${KERNELFORGE_SKIP_RFT_COLLECTION:-1}"`
   so the experiment is dispatch-ready as a single unit.

The existing `checkpoint_path = STAGE1_OUTPUT if os.path.exists(STAGE1_OUTPUT) else None`
fallback at stage2_rft.py:91 already handles the "no Stage 1 ckpt" case
correctly — `load_model_and_tokenizer(checkpoint_path=None)` loads the
base model. No change needed there.

### Before / after diff summary

- `training/stage2_rft.py`
  - Before: `main()` unconditionally calls `TrajectoryCollector(...)` and
    early-returns when `filtered` is empty.
  - After: when `SKIP_RFT_COLLECTION` is set, the collector + filter + save
    block is skipped, `rft_rows = []`, and execution proceeds to load
    `dg_sft_rows` and train.
- `scripts/cluster/kf_stage2.slurm`
  - Before: no `KERNELFORGE_SKIP_RFT_COLLECTION` export.
  - After: `export KERNELFORGE_SKIP_RFT_COLLECTION="${...:-1}"` near other
    env vars, with comment referencing EXP-002.

## Configuration held constant

| Field | Value | Source |
|---|---|---|
| model | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | `training/model_loader.py` `PRIMARY_MODEL` |
| quant | bf16 (no quantization) | `kf_stage2.slurm` `KERNELFORGE_QUANT_BITS=0` |
| stage entrypoint | `training.grpo_train --stage stage2` | `kf_stage2.slurm` |
| training GPU | H200 80GB × 1 | `kf_stage2.slurm` `--gres=gpu:h200:1` |
| eval backend | `coreweave` (unused by SFT, kept for parity) | `kf_stage2.slurm` |
| LR | 5e-6 | Stage 2 default per `training/CLAUDE.md` |
| epochs | 3 | `SFTConfig(num_train_epochs=3)` |
| batch | 1 × grad_accum 4 | `SFTConfig` |
| max_seq_length | 8192 | `SFTConfig` |
| save_steps | 50 | `SFTConfig` (already set — survives 8h SIGTERM partial) |
| dataset | `datasets/doublegraph_sft.jsonl` (192 rows, HF messages format) | loaded by `_load_doublegraph_sft_rows()` |
| model loader path | plain Transformers + PEFT (Unsloth fast-path does NOT support MoE LoRA SFT — Unsloth GH #3422) | `training/model_loader.py` |

## Managed runner

Northeastern Explorer cluster.

- Training: `sbatch scripts/cluster/kf_stage2.slurm` (H200 × 1, `gpu`
  partition, 8h walltime).
- Eval server: `sbatch scripts/cluster/kf_eval_server.slurm` (kept up for
  parity; SFT does not call eval, but the slurm script still waits on
  `eval_server.address` before launching training, so the server must be
  alive).

## Metrics to record

EXP-002 is an SFT run. RL columns are `null`. From the run log, record:

| Column | Value |
|---|---|
| timestamp | wall-clock of run completion |
| experiment_id | `EXP-002` |
| parent_master_hash | `null` |
| runner | `explorer-h200-gpu` |
| job_id | slurm `kf_stage2` jobid |
| mean_reward | `null` |
| pass_rate | `null` |
| speedup_vs_orig | `null` |
| fast_p | `null` |
| reward_version | `null` (no rollout) |
| promote | `false` (SFT cannot promote master) |
| comment | `SFT-first reorder; initial_sft_loss=<x>; final_sft_loss=<y>; ckpt_path=outputs/kernelforge-stage2/` |

`research/live/master.json` is **NOT modified** by this run.

## Researcher citations

- **`training/CLAUDE.md` abort conditions**: "reward std = 0 → model
  collapsed" (lines under "Abort Conditions"). EXP-001 hit this exactly.
- **Unsloth GH #3422**: vLLM does not support MoE LoRA. Confirms that
  Stage 2 SFT must go through plain Transformers + PEFT, not the Unsloth
  fast path. The current `model_loader.py` route is correct.
- **EXP-001 wandb run `qll2wvnl`**: empirical evidence that base
  `Qwen3-Coder-30B-A3B-Instruct` cannot bootstrap GRPO without prior SFT
  on CUDA-kernel-format text.

## Abort conditions

- **SFT loss not strictly decreasing by step 50** → dataset/format
  mismatch with SFTTrainer. Abort and re-check `_messages_to_text`
  formatting and the `dataset_text_field="text"` wiring in
  `stage2_rft.py`.
- **OOM at SFT init** → abort. Do NOT in-place lower batch — that becomes
  EXP-002b with its own variable change.
- **Walltime SIGTERM before 1 full epoch (~192 / 4 = 48 grad-accum steps
  per epoch)** → abort. Checkpoint at
  `outputs/kernelforge-stage2/checkpoint-<n>/` via `save_steps=50` should
  survive at least one partial save.
- **`KERNELFORGE_SKIP_RFT_COLLECTION` flag NOT observed at value 1 in run
  stdout** (look for the explicit log line
  `=== KERNELFORGE_SKIP_RFT_COLLECTION=1 — skipping RFT trajectory collection, training on doubleGraph SFT only ===`)
  → abort. Env-var didn't propagate; relaunch.

## Open follow-ups

1. **EXP-003 — Stage 1 GRPO from this SFT ckpt** = the actual cold-start
   of `master.json`. Will set `KERNELFORGE_STAGE1_OUTPUT_RESUME` (or
   equivalent) to `outputs/kernelforge-stage2/` and re-run the same
   Stage 1 config as EXP-001. This is the experiment that tests whether
   SFT-first lifts `p(valid_cuda_block)` above the GRPO signal floor.
2. **EXP-004+** — vLLM colocate (currently parked under the other
   worktree spec which will be renamed from EXP-002 → EXP-004), TRLOO
   advantage scaling validation, and seed-variance anchors. None of
   these can run until EXP-003 produces the first GRPO master.

## Status

Status: **READY TO DISPATCH (PENDING USER REVIEW)**.
