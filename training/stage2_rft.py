"""
Stage 2: Rejection Fine-Tuning (RFT) with SFT.

1. Collect trajectories from Stage 1 checkpoint
2. Filter: keep only trajectories with reward >= 1.0 (correct kernels)
3. Train with SFTTrainer on filtered high-quality trajectories

Critical: ByteDance ablation showed skipping RFT causes policy entropy
explosion and training collapse (Table 2, arXiv:2602.24286).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from trl import SFTConfig, SFTTrainer

from training.checkpoint_utils import find_resumable_checkpoint, has_model_files
from training.dataset_loader import Dataset, MiniDataset, load_training_dataset
from training.model_loader import load_model_and_tokenizer
from training.rft_filter import TrajectoryCollector

def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"warning: {name}={raw!r} not int; using default {default}", file=sys.stderr)
        return default


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"warning: {name}={raw!r} not float; using default {default}", file=sys.stderr)
        return default


STAGE1_OUTPUT = os.getenv("KERNELFORGE_STAGE1_OUTPUT", "outputs/kernelforge-stage1")
OUTPUT_DIR = os.getenv("KERNELFORGE_STAGE2_OUTPUT", "outputs/kernelforge-stage2")
NUM_TRAJECTORIES = _env_int("KERNELFORGE_RFT_TRAJECTORIES", 50)
MIN_REWARD = _env_float("KERNELFORGE_RFT_MIN_REWARD", 1.0)
# EXP-002: when set, skip the broken RFT collection path (requires a Stage 1
# checkpoint that does not exist at cold start) and train SFT directly on
# the primary SFT corpus selected by KERNELFORGE_SFT_DATA_SOURCE.
SKIP_RFT_COLLECTION = os.getenv("KERNELFORGE_SKIP_RFT_COLLECTION", "0") == "1"
# Primary SFT corpus selector — added to support EXP-003-verify-v2.
#   "doublegraph" (default): 192 doubleGraph expert kernels (WCC void contract).
#                            Use this for Stage 3 graph-kernel warm-starts.
#   "sakana":                200 Sakana CUDA Engineer rows (Tensor run_kernel
#                            return contract). Use this for Stage 1 ops6k
#                            warm-starts.
SFT_DATA_SOURCE = os.getenv("KERNELFORGE_SFT_DATA_SOURCE", "doublegraph")
# EXP-007 D4: cap epochs for large SFT corpora that would exceed 8h walltime.
STAGE2_EPOCHS = _env_float("KERNELFORGE_STAGE2_EPOCHS", 3.0)
# Save frequency — override when corpus is small so a 1-epoch run still
# produces at least one checkpoint (see memory: NU 8h walltime + HF default
# save_steps=500 silently drops short runs).
STAGE2_SAVE_STEPS = _env_int("KERNELFORGE_STAGE2_SAVE_STEPS", 50)
# Optional hard step cap wired by modal_train.py. 0 (default) = no cap,
# preserving the epochs-driven behavior above.
STAGE2_MAX_STEPS = _env_int("KERNELFORGE_STAGE2_MAX_STEPS", 0)
USE_BF16 = sys.platform.startswith("linux")


def _dataset_from_rows(rows: list[dict[str, Any]]) -> Dataset:
    if hasattr(Dataset, "from_list"):
        return Dataset.from_list(rows)
    return MiniDataset(rows)


def _load_doublegraph_sft_rows() -> list[dict[str, Any]]:
    # SFTTrainer detects `messages` via TRL `is_conversational` and applies the
    # real Qwen chat template via `processing_class.apply_chat_template`. Do NOT
    # pre-fill a `text` column — that path is for non-conversational data only
    # and would bypass the template.
    rows = load_training_dataset(stage="stage2")
    return [row for row in rows if row.get("messages")]


def _load_sakana_sft_rows(path: str = "datasets/sakana_sft.jsonl") -> list[dict[str, Any]]:
    """Load Sakana-derived SFT rows (ops6k Tensor run_kernel contract)."""
    import json
    rows: list[dict[str, Any]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("messages"):
                rows.append(row)
    return rows


def _load_primary_sft_rows() -> list[dict[str, Any]]:
    if SFT_DATA_SOURCE == "sakana":
        return _load_sakana_sft_rows()
    if SFT_DATA_SOURCE == "doublegraph":
        return _load_doublegraph_sft_rows()
    raise ValueError(
        f"Unknown KERNELFORGE_SFT_DATA_SOURCE={SFT_DATA_SOURCE!r}; "
        "expected one of {'doublegraph', 'sakana'}."
    )


def main():
    """Run Stage 2: collect trajectories, filter, SFT."""
    print("=== Stage 2: Rejection Fine-Tuning ===")

    if SKIP_RFT_COLLECTION:
        print(
            "=== KERNELFORGE_SKIP_RFT_COLLECTION=1 — skipping RFT trajectory "
            "collection, training on doubleGraph SFT only ==="
        )
        filtered: list[dict[str, Any]] = []
        rft_rows: list[dict[str, Any]] = []
    else:
        # Step 1: Collect trajectories using Stage 1 model
        print(f"Collecting {NUM_TRAJECTORIES} trajectories from {STAGE1_OUTPUT}...")
        collector = TrajectoryCollector(model_path=STAGE1_OUTPUT)
        collector.collect_trajectories(num_trajectories=NUM_TRAJECTORIES)

        # Step 2: Filter
        filtered = collector.filter_trajectories(min_reward=MIN_REWARD)
        if not filtered:
            # Do NOT bail out: returning here trained nothing at all, which is
            # strictly worse than SKIP_RFT_COLLECTION=1 (primary corpus only).
            print(
                "WARNING: no RFT trajectories met the quality threshold "
                f"(min_reward={MIN_REWARD}) — continuing with the primary "
                f"'{SFT_DATA_SOURCE}' SFT corpus only."
            )
            rft_rows = []
        else:
            # Step 3: Save filtered dataset
            os.makedirs("datasets", exist_ok=True)
            rft_dataset = collector.save_rft_dataset(filtered, "datasets/rft_filtered.jsonl")
            rft_rows = rft_dataset.to_list() if hasattr(rft_dataset, "to_list") else list(rft_dataset)

    primary_sft_rows = _load_primary_sft_rows()
    combined_sft_rows = primary_sft_rows + rft_rows
    train_dataset = _dataset_from_rows(combined_sft_rows)
    print(
        f"Merged Stage 2 SFT corpus: {SFT_DATA_SOURCE}={len(primary_sft_rows)} + "
        f"filtered_trajectories={len(rft_rows)} -> total={len(combined_sft_rows)}"
    )

    # Step 4: Load model from Stage 1 checkpoint and train.
    # Treat the directory as a valid checkpoint only if it actually has model
    # files — TRL creates the output dir at trainer init even when no
    # save_steps ever fires, so a bare empty dir is not a checkpoint. Stage 1
    # saves a PEFT adapter (adapter_config.json, NO config.json), so the gate
    # must accept either; requiring config.json alone silently trained the
    # base model on every handoff.
    has_ckpt = has_model_files(STAGE1_OUTPUT)
    checkpoint_path = STAGE1_OUTPUT if has_ckpt else None
    print(
        f"Loading Stage 1 checkpoint from {STAGE1_OUTPUT}..."
        if has_ckpt
        else f"No Stage 1 checkpoint at {STAGE1_OUTPUT} (or dir empty) — loading base model."
    )
    model, tokenizer = load_model_and_tokenizer(checkpoint_path=checkpoint_path)

    # Adaptive save_steps: ensure a small corpus + few epochs still gets at
    # least 2 checkpoints. (batch=1, grad_accum=4) → effective batch 4 →
    # steps_per_epoch ≈ len(corpus) / 4.
    steps_per_epoch = max(1, len(combined_sft_rows) // 4)
    total_steps = max(1, int(steps_per_epoch * STAGE2_EPOCHS))
    save_steps = min(STAGE2_SAVE_STEPS, max(1, total_steps // 2))

    sft_kwargs = dict(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        num_train_epochs=STAGE2_EPOCHS,
        logging_steps=1,
        save_steps=save_steps,
        bf16=USE_BF16,
        report_to="wandb",
        max_length=8192,
        eos_token=tokenizer.eos_token,
    )
    if STAGE2_MAX_STEPS > 0:
        sft_kwargs["max_steps"] = STAGE2_MAX_STEPS
        total_steps = min(total_steps, STAGE2_MAX_STEPS)
    config = SFTConfig(**sft_kwargs)
    print(
        f"Stage 2 SFT: corpus={len(combined_sft_rows)} rows, "
        f"epochs={STAGE2_EPOCHS}, total_steps={total_steps}, save_steps={save_steps}"
        + (f", max_steps cap={STAGE2_MAX_STEPS}" if STAGE2_MAX_STEPS > 0 else "")
    )

    # Dataset has `messages` field; TRL SFTTrainer auto-applies the chat template
    # when it sees a conversational dataset, so we do not pass dataset_text_field.
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=train_dataset,
    )

    # Auto-resume from the newest VALID checkpoint in OUTPUT_DIR if one exists.
    # HF Trainer does NOT auto-resume just because checkpoints are present —
    # it requires resume_from_checkpoint (job 6920681 lesson, 2026-05-19:
    # we expected resume but got "loading base model" instead, wasting 1.5h).
    # Passing the specific path (not True) skips half-written checkpoints a
    # walltime kill left behind.
    resume_ckpt = find_resumable_checkpoint(OUTPUT_DIR)
    if resume_ckpt:
        print(f"Stage 2 SFT: auto-resuming from checkpoint {resume_ckpt}")
    else:
        print(f"Stage 2 SFT: no valid checkpoint in {OUTPUT_DIR}; starting fresh.")
    print("Starting Stage 2 SFT training on filtered trajectories...")
    trainer.train(resume_from_checkpoint=resume_ckpt if resume_ckpt else None)

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Stage 2 complete. Checkpoint saved to {OUTPUT_DIR}")

    # Print stats
    if filtered:
        rewards = [t["reward"] for t in filtered]
        print(f"RFT stats: {len(filtered)} trajectories, "
              f"rewards min={min(rewards):.1f} max={max(rewards):.1f} "
              f"mean={sum(rewards)/len(rewards):.2f}")
    else:
        print("RFT stats: 0 trajectories (SFT-only run on the primary corpus).")


if __name__ == "__main__":
    main()
