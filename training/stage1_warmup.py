"""
Stage 1: GRPO Warm-up — bootstrap CUDA syntax on easy operators.

Multi-turn agentic training via TRL's rollout_func:
  - 3 turns per episode (model sees errors, iterates)
  - Temperature 1.0 for exploration
  - LR 2e-6 to avoid catastrophic forgetting
  - beta=0.04 default (TRL/GRPO standard KL penalty; override to 0.0 via
    KERNELFORGE_STAGE1_BETA=0.0 for the "let model explore freely" ablation)
  - G=2 generations, 100 max_steps (hackathon config)
  - vLLM disabled by default for hackathon bring-up (`KERNELFORGE_USE_VLLM=0`)

Dataset: CUDA-Agent-Ops-6K easy operators (single-op subset).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")

if __package__ in {None, ""}:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from trl import GRPOConfig

from training.checkpoint_utils import find_resumable_checkpoint
from training.custom_grpo_trainer import TRLOOGRPOTrainer
from training.dataset_loader import Dataset, MiniDataset, load_training_dataset
from training.model_loader import load_model_and_tokenizer
from training.multi_turn_rollout import make_multi_turn_rollout, reward_from_env
from training.task_support import normalize_task_row

TARGET_GPU = os.getenv("KERNELFORGE_TARGET_GPU", "A100")
TARGET_ARCH = os.getenv("KERNELFORGE_TARGET_ARCH", "sm_80")
OUTPUT_DIR = os.getenv("KERNELFORGE_STAGE1_OUTPUT", "outputs/kernelforge-stage1")
IS_LINUX = sys.platform.startswith("linux")
USE_VLLM = os.getenv("KERNELFORGE_USE_VLLM", "0") == "1" and IS_LINUX
VLLM_GPU_MEMORY_UTILIZATION = float(os.getenv("KERNELFORGE_VLLM_GPU_MEMORY_UTILIZATION", "0.6"))
OPTIMIZER = "paged_adamw_8bit" if IS_LINUX else "adamw_torch"
USE_BF16 = IS_LINUX

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


# Multi-turn configuration
MAX_TURNS = _env_int("KERNELFORGE_STAGE1_MAX_TURNS", 3)
MAX_STEPS = _env_int("KERNELFORGE_STAGE1_MAX_STEPS", 100)
MAX_COMPLETION_LENGTH = _env_int("KERNELFORGE_STAGE1_MAX_COMPLETION_LENGTH", 1024)
# EXP-003: optional warm-start from a Stage 2 SFT adapter checkpoint.
INIT_CKPT = os.getenv("KERNELFORGE_STAGE1_INIT_CKPT", "") or None
# EXP-006: env-driven G and beta so we can A/B these without code edits.
# Defaults match the original config (G=2, TRL default beta=0.04).
NUM_GENERATIONS = _env_int("KERNELFORGE_STAGE1_NUM_GENERATIONS", 2)
BETA = _env_float("KERNELFORGE_STAGE1_BETA", 0.04)
# EXP-008 C: TRL 0.29 loss_type — "dapo" (default), "grpo", "gspo".
# GSPO addresses Qwen3 MoE token-level GRPO instability (Qwen team's own paper).
LOSS_TYPE = os.getenv("KERNELFORGE_GRPO_LOSS_TYPE", "dapo")


# --- Dataset loading ---


def _dataset_from_rows(rows: list[dict]) -> Dataset:
    if hasattr(Dataset, "from_list"):
        return Dataset.from_list(rows)
    return MiniDataset(rows)


def load_stage1_dataset() -> Dataset:
    """Load stage1 prompts from unified dataset loader, with safe fallback.

    EXP-003 D6: If KERNELFORGE_STAGE1_BACKEND_FILTER is set, drop any task
    whose evaluation_backend does not match. Use "wcc" to align Stage 1
    GRPO with the doubleGraph SFT contract (verification-phase only).
    """

    backend_filter = (os.getenv("KERNELFORGE_STAGE1_BACKEND_FILTER") or "").strip()

    try:
        max_samples = int(os.getenv("CUDA_AGENT_STAGE1_SAMPLES", "512"))
        ds = load_training_dataset(
            stage="stage1",
            ops6k_max=max_samples,
            seed=42,
        )
        if len(ds) > 0:
            if backend_filter:
                rows = ds.to_list() if hasattr(ds, "to_list") else list(ds)
                filtered = [r for r in rows if str(r.get("evaluation_backend", "")) == backend_filter]
                print(
                    f"Loaded {len(ds)} unified Stage 1 prompts; "
                    f"after KERNELFORGE_STAGE1_BACKEND_FILTER={backend_filter!r}: {len(filtered)}"
                )
                if len(filtered) == 0:
                    print(
                        f"WARNING: no Stage 1 prompts matched backend={backend_filter!r}; "
                        "falling through to fallback WCC dataset."
                    )
                else:
                    return _dataset_from_rows(filtered)
            else:
                print(f"Loaded {len(ds)} unified Stage 1 prompts")
                return ds.shuffle(seed=42) if hasattr(ds, "shuffle") else ds
    except Exception as e:
        print(f"Could not load Ops-6K for Stage 1: {e}")

    print("Using fallback Stage 1 prompts with live WCC evaluation support")
    return _dataset_from_rows([
        {
            "prompt": (
                f"Write a CUDA Weakly Connected Components kernel for {TARGET_GPU} ({TARGET_ARCH}) "
                "using union-find with path compression."
            ),
            "ops": ["weakly_connected_components"],
            "difficulty": 1,
            "data_source": "fallback_wcc",
        },
        {
            "prompt": (
                f"Write a CUDA Weakly Connected Components kernel for {TARGET_GPU} ({TARGET_ARCH}) "
                "optimized for sparse disconnected graphs with early convergence."
            ),
            "ops": ["weakly_connected_components"],
            "difficulty": 1,
            "data_source": "fallback_wcc",
        },
        {
            "prompt": (
                f"Write a CUDA Weakly Connected Components kernel for {TARGET_GPU} ({TARGET_ARCH}) "
                "using shared memory staging for dense frontiers."
            ),
            "ops": ["weakly_connected_components"],
            "difficulty": 2,
            "data_source": "fallback_wcc",
        },
    ])


# --- Training ---

def main():
    """Run Stage 1 GRPO warm-up with multi-turn agentic loop."""
    print(f"=== Stage 1: Multi-Turn GRPO Warm-up for {TARGET_GPU} ({TARGET_ARCH}) ===")
    print(f"  Max turns per episode: {MAX_TURNS}")
    print(f"  Max training steps: {MAX_STEPS}")
    print(f"  Max completion length: {MAX_COMPLETION_LENGTH}")

    if INIT_CKPT:
        print(f"Warm-starting Stage 1 from checkpoint: {INIT_CKPT}")
    model, tokenizer = load_model_and_tokenizer(checkpoint_path=INIT_CKPT)
    dataset = load_stage1_dataset()
    task_rows = [normalize_task_row(row) for row in dataset.to_list()]

    rollout_func = make_multi_turn_rollout(
        max_turns=MAX_TURNS,
        skill_md_gpu=TARGET_GPU.lower(),
        problem_metadata=task_rows,
    )

    # Save every N steps so walltime kills don't waste everything. Default 5
    # gives 4 checkpoints across a 20-step run; user can raise SAVE_TOTAL_LIMIT
    # if they want longer history (each LoRA ckpt ~3.4GB on disk).
    SAVE_STEPS = _env_int("KERNELFORGE_STAGE1_SAVE_STEPS", 5)
    SAVE_TOTAL_LIMIT = _env_int("KERNELFORGE_STAGE1_SAVE_TOTAL_LIMIT", 3)

    config = GRPOConfig(
        learning_rate=2e-6,
        temperature=1.0,         # High exploration
        num_generations=NUM_GENERATIONS,
        beta=BETA,
        loss_type=LOSS_TYPE,
        max_completion_length=MAX_COMPLETION_LENGTH,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=MAX_STEPS,
        optim=OPTIMIZER,
        bf16=USE_BF16,
        report_to="wandb",
        output_dir=OUTPUT_DIR,
        logging_steps=1,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.05,
        use_vllm=USE_VLLM,
        vllm_mode="colocate" if USE_VLLM else "server",
        vllm_gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
    )

    trainer = TRLOOGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_from_env],
        rollout_func=rollout_func,
        args=config,
        train_dataset=dataset,
    )

    # Auto-resume from the newest VALID checkpoint in OUTPUT_DIR (HF Trainer
    # does NOT do this by default — requires resume_from_checkpoint). Passing
    # the specific path (not True) skips half-written checkpoints a walltime
    # kill left behind. Mirrors the stage2_rft.py fix from commit 5b59d2e.
    resume_ckpt = find_resumable_checkpoint(OUTPUT_DIR)
    if resume_ckpt:
        print(f"Stage 1: auto-resuming from checkpoint {resume_ckpt}")
    else:
        print(f"Stage 1: no valid checkpoint in {OUTPUT_DIR}; starting fresh.")
    print("Starting Stage 1 training...")
    trainer.train(resume_from_checkpoint=resume_ckpt if resume_ckpt else None)

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Stage 1 complete. Checkpoint saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
