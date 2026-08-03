"""Checkpoint discovery and validation helpers shared by the training stages.

Two failure modes these guard against (2026-08-03 audit):
  - Stage handoff gates that require config.json miss PEFT adapter-only
    checkpoints (stage1 saves only adapter_config.json) → later stages
    silently train the base model.
  - Auto-resume that fires on ANY checkpoint-* dir bricks every resubmission
    when a walltime SIGKILL leaves a half-written checkpoint (no
    trainer_state.json).
"""
from __future__ import annotations

import os


def has_model_files(path: str) -> bool:
    """Return True when `path` holds a loadable model checkpoint.

    Accepts either a full model (config.json) or a PEFT adapter-only dir
    (adapter_config.json) — model_loader._load_from_checkpoint handles both.
    """
    return os.path.isdir(path) and (
        os.path.isfile(os.path.join(path, "config.json"))
        or os.path.isfile(os.path.join(path, "adapter_config.json"))
    )


def find_resumable_checkpoint(output_dir: str) -> str | None:
    """Return the newest checkpoint-<step> dir in output_dir that is resumable.

    A checkpoint is resumable only if it contains trainer_state.json — a
    half-written save (walltime kill mid-save) lacks it and would crash
    trainer.train(resume_from_checkpoint=...). Invalid ones are skipped with
    a warning; returns None when no valid checkpoint exists.
    """
    if not os.path.isdir(output_dir):
        return None

    candidates: list[tuple[int, str]] = []
    for name in os.listdir(output_dir):
        suffix = name[len("checkpoint-"):]
        if name.startswith("checkpoint-") and suffix.isdigit():
            candidates.append((int(suffix), name))

    for _, name in sorted(candidates, reverse=True):
        ckpt = os.path.join(output_dir, name)
        if os.path.isfile(os.path.join(ckpt, "trainer_state.json")):
            return ckpt
        print(
            f"WARNING: skipping invalid checkpoint {ckpt} — no trainer_state.json "
            "(likely half-written by a walltime kill); trying the next-newest.",
            flush=True,
        )
    return None
