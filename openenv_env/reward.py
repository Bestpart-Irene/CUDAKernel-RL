"""Reward helpers: discrete milestone {-1, 1, 2, 3} + TRLOO post-process.

Discrete rewards per CUDA Agent ablation (96.8% vs 60.4% faster rate over continuous).
Milestones normalize across problem difficulty — beating torch.compile on a hard problem
and an easy problem both earn the same r=3.
"""
from __future__ import annotations

import math

_REQUIRED_EVAL_KEYS = {"compiles", "correct", "speedup_vs_orig", "speedup_vs_dg", "error"}


def validate_eval_result(result: dict) -> dict:
    """Validate Modal evaluate_kernel return schema. Missing/invalid → safe defaults."""
    missing = _REQUIRED_EVAL_KEYS - set(result)
    if missing:
        return {"compiles": False, "correct": False, "speedup_vs_orig": 0.0,
                "speedup_vs_dg": 0.0, "error": f"missing keys: {missing}"}
    out = dict(result)
    # Clamp NaN/inf speedups to 0
    for k in ("speedup_vs_orig", "speedup_vs_dg"):
        v = out.get(k, 0.0)
        if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
            out[k] = 0.0
    return out


# EXP-005: KERNELFORGE_REWARD_VERSION selects reward shape.
#   v1-discrete-milestone: legacy binary — compiled_but_wrong → -1.0 (same as compile fail).
#   v2-shaped (default):    compiled_but_wrong → 0.0 so GRPO groups can have non-zero variance.
# Default kept at v2-shaped because v1 is empirically a dead-end on this stack.
import os as _os
_REWARD_VERSION = _os.getenv("KERNELFORGE_REWARD_VERSION", "v2-shaped").strip()


def compute_reward(
    compiled: bool,
    correct: bool,
    speedup_vs_eager: float,
    speedup_vs_compile: float,
    occupancy: float | None = None,
    mem_coalescing: float | None = None,
    warp_efficiency: float | None = None,
) -> float:
    """Return shaped milestone reward.

    reward_version `v2-shaped` (EXP-005). Splits the old v1's binary -1
    bucket into compile-failed (-1) vs compiled-but-wrong (0) so GRPO
    groups can have non-zero variance even before any rollout is fully
    correct. EXP-004 v3 (slurm 6876541) showed the 4 WCC tasks split
    2/4 compile_failed + 2/4 compiled_but_wrong; under v1 both went to
    -1 and grad_norm=0. Under v2-shaped the same mix gives a mean of
    -0.5 with std=0.5 per pair, unblocking GRPO.

    Args:
        compiled: Whether the kernel compiled successfully.
        correct: Whether the kernel produces correct output.
        speedup_vs_eager: Speedup ratio vs torch.eager baseline.
        speedup_vs_compile: Speedup ratio vs torch.compile baseline.
        occupancy: SM occupancy (unused in discrete mode, kept for API compat).
        mem_coalescing: Memory coalescing (unused).
        warp_efficiency: Warp efficiency (unused).

    Returns:
        -1.0: compile failure (extraction empty OR nvcc error).
         0.0: compiled_but_wrong — kernel runs, output != reference.
         1.0: correct but not faster than baselines.
         2.0: correct and faster than eager PyTorch (>5%).
         3.0: correct and faster than torch.compile (>5%).
    """
    # EXP-009 diagnostic: confirm reward chain is reached and which version is active.
    print(
        f"[COMPUTE_REWARD] compiled={compiled} correct={correct} "
        f"sv_eager={speedup_vs_eager} sv_compile={speedup_vs_compile} "
        f"version={_REWARD_VERSION}",
        flush=True,
    )
    if not compiled:
        return -1.0
    if not correct:
        if _REWARD_VERSION == "v1-discrete-milestone":
            # Legacy binary: lumps compiled_but_wrong with compile_failed.
            return -1.0
        # v2-shaped (default): compile-pass alone is a partial signal.
        # Kevin (arXiv 2507.11948) shows this unlocks small-model GRPO
        # cold-start; EXP-004 v3 confirmed on this stack.
        return 0.0

    # Discrete milestones (highest matching tier wins)
    if speedup_vs_compile > 1.05:
        return 3.0
    if speedup_vs_eager > 1.05:
        return 2.0
    return 1.0


def trloo_post_process(advantages: list[float], n: int) -> list[float]:
    """Scale GRPO advantages by N/(N-1) to correct gradient shrinkage.

    Dr. Kernel (arXiv 2602.05885) proves GRPO's self-inclusion bias
    shrinks expected gradients by (1 - 1/N). With G=4, that is 25%.
    This post-process is a drop-in fix for TRL GRPOTrainer.
    """
    if n <= 1:
        return advantages
    scale = n / (n - 1)
    return [a * scale for a in advantages]
