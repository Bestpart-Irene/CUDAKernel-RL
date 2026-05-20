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


# EXP-005 / EXP-010: KERNELFORGE_REWARD_VERSION selects reward shape.
#   v1-discrete-milestone: legacy binary — compiled_but_wrong → -1.0 (same as compile fail).
#   v2-shaped:              compiled_but_wrong → 0.0 so GRPO groups can have non-zero variance.
#   v3-symbol-shaped (default, EXP-010): splits v2's compile-OK / wrong bucket into
#       symbol-missing (-0.5) vs symbol-OK-but-numerically-wrong (0.0). Gives GRPO a
#       directional gradient toward emitting the canonical contract symbol.
# Default kept at v3-symbol-shaped because v2 was empirically unable to push the
# model toward `extern "C" void wcc_kernel(...)` (job 6920683, 11 GRPO steps,
# 100% of compile=True rollouts hit dlsym fail with no gradient toward fixing it).
import os as _os
_REWARD_VERSION = _os.getenv("KERNELFORGE_REWARD_VERSION", "v3-symbol-shaped").strip()


def compute_reward(
    compiled: bool,
    correct: bool,
    speedup_vs_eager: float,
    speedup_vs_compile: float,
    occupancy: float | None = None,
    mem_coalescing: float | None = None,
    warp_efficiency: float | None = None,
    symbol_loaded: bool = True,
) -> float:
    """Return shaped milestone reward.

    Args:
        compiled: Whether the kernel compiled successfully.
        correct: Whether the kernel produces correct output.
        speedup_vs_eager: Speedup ratio vs torch.eager baseline.
        speedup_vs_compile: Speedup ratio vs torch.compile baseline.
        occupancy: SM occupancy (unused in discrete mode, kept for API compat).
        mem_coalescing: Memory coalescing (unused).
        warp_efficiency: Warp efficiency (unused).
        symbol_loaded: True if the verifier successfully dlsym'd the canonical
            entry point. v3-symbol-shaped uses this; v1 and v2 ignore it.

    Reward versions:
      v1-discrete-milestone:
        -1.0  if not compiled OR (compiled, correct=False)
         1/2/3 on the speedup ladder.
      v2-shaped (EXP-005):
        -1.0  if not compiled
         0.0  if compiled but correct=False (any reason)
         1/2/3 on the speedup ladder.
      v3-symbol-shaped (EXP-010, default):
        -1.0  if not compiled
        -0.5  if compiled, correct=False, AND verifier could NOT dlsym
              the canonical entry symbol (`undefined symbol: ...`)
         0.0  if compiled, correct=False, but symbol DID load (numerically wrong)
         1/2/3 on the speedup ladder.
    """
    # EXP-009 diagnostic: confirm reward chain is reached and which version is active.
    print(
        f"[COMPUTE_REWARD] compiled={compiled} correct={correct} "
        f"symbol_loaded={symbol_loaded} "
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
        if _REWARD_VERSION == "v3-symbol-shaped" and not symbol_loaded:
            # EXP-010: model compiled something but the canonical contract
            # symbol is missing (e.g. forgot `extern "C"` on wcc_kernel).
            # Worse than "numerically wrong" because there's literally no
            # callable kernel; better than compile-fail because syntax/types
            # are sound. The intermediate bucket gives GRPO gradient toward
            # fixing the contract.
            return -0.5
        # v2-shaped (and v3 when symbol DID load): compile-pass + symbol
        # is a partial signal, only the numerics are wrong.
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
