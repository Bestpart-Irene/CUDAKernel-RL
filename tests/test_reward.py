"""Tests for discrete milestone reward computation.

Default reward shape is v3-symbol-shaped: {-1, -0.5, 0, 1, 2, 3}.
v2-shaped (EXP-005): {-1, 0, 1, 2, 3}.
v1-discrete-milestone (legacy): {-1, -1, 1, 2, 3}.
Toggle via KERNELFORGE_REWARD_VERSION env var.
"""
import contextlib
import importlib
import os

import pytest

import openenv_env.reward as reward_mod
from openenv_env.reward import compute_reward, trloo_post_process


@contextlib.contextmanager
def _reward_version(version: str):
    """Reload the reward module under a specific KERNELFORGE_REWARD_VERSION.

    On exit, restores the ORIGINAL env value (or unsets it) and reloads once
    more, so the module lands back on whatever reward.py's actual default is
    — tests cannot leak reward-shape state into each other.
    """
    original = os.environ.get("KERNELFORGE_REWARD_VERSION")
    os.environ["KERNELFORGE_REWARD_VERSION"] = version
    try:
        importlib.reload(reward_mod)
        yield reward_mod.compute_reward
    finally:
        if original is None:
            os.environ.pop("KERNELFORGE_REWARD_VERSION", None)
        else:
            os.environ["KERNELFORGE_REWARD_VERSION"] = original
        importlib.reload(reward_mod)


def test_compile_fail():
    assert compute_reward(compiled=False, correct=False, speedup_vs_eager=0, speedup_vs_compile=0) == -1.0


def test_v2_shaped_compiled_but_wrong_is_zero():
    """v2-shaped: compile_pass + wrong → 0.0 (NOT -1.0)."""
    with _reward_version("v2-shaped") as fn:
        assert fn(compiled=True, correct=False, speedup_vs_eager=0, speedup_vs_compile=0) == 0.0


def test_v1_compiled_but_wrong_is_negative_one():
    """v1 legacy: compile_pass + wrong → -1.0 (lumped with compile_failed)."""
    with _reward_version("v1-discrete-milestone") as fn:
        assert fn(compiled=True, correct=False, speedup_vs_eager=0, speedup_vs_compile=0) == -1.0


def test_reload_helper_restores_default():
    """After a versioned block exits, the module must be back on reward.py's
    unset-env default (v3-symbol-shaped), not whatever the block used."""
    with _reward_version("v1-discrete-milestone"):
        pass
    assert reward_mod._REWARD_VERSION == "v3-symbol-shaped"


# --- v3-symbol-shaped tier (default, EXP-010) --------------------------------

def test_v3_symbol_missing_bucket_is_negative_half():
    """v3: compiled, wrong, verifier could NOT dlsym the symbol → -0.5."""
    with _reward_version("v3-symbol-shaped") as fn:
        assert fn(
            compiled=True, correct=False,
            speedup_vs_eager=0, speedup_vs_compile=0,
            symbol_loaded=False,
        ) == -0.5


def test_v3_compiled_but_wrong_symbol_ok_is_zero():
    """v3: compiled, symbol loaded, numerically wrong → 0.0."""
    with _reward_version("v3-symbol-shaped") as fn:
        assert fn(
            compiled=True, correct=False,
            speedup_vs_eager=0, speedup_vs_compile=0,
            symbol_loaded=True,
        ) == 0.0


def test_v3_correct_with_symbol_tiers():
    """v3: correct + symbol loaded follows the 1/2/3 speedup ladder."""
    with _reward_version("v3-symbol-shaped") as fn:
        assert fn(compiled=True, correct=True, speedup_vs_eager=1.0,
                  speedup_vs_compile=0.9, symbol_loaded=True) == 1.0
        assert fn(compiled=True, correct=True, speedup_vs_eager=1.5,
                  speedup_vs_compile=1.0, symbol_loaded=True) == 2.0
        assert fn(compiled=True, correct=True, speedup_vs_eager=2.0,
                  speedup_vs_compile=1.2, symbol_loaded=True) == 3.0


def test_correct_no_speedup():
    """Correct but speedup=1.0 (not > 1.05) -> reward 1.0."""
    r = compute_reward(compiled=True, correct=True, speedup_vs_eager=1.0, speedup_vs_compile=0.9)
    assert r == 1.0


def test_modest_speedup():
    """speedup_vs_eager=1.5 > 1.05 -> reward 2.0."""
    r = compute_reward(compiled=True, correct=True, speedup_vs_eager=1.5, speedup_vs_compile=0.9)
    assert r == 2.0


def test_large_speedup():
    """speedup_vs_eager=3.0 > 1.05 but speedup_vs_compile=1.0 not > 1.05 -> reward 2.0."""
    r = compute_reward(compiled=True, correct=True, speedup_vs_eager=3.0, speedup_vs_compile=1.0)
    assert r == 2.0


def test_slower_than_baseline():
    """Correct but speedup=0.5 < 1.05 -> reward 1.0."""
    r = compute_reward(compiled=True, correct=True, speedup_vs_eager=0.5, speedup_vs_compile=0.3)
    assert r == 1.0


def test_very_slow_clamped():
    """Correct but speedup=0.01 < 1.05 -> reward 1.0."""
    r = compute_reward(compiled=True, correct=True, speedup_vs_eager=0.01, speedup_vs_compile=0)
    assert r == 1.0


def test_nsight_ignored():
    """Nsight metrics are accepted but unused in discrete mode — same reward as without."""
    base = compute_reward(compiled=True, correct=True, speedup_vs_eager=2.0, speedup_vs_compile=1.0)
    with_nsight = compute_reward(
        compiled=True, correct=True, speedup_vs_eager=2.0, speedup_vs_compile=1.0,
        occupancy=0.8, mem_coalescing=0.9, warp_efficiency=0.7,
    )
    assert with_nsight == base == 2.0


def test_nsight_extreme_values():
    """Nsight with out-of-range values still produces same discrete reward."""
    r = compute_reward(
        compiled=True, correct=True, speedup_vs_eager=2.0, speedup_vs_compile=1.0,
        occupancy=1.5, mem_coalescing=-0.1, warp_efficiency=0.5,
    )
    assert r == 2.0


def test_beats_torch_compile():
    """speedup_vs_compile=1.2 > 1.05 -> reward 3.0."""
    r = compute_reward(compiled=True, correct=True, speedup_vs_eager=2.0, speedup_vs_compile=1.2)
    assert r == 3.0


def test_trloo_post_process_g4():
    """TRLOO scales by N/(N-1) = 4/3 for G=4."""
    advantages = [0.5, -0.3, 1.2, -0.8]
    scaled = trloo_post_process(advantages, n=4)
    scale = 4 / 3
    for orig, result in zip(advantages, scaled):
        assert result == pytest.approx(orig * scale, abs=1e-6)


def test_trloo_post_process_g1():
    """TRLOO with N=1 returns unchanged."""
    advantages = [0.5]
    assert trloo_post_process(advantages, n=1) == advantages


def test_trloo_post_process_g2():
    """TRLOO scales by 2/1 = 2.0 for G=2."""
    advantages = [0.3, -0.3]
    scaled = trloo_post_process(advantages, n=2)
    assert scaled[0] == pytest.approx(0.6, abs=1e-6)
    assert scaled[1] == pytest.approx(-0.6, abs=1e-6)
