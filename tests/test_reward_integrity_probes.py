"""EXP-019 integrity probes + regression tests for the anti-hack fixes.

The kernelforge-reward-design skill requires three canaries before any
reward/anti-hack change is allowed near master:
  (i)   known-good result        -> reward 3
  (ii)  known-broken (compile fail) -> reward -1
  (iii) ref-copy / hack_suspected  -> must NOT score positive

All probes are CPU-only synthetic eval-result dicts run against the reward
mapping (`openenv_env.reward`) and the production dict path
(`training.task_support.compute_task_reward`). The probe assertions are
version-robust: they hold under v1/v2/v3 reward shapes, so a stray
KERNELFORGE_REWARD_VERSION in the environment cannot fake a pass.

Regression tests cover EXP-019 fixes a-d
(see research/experiments/EXP-019-anti-hack-integrity.md).
"""
import math
import types

import pytest

from openenv_env.anti_hack import (
    check_not_noop,
    scan_forbidden_symbols,
    scan_source_forbidden,
)
from openenv_env.reward import compute_reward, validate_eval_result
from training.task_support import compute_task_reward


# ---------------------------------------------------------------------------
# Skill canaries (i)-(iii)
# ---------------------------------------------------------------------------


def _result(**overrides):
    base = {
        "compiles": False,
        "correct": False,
        "speedup_vs_orig": 0.0,
        "speedup_vs_dg": 0.0,
        "error": "",
        "verifier_msg": "",
    }
    base.update(overrides)
    return base


def test_probe_known_good_scores_three():
    """(i) compiles + correct + beats both baselines -> reward 3.0."""
    good = _result(compiles=True, correct=True, speedup_vs_orig=2.0, speedup_vs_dg=1.5)
    assert compute_task_reward(good) == 3.0

    validated = validate_eval_result(good)
    assert compute_reward(
        compiled=validated["compiles"],
        correct=validated["correct"],
        speedup_vs_eager=validated["speedup_vs_orig"],
        speedup_vs_compile=validated["speedup_vs_dg"],
    ) == 3.0


def test_probe_known_broken_scores_minus_one():
    """(ii) compile failure -> reward -1.0."""
    broken = _result(error="Compilation failed: identifier 'foo' is undefined")
    assert compute_task_reward(broken) == -1.0

    validated = validate_eval_result(broken)
    assert compute_reward(
        compiled=validated["compiles"],
        correct=validated["correct"],
        speedup_vs_eager=validated["speedup_vs_orig"],
        speedup_vs_compile=validated["speedup_vs_dg"],
    ) == -1.0


def test_probe_ref_copy_canary_never_positive():
    """(iii) a hack_suspected result must NOT score positive.

    Mirrors the eval-core anti-hack layer: on a flagged candidate it sets
    correct=False and error='Anti-hack: ...' while inflated speedups may still
    sit in the dict. The reward must ignore the speedups entirely.
    """
    hacked = _result(
        compiles=True,
        correct=False,
        speedup_vs_orig=4.2,
        speedup_vs_dg=2.0,
        error="Anti-hack: Output is identical to input[0] — likely passthrough",
        verifier_msg="Anti-hack: Output is identical to input[0] — likely passthrough",
    )
    assert compute_task_reward(hacked) <= 0.0


def test_probe_hack_suspected_symbol_scan_never_positive():
    """(iii'') a fail-closed symbol-scan verdict must not score positive."""
    hacked = _result(
        compiles=True,
        correct=False,
        speedup_vs_orig=3.0,
        speedup_vs_dg=1.5,
        error="Anti-hack post-link scan: hack_suspected: symbol scan failed",
        verifier_msg="Anti-hack post-link scan: hack_suspected: symbol scan failed",
    )
    assert compute_task_reward(hacked) <= 0.0


# ---------------------------------------------------------------------------
# Fix (a) — NaN/inf speedup guard in compute_reward
# ---------------------------------------------------------------------------


def test_fix_a_inf_speedup_maps_to_correctness_tier():
    """inf speedup used to satisfy `> 1.05` and mint reward 3.0."""
    r = compute_reward(
        compiled=True, correct=True,
        speedup_vs_eager=float("inf"), speedup_vs_compile=float("inf"),
    )
    assert r == 1.0


def test_fix_a_nan_speedup_maps_to_correctness_tier():
    r = compute_reward(
        compiled=True, correct=True,
        speedup_vs_eager=float("nan"), speedup_vs_compile=float("nan"),
    )
    assert r == 1.0


def test_fix_a_negative_inf_and_non_numeric():
    assert compute_reward(
        compiled=True, correct=True,
        speedup_vs_eager=float("-inf"), speedup_vs_compile=float("-inf"),
    ) == 1.0
    assert compute_reward(
        compiled=True, correct=True,
        speedup_vs_eager="bogus", speedup_vs_compile=None,  # type: ignore[arg-type]
    ) == 1.0


def test_fix_a_finite_speedup_still_earns_its_tier():
    """The guard must not disturb legitimate finite measurements."""
    r = compute_reward(
        compiled=True, correct=True,
        speedup_vs_eager=2.0, speedup_vs_compile=float("inf"),
    )
    assert r == 2.0  # inf compile-speedup clamped; finite eager tier stands


def test_fix_a_production_dict_path_with_inf():
    """The exact path that used to yield 3.0: compute_task_reward forwards
    float(x or 0.0), and float('inf') is truthy and passes through."""
    poisoned = _result(
        compiles=True, correct=True,
        speedup_vs_orig=float("inf"), speedup_vs_dg=float("inf"),
    )
    assert compute_task_reward(poisoned) == 1.0


def test_fix_a_validate_eval_result_still_clamps():
    out = validate_eval_result(
        _result(compiles=True, correct=True,
                speedup_vs_orig=float("inf"), speedup_vs_dg=float("nan"))
    )
    assert out["speedup_vs_orig"] == 0.0
    assert out["speedup_vs_dg"] == 0.0


# ---------------------------------------------------------------------------
# Fix (b) — check_not_noop dead threshold / explicit units
# ---------------------------------------------------------------------------


def test_fix_b_sub_microsecond_runtime_fails():
    passed, reason = check_not_noop(0.0005)  # 0.5 us
    assert not passed
    assert "floor" in reason


def test_fix_b_dead_zone_now_caught():
    """1.5 us sits in [1 us, 2 us): physically impossible for real work but
    PASSED under the old 0.001 ms threshold."""
    passed, _ = check_not_noop(0.0015)
    assert not passed


def test_fix_b_zero_negative_fail_closed():
    assert not check_not_noop(0.0)[0]
    assert not check_not_noop(-0.5)[0]


def test_fix_b_non_finite_fails_closed():
    """NaN < 0.001 is False, so a NaN runtime used to PASS the no-op check."""
    assert not check_not_noop(float("nan"))[0]
    assert not check_not_noop(float("inf"))[0]


def test_fix_b_non_numeric_fails_closed():
    assert not check_not_noop("bogus")[0]  # type: ignore[arg-type]
    assert not check_not_noop(None)[0]  # type: ignore[arg-type]


def test_fix_b_plausible_runtimes_still_pass():
    assert check_not_noop(0.05)[0]   # 50 us
    assert check_not_noop(5.0)[0]    # 5 ms
    assert check_not_noop(0.002)[0]  # exactly at the 2 us floor -> plausible


# ---------------------------------------------------------------------------
# Fix (c) — scan_forbidden_symbols fails closed on nm failure
# ---------------------------------------------------------------------------


def _fake_proc(returncode=0, stdout="", stderr=""):
    return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_fix_c_missing_binary_fails_closed():
    result = scan_forbidden_symbols("/nonexistent/definitely_missing_kernel.so")
    assert isinstance(result, str)
    assert "hack_suspected" in result


def test_fix_c_nm_unavailable_fails_closed(monkeypatch):
    import openenv_env.anti_hack as ah

    def _raise(*args, **kwargs):
        raise FileNotFoundError("nm not on PATH")

    monkeypatch.setattr(ah.subprocess, "run", _raise)
    result = scan_forbidden_symbols("whatever.so")
    assert isinstance(result, str)
    assert "hack_suspected" in result
    assert "nm" in result


def test_fix_c_nm_nonzero_exit_fails_closed(monkeypatch):
    import openenv_env.anti_hack as ah

    monkeypatch.setattr(
        ah.subprocess, "run",
        lambda *a, **k: _fake_proc(returncode=1, stderr="nm: no symbols"),
    )
    result = scan_forbidden_symbols("broken.so")
    assert isinstance(result, str)
    assert "hack_suspected" in result
    assert "exit 1" in result


def test_fix_c_clean_scan_still_passes(monkeypatch):
    import openenv_env.anti_hack as ah

    monkeypatch.setattr(
        ah.subprocess, "run",
        lambda *a, **k: _fake_proc(stdout="0000000000001000 T run_kernel\n"),
    )
    assert scan_forbidden_symbols("clean.so") is None


def test_fix_c_forbidden_symbol_still_detected(monkeypatch):
    import openenv_env.anti_hack as ah

    monkeypatch.setattr(
        ah.subprocess, "run",
        lambda *a, **k: _fake_proc(stdout="U _ZN5torch3fooEv\n"),
    )
    result = scan_forbidden_symbols("dirty.so")
    assert result is not None
    assert "torch" in result


# ---------------------------------------------------------------------------
# Fix (d) — macro-indirect include / identifier bypass
# ---------------------------------------------------------------------------

LEGIT_KERNEL = """\
// CU_FLAGS: --use_fast_math
#include <cuda_runtime.h>
#include <math.h>
#define BLOCK 256
/* tiled elementwise ELU */
__global__ void elu_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) y[i] = x[i] > 0.f ? x[i] : expf(x[i]) - 1.f;
}
extern "C" void run_kernel(const float* x, float* y, int n) {
    elu_kernel<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(x, y, n);
}
"""


def test_fix_d_legit_kernel_still_passes():
    assert scan_source_forbidden(LEGIT_KERNEL) is None


def test_fix_d_direct_include_still_caught():
    assert scan_source_forbidden('#include <torch/extension.h>\n') is not None


def test_fix_d_computed_include_rejected():
    src = "#define TORCH_HDR <torch/extension.h>\n#include TORCH_HDR\n"
    reason = scan_source_forbidden(src)
    assert reason is not None
    assert "macro-indirect" in reason


def test_fix_d_object_macro_alias_rejected():
    src = "#define NS torch\nvoid f() { NS::relu(0); }\n"
    reason = scan_source_forbidden(src)
    assert reason is not None
    assert "macro-indirect" in reason


def test_fix_d_chained_object_macros_rejected():
    src = (
        "#define P1 tor\n"
        "#define P2 ch\n"
        "#define NS P1P2\n"  # not a real paste, but P1P2 as one identifier stays clean
        "#define REALNS torch\n"
        "void f() { REALNS::relu(0); }\n"
    )
    assert scan_source_forbidden(src) is not None


def test_fix_d_token_pasting_rejected():
    src = "#define NS tor##ch\nvoid f() { NS::relu(0); }\n"
    reason = scan_source_forbidden(src)
    assert reason is not None
    assert "##" in reason


def test_fix_d_function_like_paste_macro_rejected():
    src = "#define CAT(a, b) a##b\nvoid f() { CAT(tor, ch)::relu(0); }\n"
    reason = scan_source_forbidden(src)
    assert reason is not None
    assert "##" in reason


def test_fix_d_line_splice_rejected():
    src = '#include <tor\\\nch/extension.h>\n'
    assert scan_source_forbidden(src) is not None


def test_fix_d_ucn_identifier_rejected():
    src = "void f() { tor\\u0063h::relu(0); }\n"
    assert scan_source_forbidden(src) is not None


def test_fix_d_macro_blowup_fails_closed():
    src = "#define A A A A A A A A\nA\n"
    reason = scan_source_forbidden(src)
    assert reason is not None
    assert "size cap" in reason


def test_fix_d_comment_split_identifier_not_false_positive():
    """`tor/**/ch` is two tokens to the compiler (comment -> space) and must
    NOT be flagged; only a `##` paste joins across a comment."""
    src = "void f() { int tor/**/ch = 0; (void)torch; }\n"
    # NB: 'torch' bare identifier is fine — only 'torch::' is forbidden.
    assert scan_source_forbidden(src) is None
