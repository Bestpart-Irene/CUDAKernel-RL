# EXP-010 — Symbol-Shaped Reward (v3-symbol-shaped)

**Skill:** kernelforge-reward-design
**Date:** 2026-05-19
**Status:** designed, not yet promoted
**Parent master hash:** null (master still uninitialized after 2026-05-18 invalidation)

## Hypothesis

Adding a `-0.5` reward bucket for "compile=True but verifier symbol missing"
gives GRPO a directional gradient toward emitting the canonical contract
(`extern "C" void wcc_kernel(...)`). Under the current v2-shaped reward,
both "symbol missing" and "numerically wrong" return `0.0`, so the model
has no signal that the contract symbol matters.

## Failure mode being addressed

Diagnosed 2026-05-19 from job 6920683 (Stage 1 GRPO, 11 steps, ~3h on
d4053). The full trajectory:

- Compile rate climbed from 1/4 → 3/4 over the first 11 steps.
- `grad_norm` non-zero throughout (0.035–0.131).
- Reward oscillated in `[-0.75, -0.25]`; mean step reward never positive.
- **Every** compile=True rollout's `[VERIFIER_INLINE]` reported
  `Verification exception: Kernel FFI verification failed... undefined
  symbol: wcc_kernel`.
- I.e. the model was learning to emit syntactically-valid CUDA, but the
  Stage 2 SFT prior didn't fully transfer the `extern "C"` convention,
  and GRPO had no gradient toward it because v2-shaped collapses
  "symbol missing" and "numerically wrong" into the same `0.0` bucket.

## OLD reward (v2-shaped, copied verbatim from `openenv_env/reward.py`)

```python
# v2-shaped (current default)
if not compiled:
    return -1.0
if not correct:
    return 0.0       # ← lumps "symbol missing" with "numerically wrong"
if speedup_vs_compile > 1.05:
    return 3.0
if speedup_vs_eager > 1.05:
    return 2.0
return 1.0
```

## NEW reward (v3-symbol-shaped, this experiment)

```python
# v3-symbol-shaped
#   compile fail              -> -1.0
#   compile OK, symbol missing -> -0.5    ← NEW bucket
#   compile OK, symbol OK, wrong -> 0.0
#   compile OK, symbol OK, correct, no speedup -> 1.0
#   ... > eager  -> 2.0
#   ... > compile-> 3.0
```

Implementation requires `compute_reward` to know whether the verifier
loaded the entry symbol. That bit is detected in `task_support.compute_task_reward`
by substring-matching the eval result's `verifier_msg` against
`"undefined symbol"` / `"FFI verification failed"`, then passing a new
`symbol_loaded: bool` argument to `compute_reward`. v1 and v2 ignore that
argument (back-compat).

## Single variable changed

- `openenv_env/reward.py`: add `symbol_loaded: bool = True` parameter to
  `compute_reward`; add new branch for `_REWARD_VERSION == "v3-symbol-shaped"`.
- `training/task_support.py`: `compute_task_reward` now derives
  `symbol_loaded` from result's `verifier_msg` and forwards it.

No other files touched. No anti-hack edits.

## Integrity probes (run before any managed dispatch)

Per `kernelforge-reward-design` skill workflow:

1. **Known-good kernel** (compile=True, correct=True, speedup_vs_compile=2.0):
   v3 must still return `3.0`.
2. **Known-broken kernel** (compile=False, correct=False):
   v3 must still return `-1.0`.
3. **Reward-hack canary** (compile=True, correct=False, symbol_loaded=False):
   v3 must return `-0.5`, NOT 0.0 or higher.
4. **No-regression check** (compile=True, correct=False, symbol_loaded=True):
   v3 must return `0.0` (matches v2's "compiled but wrong" bucket).

All four probes implemented as assertions in `scripts/smoke_test.py`.

## Managed run plan

Single Stage 1 GRPO 20-step run on Modal / NU, env:

```
KERNELFORGE_REWARD_VERSION=v3-symbol-shaped
KERNELFORGE_STAGE1_INIT_CKPT=/scratch/.../stage2/checkpoint-N (newest)
KERNELFORGE_STAGE1_MAX_TURNS=1
KERNELFORGE_STAGE1_NUM_GENERATIONS=2
KERNELFORGE_STAGE1_MAX_STEPS=20
KERNELFORGE_STAGE1_BETA=0.0
KERNELFORGE_STAGE1_MAX_COMPLETION_LENGTH=2048
KERNELFORGE_EVAL_BACKEND=local
KERNELFORGE_STAGE1_BACKEND_FILTER=wcc
KERNELFORGE_ROLLOUT_DEBUG=1
KERNELFORGE_VERIFIER_DEBUG=1
```

## Comparability boundary

v3-symbol-shaped is a NEW reward version. **All rows before this
experiment are not directly comparable** to v3 rows. `results.tsv`
records the version in `reward_version` column;
`research/live/master.json` records it on promotion.

## Promotion gate

Per `kernelforge-reward-design` skill:
- v3 promotes only if BOTH `mean_reward` AND `pass_rate` improve vs
  the parent (currently null master, so cold-start rule applies).
- All 4 integrity probes must pass.
- Reviewer sign-off recorded in `research/notes.md`.

## Failure recovery

If v3 makes things worse (reward floor drops because all compile=True
rollouts hit the -0.5 bucket and stay there), revert to v2-shaped via
env var: `KERNELFORGE_REWARD_VERSION=v2-shaped`. Code path is unchanged;
only the env switch matters.
