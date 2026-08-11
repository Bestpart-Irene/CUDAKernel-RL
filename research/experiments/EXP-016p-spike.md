# EXP-016p-spike — Multi-step monotone-mean-reward verification spike

⚠️ VERIFICATION PHASE — NOT MASTER-PROMOTION-ELIGIBLE.

> **SUPERSEDED 2026-05-30 by CAMPAIGN-018-unify-contract.**
> This spike's anti-hack mechanism (source-level regex blocklist) was
> a stopgap that `do-not-repeat.md` 2026-05-28 explicitly flagged as
> incomplete. The replacement (CAMPAIGN-018) closes the hack channel
> structurally via the extern "C" contract on the ops6k path rather
> than via a regex layered on top of cpp_extension. EXP-018c is the
> verification-run spec that supersedes this one — same binary
> question, stronger evaluator. Do NOT run this spike as-is.

## Why this exists

Project history (EXP-001 through EXP-015'-A) has never produced a training
run where `mean_reward` increases across multiple GRPO steps under a
non-reward-hacked evaluator. Every prior "signal" has either been flat
(-1 collapse) or shown to be a hack channel (EXP-015'-A torch:: API
delegation, see `do-not-repeat.md` 2026-05-28).

Before committing to Phase 0 / Phase 1 / Phase 2 of the
"keep the original goal, accept research-level cost" path
(approx. 2-3 months calendar + ~$5-10K eval cash + 2-4 weeks throughput
engineering), this spike answers one binary question:

**Under a patched evaluator that closes the obvious torch-op hack channel,
on the simplest possible ops6k task pool, with all known-best GRPO
config, does `mean_reward` trend upward across ≥ 3 GRPO steps?**

If yes → invest in the full path.
If no → the bottleneck is not the hack, it's the RL signal density / SFT
prior / base-model capability, and we redirect to evolutionary search
or a narrower scope before spending more.

## Spec

- campaign: standalone (pre-investment gate)
- hypothesis: 50-step Stage 1 GRPO on 3-5 elementary ops6k tasks
  (vector_add, relu, scaled_add or equivalent), with the patched
  evaluator described below, produces a `mean_reward` curve whose last
  10 steps' mean is strictly greater than its first 10 steps' mean.
- parent master hash: null (cold-start; master was rolled back 2026-05-28)
- variable changed: NOT a single-variable run — this is a verification
  spike, not a promotion candidate. Composite changes:
  - `eval_service/eval_core.py::evaluate_ops6k_kernel_impl` —
    source-level regex scan added that rejects candidates containing
    direct torch-op call patterns (`torch::relu(`, `torch::softmax(`,
    `torch::nn::functional::`, `.relu()`, `.softmax(`, `at::add(`,
    `at::mul(`, etc.) — a curated stopgap blocklist, not a full
    dispatch interception. Stopgap is sufficient for the spike because
    we are testing whether RL can move the policy at all, not
    measuring a final pass_rate.
  - task pool restricted to 3-5 tasks with reference operators that
    are elementary enough that hand-written CUDA is feasible in
    < 1500 tokens (vector_add / relu / scaled_add or closest ops6k
    equivalents).
- runner: explorer-h200, eval_backend=local (subprocess-isolated)
- expected upside: settles a foundational question that gates ~$5-10K
  of downstream investment.
- duplicate check: no prior run has tested multi-step
  monotone-trajectory under a patched evaluator (see `do-not-repeat.md`
  2026-05-28 second entry).
- touches reward? no — reward.py unchanged, KERNELFORGE_REWARD_VERSION
  stays v3-symbol-shaped.

## Run config

| Param | Value | Rationale |
|---|---|---|
| Task pool | 3-5 simplest ops6k tasks | Decouple "learn correctness" from "learn speedup"; eliminate failure modes from task difficulty so the test is GRPO signal, not capability |
| SFT prior | Sakana 400 (existing) | Do NOT rebuild SFT yet — that's downstream investment we have not authorized |
| Reward version | v3-symbol-shaped (default) | unchanged |
| G (num_generations) | 2 | known good |
| beta (KL coef) | 0.0 | known good per EXP-009-B |
| max_turns | 1 | known good (multi-turn rollout_func is TRL 0.29 dead code anyway) |
| max_completion_length | 2048 | post-EXP-009-B floor |
| max_steps | 50 | enough to distinguish trend from noise; fits 2 × 8h slurm with auto-resume |
| eval_backend | local (subprocess) | H200 free; A100 timing not needed for spike (verification-only) |
| Platform | single H200, NU Explorer | $0 cash |
| Walltime | ~16h split across 2 × 8h slurm | auto-resume |

## Pass criteria

The spike PASSES (broader investment is authorized) iff ALL of:

1. `mean(reward_last_10_steps) > mean(reward_first_10_steps)` by an
   amount that survives a per-step bootstrap at 95% confidence.
2. At least one `correct=True` rollout occurs under the patched
   evaluator. This is a sanity check that the blocklist did not
   over-reject all kernels — a 0/50-step zero-correct run is
   consistent with "blocklist too aggressive", not "RL can't learn".
3. The bucket distribution of correct rollouts is plausible — code
   length not pathologically short (no 100-char "stub" hacks), runtime
   not pathologically fast (>10μs typical), at least one task
   produces a correct rollout (not all on a single trivially solvable
   task).

The spike FAILS (broader investment paused, redirect to evolutionary
search or scope reduction) iff:

- `mean_reward` is flat across all 50 steps (reproduces EXP-001..EXP-008
  pattern); OR
- `mean_reward` is non-flat but shows no upward trend (random
  fluctuation); OR
- All `correct=True` rollouts in the spike are themselves caught by a
  post-hoc deeper anti-hack scan (i.e. the blocklist was insufficient
  and we are back where we started).

## Run

EXP-016p-spike was superseded by EXP-018c before it ran (see banner above).
See [EXP-018c](EXP-018c-verification-extern-c.md) for the verification-run record (v1 VOID; rerun pending).
