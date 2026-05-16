---
name: kernelforge-managed-experiment
description: "Run one KernelForge benchmark experiment safely on Modal or Northflank-managed CoreWeave. Use when a planner, reviewer, or experiment worker is preparing, auditing, launching, or recording a single training change against the current local promoted master."
version: 1.0.0
---

Use this for any single KernelForge experiment that should result in exactly
one managed run (Modal or Northflank), one parsed metric, and one row in
`research/results.tsv`.

## Workflow

1. Refresh from local master:
   - read `research/live/master.json` for current hash and metrics
   - read the latest row of `research/results.tsv`
   - if either is missing for the current eval split, stop and report
2. Edit only the assigned surface for the single intended hypothesis. Default:
   `training/grpo_train.py` plus at most one of
   `training/stage3_grpo.py`, `training/multi_turn_rollout.py`,
   `training/custom_grpo_trainer.py`, or `configs/scaling_ladder.json`.
3. Local smoke (when the spec includes it):
   - `python scripts/smoke_test.py`
   - `python scripts/run_pipeline.py --smoke`  (if available)
4. Launch exactly one managed run.
   - Modal training path:
     - `modal run modal_train.py::train <args>` (use the spec's exact args)
     - logs: `modal app logs <APP_NAME>`
   - Northflank / CoreWeave eval-bearing path:
     - hit the deployed `eval_service/app.py` endpoint per
       `KERNELFORGE_EVAL_URL`
     - confirm `KERNELFORGE_EVAL_BACKEND=coreweave`
5. Stream logs to a stable path:
   - export `KERNELFORGE_LOG_PATH=research/live/run-<experiment-id>.log`
   - capture stdout to that path
6. Parse the final metric. Required fields: `mean_reward`, `pass_rate`,
   `speedup_vs_orig`. Optional but preferred: `fast_p`, `correct_count`,
   `compile_pass_rate`.
7. Record the run locally:
   - append one row to `research/results.tsv` with: timestamp,
     experiment-id, parent-master-hash, runner, job-id,
     `mean_reward`, `pass_rate`, `speedup_vs_orig`, `fast_p`,
     promote-decision, one-line comment.
8. Promotion is local. Only rewrite `research/live/master.json` when:
   - same eval split as current master
   - same seed count
   - same `max_turns`
   - `mean_reward` beats current master (and `pass_rate` did not regress)

## Guardrails

- Treat `research/live/master.json` plus the latest rows of
  `research/results.tsv` as the comparable base, not git `main`.
- Never launch a second job for the same experiment id unless you have a
  specific reason and intentionally override the duplicate check.
- Never launch `modal run modal_train.py::prepare` (or equivalent dataset /
  warmup bootstrap) from an experiment-scoped worktree. Bootstrap is shared
  work, not per-experiment work.
- If the workspace looks stale against current master, stop and rewrite the
  experiment rather than rationalizing the mismatch.
- A `mean_reward` improvement paired with a `pass_rate` regression is NOT a
  promotion — flag for `reviewer`.

## Fast Checks Before Launch

- `python scripts/compare_results.py --tail 10` — sanity-check the recent
  ledger.
- `python scripts/run_benchmark.py --dry-run` — confirm the eval invocation
  the spec pins is the one the runner will use.
- `git -C $(git rev-parse --show-toplevel) diff` — confirm exactly one
  variable changed.

## Comparability Rules

A run is only comparable to current master when ALL of these match:

- eval split (`evaluation/eval_model.py` --split)
- seed count
- `max_turns` (default 3)
- model checkpoint family (e.g. `Qwen3-Coder-30B-A3B-Instruct`)
- reward shape (`openenv_env/reward.py` unchanged unless it IS the
  experiment, see `kernelforge-reward-design`)
- evaluator (`eval_service/eval_core.py`) — must be unchanged
