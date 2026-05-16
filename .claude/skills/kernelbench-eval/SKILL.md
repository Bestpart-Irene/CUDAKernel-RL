---
name: kernelbench-eval
description: "Run the frozen KernelForge / KernelBench-style evaluation harness and report mean_reward, pass_rate, speedup_vs_orig, and fast_p without modifying the evaluator."
version: 1.0.0
---

Use this for any eval-only invocation: a reporter checking a job's metrics,
a worker producing its post-run summary, or a reviewer verifying a claimed
score.

The evaluator is FROZEN. Never modify `eval_service/eval_core.py`,
`evaluation/verifier.py`, `evaluation/compiler.py`, or `evaluation/ablation.py`
inside this skill.

## Workflow

1. Pick the eval entrypoint:
   - local A100 box (rare): `python evaluation/eval_model.py --model-path
     <checkpoint> --split heldout`
   - Modal fallback: `python scripts/run_benchmark.py --backend modal
     --model-path <checkpoint>`
   - Northflank / CoreWeave canonical: POST to the `eval_service/app.py`
     endpoint at `KERNELFORGE_EVAL_URL`
2. Confirm the eval split. The held-out split is whatever
   `evaluation/eval_model.py` uses as default. Never feed held-out tasks
   into training rollouts.
3. Use 5-seed correctness checking (matches CUDA Agent paper).
4. Capture all of:
   - `mean_reward` (the discrete `{-1,1,2,3}` mean)
   - `pass_rate` (fraction of tasks with reward ≥ 1)
   - `speedup_vs_orig` (geometric mean vs reference)
   - `speedup_vs_dg` (when DoubleGraph baseline is in play)
   - `fast_p` (KernelBench metric, when computed)
   - `compile_pass_rate`
5. Report:
   - which eval entrypoint was used
   - which eval split
   - seed count
   - `max_turns` (default 3)
   - the metrics above
   - a one-line comparison against `research/live/master.json` if available

## Forbidden

- editing the evaluator
- training on eval tasks
- mixing eval splits across rows in `research/results.tsv`
- reporting `mean_reward` without `pass_rate` and `speedup_vs_orig` — the
  three move together, and a single number is not a comparable claim

## Quick Checks

- `python scripts/compare_results.py --tail 5` — recent ledger rows
- `python scripts/run_benchmark.py --help` — confirm flags match the spec
- inspect `KERNELFORGE_EVAL_BACKEND` and `KERNELFORGE_EVAL_URL` to confirm
  which backend will actually answer
