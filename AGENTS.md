# Agent Instructions — KernelForge

This repo is a live RL-for-CUDA-kernels experiment workspace. Treat every change
as a benchmark experiment that must be comparable to a recorded baseline.

## Goal

Improve held-out kernel benchmark score (`mean_reward`, `pass_rate`,
`speedup_vs_orig`, `fast_p`) with disciplined, comparable single-change
experiments. Do not claim a win without a recorded managed run.

## Hard Rules

- Default editable surface is `training/grpo_train.py` plus the immediate
  rollout files: `training/stage3_grpo.py`, `training/multi_turn_rollout.py`,
  `training/custom_grpo_trainer.py`, `configs/scaling_ladder.json`.
- Never modify the frozen evaluator: `eval_service/eval_core.py`,
  `evaluation/verifier.py`, `evaluation/compiler.py`, `evaluation/ablation.py`.
- Never modify `openenv_env/anti_hack.py` unless the task is explicitly a
  reward-integrity change with reviewer sign-off.
- Treat `openenv_env/reward.py` as quasi-frozen: only edit when the experiment
  IS a reward-design experiment (use the `kernelforge-reward-design` skill).
- Never train on the eval split. The held-out eval set is whatever
  `evaluation/eval_model.py` consumes; do not feed those tasks into rollouts.
- Make exactly one hypothesis change per run.
- Run the timed benchmark (`scripts/run_benchmark.py` or the managed Modal /
  Northflank path) before claiming success.
- Record every completed run with one row appended to `research/results.tsv`
  plus a one-paragraph note in `research/notes.md`.
- Promotion is local: a new master only takes over when its `mean_reward`
  beats current master on the same eval split, same seed count, same
  `max_turns`.
- Keep machine-local compatibility shims (Modal-only paths, local A100 mocks)
  out of promoted diffs.

## Source of Truth

- `research/results.tsv` — append-only local run ledger
- `research/live/master.json` — current promoted master (hash, metrics, date)
- `research/notes.md` — durable narrative
- `research/do-not-repeat.md` — failed approaches and why
- `research/campaigns/` — multi-experiment campaigns
- `research/experiments/` — per-experiment specs and outcomes
- `docs/SYSTEM_TRUTH.md` and `docs/KERNELFORGE_FINAL_PRD.md` — architecture
  reference, not experiment ledger

Do not use git `main` history to decide whether a hypothesis is fresh — main
also carries doc and infra commits. Compare against `research/live/master.json`
and `research/results.tsv`.

## Agent Roster

Roles live in `.claude/agents/`:

- `kernelforge` — primary coordinator (only role that launches subagents)
- `planner` — read-only, fresh experiment queue
- `reviewer` — read-only, hard-rule and comparability checks
- `researcher` — read-only literature scout (CUDA-Agent, Dr. Kernel, KernelGYM,
  KernelBench priors)
- `reporter` — read-only Modal / Northflank fleet observer
- `memory-keeper` — only writer of `research/*.md`
- `experiment-worker` — only code-mutating role; must run in an isolated git
  worktree

Concurrency: active `experiment-worker` count must never exceed real GPU
capacity (Northflank A100 slots for eval, plus H200/H100 slots for training).

## Managed Runner

The default benchmark path is **Northflank-managed CoreWeave A100** for
reward-bearing eval and **Modal** as the fallback / legacy launcher. Local
CUDA boxes are not the canonical rig.

Per experiment:

1. Refresh from local master: read `research/live/master.json` and the latest
   row of `research/results.tsv`.
2. Edit only the assigned surface (see Hard Rules).
3. Launch one managed training run (Modal or Northflank, whichever the
   campaign pins).
4. Stream logs; parse final `mean_reward` / `pass_rate` /
   `speedup_vs_orig`.
5. Append one row to `research/results.tsv`.
6. Hand off the run summary to `memory-keeper` for `research/notes.md`.

## Standard Workflow

1. `planner` proposes 1–3 fresh single-change experiments against current
   master.
2. `reviewer` checks for hard-rule violations, duplicates, stale master,
   multi-change patches.
3. `kernelforge` selects one experiment and spawns an `experiment-worker` in
   an isolated worktree.
4. Worker makes the single change, runs the managed benchmark, parses
   metrics, returns a structured summary.
5. `memory-keeper` folds the summary into `research/notes.md`,
   `research/results.tsv`, and the campaign / do-not-repeat file.
6. If the run beats current master, `memory-keeper` updates
   `research/live/master.json`.

## Literature Scouting Mode

When the task is paper research rather than a timed run:

- `researcher` may add or update `research/paper-ideas.md`.
- Translate every paper claim into one minimum-change `training/grpo_train.py`
  hypothesis that can be benchmarked cleanly.
- Reject ideas already ruled out in `research/do-not-repeat.md`.
- Never call a paper-derived idea a win without a managed benchmark run.

## Local Skills

- `kernelforge-managed-experiment` — single managed run discipline.
- `kernelforge-reward-design` — only path for reward / anti-hack changes.
- `kernelbench-eval` — held-out eval, pass@k, fast_p reporting.

## Cross-Harness Notes

The agent contract is the same regardless of harness. If we ever publish
OpenCode (`opencode.json` + `.opencode/agent/`) or Codex (`.codex/agents/`)
mirrors, they must follow the same hard rules, source-of-truth files, and
managed-runner workflow.
