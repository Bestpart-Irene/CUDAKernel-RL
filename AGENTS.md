# Agent Instructions — KernelForge

This repo is a live RL-for-CUDA-kernels experiment workspace. Treat every change
as a benchmark experiment that must be comparable to a recorded baseline.

## Goal

Improve held-out kernel benchmark score (`mean_reward`, `pass_rate`,
`speedup_vs_orig`, `fast_p`) with disciplined, comparable single-change
experiments. Do not claim a win without a recorded managed run.

## TRL 0.29 invariants (verified 2026-05-18)

- `rollout_func` passed via `make_multi_turn_rollout` is **NOT invoked** by TRL
  0.29 GRPOTrainer. `[ROLLOUT_CALL]` never fires in any current run; TRL emits
  a runtime warning that the parameter is experimental.
- All training is effectively **single-turn**, regardless of
  `KERNELFORGE_STAGE1_MAX_TURNS` / `KERNELFORGE_STAGE3_MAX_TURNS`. Those env
  vars are currently a no-op.
- The active reward path is `reward_from_env(...)` in
  `training/multi_turn_rollout.py`. TRL calls it with completions, and the
  `env_reward len=0` branch inline-evaluates each via local compile + eval.
- Diagnostic env vars `KERNELFORGE_ROLLOUT_DEBUG=1` /
  `KERNELFORGE_VERIFIER_DEBUG=1` print under the `[ROLLOUT_INLINE]` /
  `[VERIFIER_INLINE]` tags from the inline branch (commit 469f6a4).
- Do NOT propose experiments that hinge on multi-turn feedback semantics —
  they will silently regress to single-turn. To re-enable multi-turn, pin to
  a TRL version that calls rollout_func or drive the turn loop manually
  inside `reward_from_env`.

## Resume-and-save discipline (MANDATORY, verified 2026-05-20)

Every `training/stage*.py` MUST satisfy ALL of:

1. `trainer.train()` is called with `resume_from_checkpoint=True` when the
   `output_dir` already contains a `checkpoint-*` subdir. HF Trainer does
   NOT auto-resume just because checkpoints exist — verified by job
   6920681 wasting 1.5h training from base while checkpoint-100 sat
   unused. Reference implementation: `training/stage2_rft.py` (commit
   5b59d2e), mirrored to stage1/stage3 in commit bf1b62a.

2. `save_steps` is small enough that a walltime kill loses ≤ ~25% of the
   run. Default cap: `save_steps <= max_steps // 4`. NU's 8h walltime is
   a hard ceiling for our account, so any multi-hour stage must
   checkpoint at least every quarter of its planned duration.

3. `save_steps` is env-overridable (`KERNELFORGE_STAGE{1,2,3}_SAVE_STEPS`)
   so operators can tighten cadence per run without code edits.

4. `output_dir` points into `/scratch/$KF_NETID/kernelforge/checkpoints/stage*`,
   never into `/home`. `/home` is quota-limited (~40GB) and a single LoRA
   checkpoint is ~10GB. The slurm scripts default this via
   `KERNELFORGE_STAGE*_OUTPUT` (commit 1721927). Verified by job 6906522
   et al. silently dying with `RaisedSignal:53` after `/home` saturated.

If you add a new stage or modify the trainer init, run through these
four checks. The reviewer agent rejects diffs that regress any of them.

## Hard Rules

- Default editable surface is `training/grpo_train.py` plus the immediate
  rollout files: `training/stage3_grpo.py`, `training/multi_turn_rollout.py`,
  `training/custom_grpo_trainer.py`, `configs/scaling_ladder.json`.
- Versioned evaluator freeze (2026-08-10): the evaluator —
  `eval_service/eval_core.py`, `evaluation/verifier.py`,
  `openenv_env/anti_hack.py` — may only change via a campaign-authorized,
  ledgered change. Every evaluator change must appear as a new
  `evaluator_sha` in the affected `research/results.tsv` rows (and in
  `research/live/master.json`), and it re-opens cold-start comparability:
  results recorded under different `evaluator_sha` values are not
  comparable. Silent modification remains forbidden.
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
  plus a one-paragraph note in `research/notes.md`. Ledger rows use the
  19-column schema (2026-08-10): `timestamp, experiment_id,
  parent_master_hash, runner, job_id, mean_reward, pass_rate,
  speedup_vs_orig, fast_p, reward_version, eval_split, seed_count,
  max_turns, eval_backend, evaluator_sha, task_pool_hash, init_ckpt,
  promote, comment` — fill unknown values with literal `null`;
  `task_pool_hash` comes from `scripts/task_pool_hash.py`.
- Promotion is local: a new master only takes over when its `mean_reward`
  beats current master on the same eval split, same seed count, same
  `max_turns`.
- Keep machine-local compatibility shims (Modal-only paths, local A100 mocks)
  out of promoted diffs.

## Source of Truth

- `research/results.tsv` — append-only local run ledger (19-column schema,
  2026-08-10; see Hard Rules for the column list)
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
