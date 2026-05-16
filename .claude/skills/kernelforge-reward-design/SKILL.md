---
name: kernelforge-reward-design
description: "Use when, and only when, the experiment is a reward-shape or anti-hack change in openenv_env/reward.py or openenv_env/anti_hack.py. Adds extra integrity checks before such a change is allowed onto current master."
version: 1.0.0
---

Most KernelForge experiments must NOT touch reward. Use this skill only when
the explicit hypothesis is a reward-design or anti-hack change.

## Why a Separate Skill

The discrete milestone reward `{-1, 1, 2, 3}` is part of the comparability
contract for every other experiment in `research/results.tsv`. A reward
change silently invalidates all prior comparisons. Treat the change like
shipping a new evaluator: it forks the ledger.

Sources for reward shape:

- `openenv_env/reward.py`
- `openenv_env/anti_hack.py`
- CUDA Agent paper, Section on milestone reward ablation
- Dr. Kernel paper, anti-hack failure modes

## Workflow

1. Confirm the experiment spec says "reward design" or "anti-hack" and is
   reviewer-signed-off. If not, stop and route to `reviewer`.
2. Record the OLD reward / anti-hack behavior:
   - copy the relevant function bodies into the experiment spec
     under `research/experiments/<id>.md`
   - record the rationale and the failure mode being addressed
3. Implement the single reward change in `openenv_env/reward.py` or the
   single anti-hack check in `openenv_env/anti_hack.py`. Do NOT touch
   anything else.
4. Run integrity probes BEFORE the managed training run:
   - synthetic kernel: known-good submission must still return reward 3
   - synthetic kernel: known-broken submission must still return reward -1
   - reward-hack canary: a submission that copies reference output without
     a real kernel must NOT score positive
5. Local smoke as usual.
6. Run exactly one managed training run.
7. When recording the result in `research/results.tsv`, mark the row with
   `reward_version=<new>` and treat all prior rows as not directly
   comparable. `memory-keeper` must also append a new section to
   `research/notes.md` explaining the reward-version boundary.

## Promotion Rules

- A reward-design experiment can only promote to master if it both:
  - improves `mean_reward` AND `pass_rate` on the same eval split, and
  - passes all three integrity probes above.
- Promotion under this skill ALSO requires reviewer sign-off in
  `research/notes.md` and a new entry in `research/live/master.json`
  recording the reward version.

## Forbidden

- editing `eval_service/eval_core.py`
- editing `evaluation/verifier.py`
- changing seed count or `max_turns` in the same experiment
- claiming a reward change is a win because `mean_reward` rose while
  `pass_rate` fell (reward inflation, not improvement)
