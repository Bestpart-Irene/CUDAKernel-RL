# KernelForge Playbook

Disciplined-experiment workflow for KernelForge, adapted from the
multi-autoresearch pattern (`burtenshaw/multiautoresearch`).

The contract: every code change is a single-hypothesis experiment, measured
on a managed runner, recorded in an append-only ledger, and promoted only
when it beats current master on a comparable eval.

## Map

| Path | Purpose |
|------|---------|
| [AGENTS.md](AGENTS.md) | Repo rulebook — hard rules, source-of-truth files, workflow |
| [.claude/agents/](.claude/agents/) | 7 subagent definitions Claude Code can dispatch |
| [.claude/skills/](.claude/skills/) | 3 reusable runbooks referenced by agents |
| [research/](research/) | Durable experiment memory + append-only ledger |
| [docs/SYSTEM_TRUTH.md](docs/SYSTEM_TRUTH.md) | Architecture truth (kept; unchanged) |
| [docs/KERNELFORGE_FINAL_PRD.md](docs/KERNELFORGE_FINAL_PRD.md) | PRD (kept; unchanged) |

## Agents

| Agent | Read/Write | Role |
|-------|-----------|------|
| `kernelforge` | read + dispatch | primary coordinator |
| `planner` | read-only | propose 1–3 fresh single-change experiments |
| `reviewer` | read-only | hard-rule and comparability checks |
| `researcher` | read-only + web | translate papers into single-change hypotheses |
| `reporter` | read-only | observe Modal / Northflank fleet |
| `memory-keeper` | write `research/*` only | durable notes + ledger + master pointer |
| `experiment-worker` | write code in worktree | exactly one change, one managed run |

## Skills

| Skill | When |
|-------|------|
| [kernelforge-managed-experiment](.claude/skills/kernelforge-managed-experiment/SKILL.md) | every single-change training experiment |
| [kernelforge-reward-design](.claude/skills/kernelforge-reward-design/SKILL.md) | only when the hypothesis IS a reward / anti-hack change |
| [kernelbench-eval](.claude/skills/kernelbench-eval/SKILL.md) | any eval-only invocation |

## Hard Rules (Summary — full text in AGENTS.md)

1. Default editable surface: `training/grpo_train.py` plus its immediate
   rollout files.
2. Never edit `eval_service/eval_core.py`, `evaluation/verifier.py`,
   `evaluation/compiler.py`, `evaluation/ablation.py`,
   `openenv_env/anti_hack.py`.
3. `openenv_env/reward.py` is quasi-frozen — only via
   `kernelforge-reward-design`.
4. Never train on the eval split.
5. One hypothesis change per run.
6. Managed run before claiming success.
7. Promotion only when `mean_reward` beats current master on the same eval
   split, seed count, and `max_turns`, with no `pass_rate` regression.

## Source of Truth

- `research/results.tsv` — append-only run ledger (one row per run)
- `research/live/master.json` — current promoted master
- `research/notes.md` — durable narrative
- `research/do-not-repeat.md` — ruled-out approaches
- `research/campaigns/` — thematic experiment groups
- `research/experiments/` — per-experiment specs

Do NOT use `git log main` to decide whether a hypothesis is fresh — main
also carries doc and infra commits.

## Standard Workflow (One Cycle)

1. **Plan** — `planner` reads the ledger and produces a ranked queue.
2. **Review** — `reviewer` flags duplicates, stale-master risk, multi-change
   patches.
3. **Dispatch** — `kernelforge` picks one and spawns
   `experiment-worker` in an isolated worktree.
4. **Execute** — worker applies the single change, runs one managed
   benchmark on Modal or Northflank, parses metrics.
5. **Record** — worker emits a structured summary; `memory-keeper` appends
   one row to `research/results.tsv` and one entry to `research/notes.md`.
6. **Promote (only if beats master)** — `memory-keeper` rewrites
   `research/live/master.json`.

## Managed Runner

- **Northflank-managed CoreWeave A100** — canonical reward-bearing eval
  (`eval_service/app.py`, `KERNELFORGE_EVAL_URL`)
- **Modal** — fallback / legacy launcher (`modal_train.py`, `modal_app.py`)
- **Local box** — not the default rig; only for smoke

## Pattern Origin

This playbook absorbs:

- multi-role agent system with strict read/write separation
- worktree isolation for the only code-mutating role
- single-truth files (`results.tsv` + `live/master.json`) over git history
- skills as reusable runbooks referenced by agents
- promotion gate tied to comparable evals, not to vibe checks
- "never claim a win without a managed run"

from `burtenshaw/multiautoresearch` (pre-training, post-training, inference
sub-projects), re-targeted to KernelForge's training surface, frozen
evaluator, and Modal / Northflank managed runners.
