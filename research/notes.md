# KernelForge Research Notes

Durable narrative for KernelForge experiments. Append, do not rewrite.

`memory-keeper` is the only agent that edits this file. Workers and planners
hand structured summaries to `memory-keeper` after a managed run completes.

## Conventions

### Cold-start promotion (master = null)

When `research/live/master.json` has `hash=null`, the first managed run
that produces parseable `mean_reward` and `pass_rate` is **unconditionally
promoted** to master. That run writes its commit SHA into `hash`,
`parent_master_hash=null` in the ledger row, and `promote=true`.

### Promotion gate (master != null)

A subsequent run promotes only when it satisfies ALL of:

1. `mean_reward` strictly greater than master.
2. `pass_rate` not regressed (>=) versus master.
3. `eval_split`, `seed_count`, `max_turns`, `reward_version`, and
   `eval_backend` all match master.

Ties or `mean_reward` equal-to-master do not promote.

### Backend lock

`eval_backend` is part of the comparability contract. A run measured on
`coreweave` (FastAPI on A100) cannot be promoted against a master measured
on `local` (in-process eval), and vice versa. Switching backend re-opens
cold-start.

### Verification-phase runs

A run is **verification-phase** when its goal is to test a binary
hypothesis (does X unlock signal? does Y produce gradient?) rather than
to produce a master-promotion candidate. Verification runs are marked
explicitly in three places:

1. Filename: `research/experiments/EXP-NNN-verification-<context>.md`
2. Header banner in the spec: "⚠️ VERIFICATION PHASE — NOT
   MASTER-PROMOTION-ELIGIBLE."
3. `results.tsv` row: `experiment_id` suffix `-verify`, `promote=false`,
   `comment` begins with `VERIFICATION ONLY — ...`.

Verification runs **never** write `research/live/master.json`, regardless
of metric values. They may run on non-master-comparable backends (e.g.
`local` H200 eval while master is locked to `coreweave` A100). A passing
verification result triggers a follow-up production run (same config,
correct backend) which IS master-promotion-eligible.

This convention exists because the A100 eval queue at NU Explorer is
unreliable, and we don't want infrastructure availability to block
testing scientific hypotheses about the RL pipeline.

## Format Per Entry

```
## <YYYY-MM-DD> — <experiment-id> — <one-line title>

- hypothesis: <one sentence>
- parent master hash: <hash>
- variable changed: <file> :: <symbol> :: <before> → <after>
- runner / job id: <modal | northflank> / <job-id>
- metrics: mean_reward=<x> pass_rate=<x> speedup_vs_orig=<x> fast_p=<x>
- decision: <promote | no-promote> — <one-line reason>
- interpretation: <one short paragraph>
```

## Entries

_(empty — first run will land below)_
