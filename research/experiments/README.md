# Experiments

Per-experiment specs and post-run summaries.

One file per experiment, named `<experiment-id>.md`. `planner` drafts the
spec; `experiment-worker` fills in the run section; `memory-keeper`
finalizes after the result is recorded in `research/results.tsv`.

## Format

```
# <experiment-id> — <title>

## Spec
- campaign: <campaign-id | standalone>
- hypothesis: <one sentence>
- parent master hash: <hash>
- variable changed: <file> :: <symbol> :: <before> → <after>
- runner: <modal | northflank>
- expected upside: <one line>
- duplicate check: <link to results.tsv row or do-not-repeat.md or "none">
- touches reward? <yes | no — if yes link kernelforge-reward-design>

## Run
- job id: <id>
- log path: <path>
- metrics: mean_reward=<x> pass_rate=<x> speedup_vs_orig=<x> fast_p=<x>
- promote: <yes | no>

## Interpretation
<one short paragraph>
```
