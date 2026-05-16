# Campaigns

A campaign is a thematic group of single-change experiments aiming at the
same metric pressure (e.g. "raise `pass_rate` on memory-bound ops",
"reduce reward-hack rate on reduction kernels").

`memory-keeper` writes campaign files. `planner` dispatches from them.

## Format

```
# <campaign-id> — <title>

## Goal
<one paragraph>

## Hypotheses queued
- <experiment-id> — <one-line hypothesis> — <status>

## Hypotheses ruled out (link to do-not-repeat.md entries)
- <link>

## Current leader within campaign
- <experiment-id> — mean_reward=<x> pass_rate=<x> speedup=<x>
```
