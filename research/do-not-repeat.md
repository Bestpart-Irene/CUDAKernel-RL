# Do Not Repeat

Approaches that have been tried and ruled out. Keeps `planner` and
`researcher` from re-proposing them.

`memory-keeper` is the only agent that edits this file. Add an entry when a
managed run fails or regresses in a way that is not just noise.

## Format Per Entry

```
## <YYYY-MM-DD> — <one-line title>

- what was tried: <one sentence>
- why it failed: <one short paragraph>
- evidence: <experiment-id(s) in results.tsv>
- conditions under which it could be revisited (if any): <one line or none>
```

## Entries

## 2026-05-16 — Stage 1 GRPO directly on non-SFT'd base Qwen3-Coder-30B-A3B

- what was tried: Cold-start Stage 1 GRPO warm-up (G=2, 100 steps,
  max_completion_length=1024, max_turns=3, v1-discrete-milestone reward)
  on base `Qwen/Qwen3-Coder-30B-A3B-Instruct` with no prior SFT.
- why it failed: Base model emits zero parseable ```cuda blocks in 1024
  tokens. All rollouts hit clipped_ratio=1.0 with mean_terminated_length=0,
  compile_failed → reward=-1 constant, reward_std=0, frac_reward_zero_std=1.0,
  advantage=0, grad_norm=0. GRPO is mathematically correct but has no
  signal. Entropy collapsed to 0.15 within 13 steps. wandb run qll2wvnl,
  slurm 6858988, scancel'd at step 13 / ~5h wall-clock.
- evidence: EXP-001 (no results.tsv row — cancelled before completion).
- conditions under which it could be revisited: only after an SFT pass on
  `datasets/doublegraph_sft.jsonl` (or equivalent CUDA-kernel format
  corpus) lifts p(valid_cuda_block) above ~0.05 per rollout. Until then,
  do not propose Stage 1 GRPO from a raw base checkpoint regardless of
  LR / G / temperature / max_completion_length sweeps — none of those
  knobs fix a zero-signal reward distribution.
