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

## 2026-05-16 — TECH DEBT (REVERTED): ops6k contract worked-example hardcoded in prompt (EXP-003-verify)

- what was tried: EXP-003 verification kept producing `reward=-1` with
  `reward_std=0` despite SFT from EXP-002 ckpt-50 clearly changing model
  behavior (entropy 0.15→0.35, EOS rate 0→50%). Single-rollout debug
  (`scripts/debug_eval_pipeline.py`, slurm 6870504) showed the model
  produces extractable CUDA that compiles, but writes
  `void run_kernel(torch::Tensor& output, ...)` (Modal-output-arg style)
  instead of the ops6k-required `torch::Tensor run_kernel(...)`
  return-value style. SFT on doubleGraph (WCC-style void signatures)
  biased the model away from the ops6k contract even though the contract
  was already in the prompt as a one-line description.
- what was tried as the F1 fix: added a worked code template inside the
  ops6k branch of `training/task_support.py::task_interface_contract`,
  explicit "do NOT use void run_kernel(output, ...)" rule, and a 28-line
  cpp `torch::Tensor run_kernel(...)` example.
- why F1 failed: second debug run (slurm 6870925) showed the model STILL
  emitted `void run_kernel(arg0, arg1, arg2) -> None` (3-arg void style).
  192 doubleGraph SFT examples in WCC void contract overrode the
  in-context prompt instructions completely. In LLM training, SFT bias
  beats in-context instruction when they structurally conflict.
- decision: **REVERTED on 2026-05-16** (same day) — the F1 prompt
  hardcode is removed from `training/task_support.py`. The structural
  fix is instead: replace the doubleGraph SFT dataset with one whose
  assistant messages use the ops6k `torch::Tensor run_kernel(...)`
  contract, then re-run Stage 2 SFT. Candidate source: SakanaAI/AI-CUDA-Engineer-Archive
  (30K AI-generated kernels in the right contract, filterable on
  `Correct=True`).
- evidence: EXP-003 debug logs `kf_debug_eval_6870504.out` and
  `kf_debug_eval_6870925.out` on Explorer.
- conditions under which it could be revisited: do NOT re-add the F1
  prompt hardcode unless a future experiment specifically demonstrates
  that prompt-level instructions can override SFT bias for structural
  output constraints on this model class. Even then, prefer fixing the
  SFT data over inflating the prompt.
