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

## 2026-05-17 — doubleGraph WCC SFT for ops6k Tensor-return tasks

- what was tried: warm-start Stage 1 GRPO on the mixed Stage 1 task pool
  (15 ops6k + 4 wcc) from the doubleGraph-SFT'd ckpt-50.
- why it failed: structural mismatch between SFT contract and eval
  contract. doubleGraph teaches `void wcc_kernel(row_ptr, col_idx, n, labels)`
  (output-as-param). ops6k tasks call `extension.run_kernel(*inputs)` and
  expect a returned Tensor. EXP-004 v1 debug confirmed the SFT'd model
  emits void-output-arg signatures on ops6k prompts and gets rejected at
  pybind invocation with "incompatible function arguments".
- evidence: EXP-003-v1 (slurm 6869808), EXP-004 v1 debug (slurm 6870504).
- conceptual family ruled out: **any SFT corpus whose assistant
  signatures structurally conflict with the target eval harness's pybind
  contract**, regardless of prompt engineering on top. Recognized
  symptoms: model emits compile-passes-but-wrong-pybind-signature
  kernels at high rate; single-rollout shows `compiles=True, correct=False`
  with `error` starting `Correctness check failed: run_kernel():
  incompatible function arguments`.
- conditions under which it could be revisited: pair doubleGraph SFT
  *only* with WCC-backend Stage 1 tasks (EXP-003-v3 D6 pattern) — both
  sides of the contract aligned. For ops6k Stage 1 tasks, switch SFT
  source to one that already emits Tensor-return signatures.

## 2026-05-17 — Sakana AI-CUDA-Engineer-Archive SFT without token-length filter

- what was tried: warm-start Stage 1 GRPO from a Stage 2 SFT trained on
  200 Sakana rows filtered only by `Correct=True, Speedup>0.5,
  Max_Diff<0.01`.
- why it failed: Sakana kernels average ~4000 chars (~1200 tokens). The
  SFT'd model reproduces that length under sampling and Stage 1's
  `max_completion_length=1024` cuts every generation mid-statement.
  EXP-003-v2 wandb showed `clipped_ratio=1.0`, `mean_terminated_length=0`
  for every step.
- evidence: EXP-003-v2 (slurm 6872320), EXP-002b Stage 2 SFT (slurm 6871736).
- conceptual family ruled out: **any SFT corpus whose body-length
  distribution exceeds the GRPO `max_completion_length`**, regardless of
  reward shape or G. The model can only emit what it was trained to.
- conditions under which it could be revisited: re-build Sakana SFT with
  `len(CUDA_Code) <= 2500` chars (~700 tokens) filter — D4 pipeline
  already in `scripts/build_sakana_sft.py`. OR bump
  `KERNELFORGE_STAGE1_MAX_COMPLETION_LENGTH=2048+` and accept the 2-3x
  per-step slowdown.

## 2026-05-17 — Single-rollout debug as cheap-kill before GRPO

- what was learned: 5 sequential Stage 1 GRPO runs (EXP-003 v1/v2/v3,
  EXP-005, EXP-006/v4, EXP-007/v6 — total ~30 H200 hours) all hit
  `reward=-1, std=0, grad_norm=0`. Initial diagnoses pointed at
  exploration / data scale / reward shape. EXP-004 v1-v3 single-rollout
  probes (~30 H200 minutes total) then exposed 4 cascading infrastructure
  bugs that explained most of the -1 signal:
  - slurm did not `module load cuda/12.8.0` → eval_core nvcc subprocess
    raised FileNotFoundError on every rollout
  - `_local_compile_check` swallowed FileNotFoundError silently and
    returned "compile succeeded" — masking the eval failure upstream
  - `extract_cuda_code` regex left the opening ``` fence when closing
    fence was truncated by max_new_tokens
  - sakana entrypoint pybind name was `forward`, not `kernel_function`
- generalizable rule: **before pinning a `reward=-1` finding on RL or SFT
  science, dispatch a single-rollout debug that prints
  extract/compile/eval verdicts**. ~30 min of GPU time saves multi-hour
  GRPO runs that were doomed by silent infra failures. The debug script
  template is `scripts/debug_eval_pipeline.py` + `scripts/cluster/debug_eval.slurm`.
- evidence: EXP-001..EXP-007 chronology; commits 5b5db65, 75e34f1,
  e115cf6, ed628a9 ship the four infra fixes.
- conditions under which it could be relaxed: never. The single-rollout
  probe must stay in the workflow for any future "binary -1 collapse"
  diagnosis.

## 2026-05-17 — Multi-turn feedback append at max_turns=3 self-poisons context budget

- what was tried: Stage 1 GRPO with `max_turns=3` and turn-N feedback
  appended to the prompt for turn N+1.
- why it failed: EXP-D1 (slurm 6885765) wandb shows clipped_ratio
  jumping 0.25 (step 1) → 1.0 (step 2). Feedback text consumes the
  context budget so that later turns hit max_new_tokens before
  reaching the closing ``` fence. By step 2 every completion is
  truncated. Independent of reward shape.
- evidence: EXP-D1 (slurm 6885765).
- conceptual family ruled out: **multi-turn with append-mode feedback at
  high max_turns when prompt+feedback+completion sum exceeds the model's
  effective working budget on this hardware**.
- conditions under which it could be revisited: (a) `max_turns=1` —
  verification context, no feedback loop. (b) replace-mode feedback that
  swaps the previous turn's content rather than appending. (c) bump
  max_completion_length or move to vLLM-backed generation with larger
  KV cache budget.

## 2026-05-17 — Shaped reward + G=2 + beta=0.04 default did NOT recover variance

- what was tried: Stage 1 GRPO with reward_version=v2-shaped
  (compiled_but_wrong → 0.0 instead of -1.0), G=2, beta default 0.04.
  3-way A/B/C against v1 binary reward and v2-shaped+G=4+beta=0.
- why it failed: ALL 3 runs (EXP-005 v5, EXP-006/v4, EXP-007/v6) — total
  ~1200 rollouts over ~24 H200 hours — produced `unique reward values:
  [-1]`. Not a single rollout returned reward != -1, even in the
  v2-shaped runs where compiled_but_wrong should yield 0.0.
- evidence: EXP-005 slurm 6880035, EXP-006 slurm 6880846, EXP-007 slurm
  6881355.
- conceptual family **PROVISIONALLY** ruled out: reward-shape-alone
  fixes for the binary-collapse symptom on this stack. Cannot be a
  definitive ruling until EXP-S4 (single-rollout v2-shaped verification,
  slurm 6885946) returns. Two candidate explanations for why v2-shaped
  showed no effect:
  - Q1: v2-shaped does not actually reach the GRPO reward path in the
    multi_turn_rollout chain (module caching / wrong call site).
  - Q2: the multi_turn_rollout context produces different model outputs
    than the debug script, and 100% of rollouts genuinely fail
    extract/compile before reaching eval (EXP-D1 step 1 already showed
    75% complete rollouts STILL got reward=-1, supporting Q2).
- conditions under which it could be revisited: only after EXP-S4
  resolves Q1 vs Q2. If Q1 (chain broken), fix the chain and retry. If
  Q2 (downstream rejects everything), the right next move is not more
  reward shapes — it is to find why extract/compile/eval rejects 100%
  of well-formed completions.

## 2026-05-17 — `KERNELFORGE_ROLLOUT_DEBUG=1` env var does not propagate through `sbatch --export=ALL,VAR=val`

- what was tried: dispatch Stage 1 GRPO with `sbatch --export=ALL,KERNELFORGE_STAGE1_MAX_STEPS=2,KERNELFORGE_ROLLOUT_DEBUG=1`
  to enable per-rollout stdout prints inside `multi_turn_rollout.py`.
- why it failed: `MAX_STEPS=2` correctly propagated to the slurm job
  (job ran 2 steps), but no `[ROLLOUT prompt=...]` line appeared in the
  job stdout. The python-side `os.getenv("KERNELFORGE_ROLLOUT_DEBUG", "0")`
  never saw "1".
- evidence: EXP-D1 slurm 6885765 stdout (zero `ROLLOUT` matches).
- conditions under which it could be revisited: prefer adding the env
  var as an explicit `export` line inside the slurm script body rather
  than relying on the `--export=ALL,VAR=val` comma-list. Then control
  it via `KERNELFORGE_ROLLOUT_DEBUG=1 sbatch ...` from the calling shell.
