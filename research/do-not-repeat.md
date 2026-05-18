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

## 2026-05-17 — ~~Shaped reward + G=2 + beta=0.04 default did NOT recover variance~~  **WITHDRAWN**

**Withdrawn 2026-05-17 — this conclusion was based on a bypassed reward
function.** EXP-005/006/007 produced uniform reward=-1 not because v2-shaped
failed, but because `training/multi_turn_rollout.py:228/240/243/260` (and
`openenv_env/kernel_forge_env.py:162/167`) hardcoded `reward = -1.0` in 4+2
branches without ever calling `compute_reward()`. The v1/v2 A/B was
effectively `if False: ...` — neither code path actually ran reward.py.
See "2026-05-17 — 9 hardcoded reward=-1.0 bypasses found and removed" below.
v1 vs v2-shaped MUST be re-tested with the bypass fix in place; do NOT
assume v2 is "ruled out".

## 2026-05-17 — TRL 0.29 does NOT support `loss_type="gspo"` (we assumed it did)

- what was tried: dispatch Stage 1 GRPO with
  `KERNELFORGE_GRPO_LOSS_TYPE=gspo` to test the Qwen team's recommended
  sequence-level loss for Qwen3 MoE (Qwen GSPO arXiv:2507.18071,
  researcher contingency #3).
- why it failed: job 6886192 raised `ValueError: Unknown loss type:
  gspo` at TRL trainer init. `GRPOConfig.loss_type` parameter exists in
  TRL 0.29.0 but the *value* "gspo" is not in the allowed enum. TRL 0.29
  loss_type options are limited to `{grpo, dr_grpo, dapo, bnpo}` —
  default is `dapo` (we have been on DAPO loss the entire night).
- evidence: slurm 6886192 stderr — `ValueError: Unknown loss type: gspo`
  at `trl/trainer/grpo_trainer.py` validation.
- conceptual family ruled out (until library catches up): **using TRL
  0.29.x to test GSPO sequence-level loss**. Either bump TRL to a
  version that ships GSPO, or implement a custom subclass overriding
  `_compute_loss`. The library check was the correct cheap-kill — we
  burned 19 H200 minutes before TRL's init validator caught the typo
  in our hypothesis instead of after a 3h run.
- conditions under which it could be revisited: install TRL ≥ a release
  that documents GSPO support, or accept the engineering cost of a
  custom `TRLOOGRPOTrainer` subclass that implements
  `_compute_loss` with the GSPO sequence-level form. The Qwen GSPO
  paper provides the equation; non-trivial but tractable.

## 2026-05-17 — `num_generations=8` requires generation_batch_size divisible by 8

- what was tried: dispatch Stage 1 GRPO with
  `KERNELFORGE_STAGE1_NUM_GENERATIONS=8` (researcher contingency #1, G≥8
  per DAPO / Kevin / Dr.GRPO floor) on top of the default
  `per_device_train_batch_size=1, gradient_accumulation_steps=4` →
  generation_batch_size = 1 × 4 = 4.
- why it failed: job 6886282 raised `ValueError: generation_batch_size
  (4) must be divisible by num_generations (8)` at TRL trainer init,
  after 1m16s including model load. Failed before any training step.
- evidence: slurm 6886282 stderr — TRL config validator.
- conceptual family ruled out: **G=8 without bumping
  `gradient_accumulation_steps` to ≥8**. Numerical constraint, not a
  scientific finding.
- conditions under which it could be revisited: dispatch G=8 with
  `gradient_accumulation_steps=8` (effective batch 8) — doubles peak
  memory load from GRPO buffer sizing, OOM risk on H200 80GB with bf16
  30B-A3B actor at `max_completion_length=2048`. Or step down to G=4
  (which divides 4 cleanly) — that's the F-retry we dispatch next.

## 2026-05-17 — `KERNELFORGE_ROLLOUT_DEBUG=1` env var does not propagate through `sbatch --export=ALL,VAR=val`

- what was tried: dispatch Stage 1 GRPO with `sbatch --export=ALL,KERNELFORGE_STAGE1_MAX_STEPS=2,KERNELFORGE_ROLLOUT_DEBUG=1`
  to enable per-rollout stdout prints inside `multi_turn_rollout.py`.
- why it failed: `MAX_STEPS=2` correctly propagated to the slurm job
  (job ran 2 steps), but no `[ROLLOUT prompt=...]` line appeared in the
  job stdout. The python-side `os.getenv("KERNELFORGE_ROLLOUT_DEBUG", "0")`
  never saw "1". The "explicit export inside slurm" fix (commit 2efe179)
  did NOT resolve this — D-retry (slurm 6886408) with explicit
  `export KERNELFORGE_VERIFIER_DEBUG="${KERNELFORGE_VERIFIER_DEBUG:-0}"`
  in the slurm body still saw `os.getenv()` return "0" inside Python.
- evidence: EXP-D1 slurm 6885765 + EXP-008 D slurm 6886283 + EXP-008
  D-retry slurm 6886408 all produced zero `[ROLLOUT` or `[VERIFIER`
  lines in stdout despite different propagation paths.
- conditions under which it could be revisited: needs root-cause
  investigation (probably interactive `srun --jobid=<live job>` and
  `env | grep KERNELFORGE` inside the running job, then trace why
  bash-level export isn't visible to the spawned python process). DO
  NOT propose another diagnostic that depends on this env var until
  the propagation is fixed.

## 2026-05-17 — ~~GRPO param space is NOT the bottleneck for binary -1 collapse~~  **WITHDRAWN**

**Withdrawn 2026-05-17 — the ablations (E max_turns=1, F-retry G=4,
C-retry loss_type=grpo) all observed `reward=-1, std=0, grad_norm=0` —
but the rollout code never called `reward.py`, so the experiments tested
whether different GRPO knobs change the output of `reward = -1.0` (a
hardcoded constant). Trivially no.** The "single-variable GRPO knob tuning
cannot break binary -1 collapse" conclusion does not survive removal of
the bypass. The observation that `clipped_ratio` reached 0 at max_turns=1
(token-level, independent of reward) is still valid. EVERYTHING about
reward distribution in EXP-008 must be re-tested after the bypass fix.

## 2026-05-17 — 9 hardcoded reward=-1.0 bypasses found and removed

- what was discovered: 9 sites across 3 files (`training/multi_turn_rollout.py`
  L228/240/243/260, `openenv_env/kernel_forge_env.py` L162/167,
  `skydiscover_integration/evaluator.py` L84/89/160) hardcoded
  `reward = -1.0` (or `combined_score = -1.0`) directly, bypassing
  `compute_reward()` in `openenv_env/reward.py`. Only the "compile+correct+remote
  eval succeeded" branch ever called the canonical reward function.
- why it broke everything: `KERNELFORGE_REWARD_VERSION=v2-shaped` was a
  no-op for the `compile_failed` and `compiled_but_wrong` buckets — the
  exact buckets where v2 was supposed to differ from v1. Every prior
  reward-shape A/B (EXP-005/006/007) effectively compared two identical
  hardcoded paths. ~1500 rollouts of "v2 didn't help" are uninformative.
- evidence: pytest 115/115 pass after refactor routes all branches
  through `training.task_support.compute_task_reward()`; manual
  fixture replay confirms `KERNELFORGE_REWARD_VERSION` toggle now flips
  `compiled_but_wrong` between -1.0 (v1) and 0.0 (v2-shaped) as designed.
- generalizable rule: **before declaring any reward-related conclusion,
  verify that `compute_reward()` is actually called on the rollout path
  for every reward bucket.** Unit tests on `reward.py` alone are not
  sufficient — they exercise a function the rollout may not invoke.
  Validate via fixture replay through the rollout entry point (see
  researcher's eval-first methodology, InstructGPT/Constitutional-AI/
  DeepSeek-R1 standard practice).
- conditions under which it could be revisited: never. The fix is in
  place and tested. Future reward changes must include a fixture-replay
  test through the rollout call graph.
