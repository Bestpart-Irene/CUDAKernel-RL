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

## 2026-05-18 — INVALIDATED PRIOR: every Stage 1/2/3 run before this date ran on a contract-mismatched SFT corpus

- what was discovered: The 4 WCC entries (indices 76-79) in
  `datasets/doublegraph_sft.jsonl` are upstream doubleGraph production code
  that lives in `namespace aai`, depends on `cugraph/aai/algorithms.hpp`, and
  **never exposes `extern "C" void wcc_kernel(...)`** — the exact C symbol
  the verifier dlsym's at `verification/pac_verify.py:192`. The full corpus
  contains zero occurrences of the literal string `wcc_kernel`. Stage 2 SFT
  trained the model on a prior that is fundamentally incompatible with the
  Stage 1/3 verifier contract.
- why this invalidates prior records:
  - All Stage 2 checkpoints produced before this date were SFT'd on the
    broken corpus. Affected jobs on NU Explorer: 6868418, 6868569, 6871736,
    6886193, 6887326.
  - All Stage 1/3 GRPO runs warm-started from those checkpoints
    (`KERNELFORGE_STAGE1_INIT_CKPT=outputs/kernelforge-stage2/checkpoint-*`)
    inherit the same prior. This includes every results.tsv row up to and
    including EXP-009-D (job 6888916, cancelled mid-flight after diagnosis).
  - EXP-009-B (job 6888764, first non-(-1) signal) still made a real
    finding — `MAX_COMPLETION_LENGTH=1024` truncates kernels — but its
    `reward = -0.25` and the 6/7 compile=True / 0/7 correct=True ratio are
    **NOT comparable** to any future row, because every compile=True
    rollout failed with `undefined symbol: wcc_kernel`. The compile rate
    measured the model's prior, not its kernel-writing skill.
- evidence:
  - results.tsv rows through 2026-05-18 (no post-fix row exists yet).
  - kf_stage1_6888876.out: 4/4 `[VERIFIER_INLINE]` lines show
    `verifier_msg='Kernel FFI verification failed... undefined symbol: wcc_kernel'`.
  - `grep wcc_kernel datasets/doublegraph_sft.jsonl | wc -l` returns 0.
- conditions under which prior numbers could be cited again: never. They
  measured a different system. They stay in the ledger as historical
  diagnostics, not as performance baselines.
- conditions for future comparability:
  - `scripts/build_wcc_sft_replacements.py` (in tree) rewrites the 4 WCC
    rows with self-contained kernels exposing the canonical contract.
  - After running it, re-train Stage 2 SFT on the patched corpus. The
    resulting checkpoint is the new comparability anchor.
  - All future comparability claims must cite a Stage 2 ckpt produced
    after this date AND `KERNELFORGE_STAGE1_MAX_COMPLETION_LENGTH >= 2048`.

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

## 2026-05-16 — B: combined beta + max_prompt_length + max_completion_length tweak (planner-rejected, never run)

- what was tried: planner originally proposed a single experiment that
  changed `beta`, `max_prompt_length`, and `max_completion_length`
  simultaneously on top of the EXP-001 baseline.
- why it failed: violates the one-variable rule — three coupled knobs
  in one run produces no actionable signal. Must be split into three
  separate single-variable experiments **after vLLM colocate lands in
  master**, because rollout-backend differences dominate the wall-clock
  and reward picture today.
- evidence: pre-dispatch planning notes for EXP-002; no results.tsv row.
- conditions under which it could be revisited: only after EXP-002
  (vLLM colocate) is in master, and only as three split experiments
  with EXP-002's master as parent.
- [note 2026-08-10: the "EXP-002 (vLLM colocate)" referenced in this
  entry was a draft that never ran; the EXP-002 identifier was later
  consumed by the Stage-2 SFT run.]
- provenance: (recovered 2026-08-10 from a stale worktree draft; see
  archive/worktree-salvage-20260516/)

## 2026-05-16 — C: async H200↔A100 pipelining via AsyncGRPOTrainer (planner-rejected, never run)

- what was tried: planner proposed running rollout async on H200 while
  the optimizer step proceeds on A100 (or vice versa) to overlap the
  long rollout latency with policy update compute.
- why it failed: AsyncGRPOTrainer's "Keep Routing" stabilizer is not
  implemented for MoE actors (HF async-rl-landscape blog). Running
  async on Qwen3-Coder-30B-A3B without that stabilizer is known to
  destabilize MoE expert routing during training and is not a safe
  next step. Also: while we are still rollout-bound on
  H200-`.generate()`, the async win is mostly addressed more cheaply
  by switching to vLLM colocate first.
- evidence: pre-dispatch planning notes for EXP-002; no results.tsv row.
- conditions under which it could be revisited: defer until vLLM
  colocate is in master AND either (a) HF ships MoE-aware "Keep
  Routing" in AsyncGRPOTrainer, or (b) we move to a dense actor.
- [note 2026-08-10: the "EXP-002 (vLLM colocate)" referenced in this
  entry was a draft that never ran; the EXP-002 identifier was later
  consumed by the Stage-2 SFT run.]
- provenance: (recovered 2026-08-10 from a stale worktree draft; see
  archive/worktree-salvage-20260516/)

## 2026-05-16 — D: eval server batching (planner-rejected, never run)

- what was tried: planner proposed batching `/evaluate` requests on
  the eval server to reduce per-completion HTTP overhead during
  multi-turn rollouts.
- why it failed: requires editing `eval_service/` (touches
  `eval_service/eval_core.py` and `eval_service/app.py`); the
  worktree workflow is single-file, and the proposal does not yet
  cleanly fit the one-file change rule because batching changes both
  the request schema and the timing path. Needs separate scoping
  before it is dispatchable.
- evidence: pre-dispatch planning notes for EXP-002; no results.tsv row.
- conditions under which it could be revisited: scope it as a
  dedicated eval-service refactor experiment with its own design
  doc; identify the single batching parameter to flip; ensure
  comparability-key fields (eval backend, eval split) are unchanged.
- [note 2026-08-10: the "EXP-002" referenced in this entry was the
  draft vLLM-colocate experiment, which never ran; the EXP-002
  identifier was later consumed by the Stage-2 SFT run.]
- provenance: (recovered 2026-08-10 from a stale worktree draft; see
  archive/worktree-salvage-20260516/)

## 2026-05-16 — E: Unsloth fast path retry on Qwen3 MoE (planner-rejected, never run)

- what was tried: planner suggested re-enabling Unsloth's
  FastLanguageModel + PatchFastRL("GRPO") fast path on
  Qwen3-Coder-30B-A3B-Instruct.
- why it failed: upstream broken — Unsloth GH issues #3807 and #3422
  document the Qwen3 MoE fast-path regression. Multiple users have
  reported the same failure mode; Datta0 has not yet shipped a fix.
  Retrying it now just consumes a planner slot for a known-bad
  outcome and burns Explorer GPU-hours we don't have.
- evidence: Unsloth GH #3807, #3422; the very reason EXP-001 falls
  back to vanilla Transformers + PEFT.
- conditions under which it could be revisited: only after Datta0
  ships a Qwen3 MoE fix in mainline Unsloth. When that happens, pin
  the exact Unsloth version that contains the fix in `pyproject.toml`
  / `requirements*.txt`, and only then propose a re-enable experiment.
- provenance: (recovered 2026-08-10 from a stale worktree draft; see
  archive/worktree-salvage-20260516/)

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

## 2026-05-18 — Stage 1 / Stage 3 with `KERNELFORGE_STAGE1_MAX_COMPLETION_LENGTH=1024`

- what was tried: every Stage 1 GRPO run from EXP-001 through EXP-008
  used `max_completion_length=1024` as the default (or implicit) limit.
- why it failed: kernels are systematically truncated mid-source. The
  doubleGraph-SFT'd model emits WCC kernel bodies that need ~1100-1200
  tokens; ops6k-style kernels run longer still. EXP-009-B isolated the
  causal variable: flipping 1024 -> 2048 (no other change vs EXP-008-E)
  produced the **first non-(-1) reward in project history** — compile
  rate jumped 0/8 -> 6/7 (~86%), reward_std 0 -> 0.5, grad_norm 0 ->
  0.046. All of EXP-001..EXP-008's reward=-1 collapse retroactively
  attributes here in large part: well-formed code was being cut before
  the closing brace and failing compile, not reward shape or GRPO knob.
- evidence: EXP-009-B (slurm 6888764) at parent master 117bb5d.
- conceptual family ruled out: **any Stage 1 / Stage 3 GRPO run with
  `KERNELFORGE_STAGE1_MAX_COMPLETION_LENGTH <= 1024`**, regardless of
  reward version, G, max_turns, or loss_type. The completion budget
  must be large enough to fit the SFT-distribution-of-lengths tail.
- conditions under which it could be revisited: only if the SFT corpus
  is rebuilt with a hard `<=800` token filter on assistant bodies AND
  the new SFT'd model is verified to EOS naturally under 1024. Until
  then, default to >=2048.

## 2026-05-28 — Sakana SFT + ops6k torch-extension eval path is a reward-hack channel via torch:: C++ API

- what was tried: EXP-015'-A (job 7040100) Stage 1 GRPO on ops6k tasks with
  Sakana 400 SFT prior, `eval_backend=local` (subprocess-isolated), reward
  v3-symbol-shaped. Produced 5 unique `correct=True` rollouts out of 80
  (6.25% pass_rate, originally double-counted as 10/152) — claimed as the
  first correct=True signal in project history and unconditionally
  promoted to master under the cold-start rule.
- why it was reward-hacked: the ops6k evaluation path
  ([eval_service/eval_core.py:588](eval_service/eval_core.py#L588)
  `evaluate_ops6k_kernel_impl`) does **NOT** call
  `scan_forbidden_symbols` after compile — only the WCC path at L356 does.
  Combined with:
  - **SFT bias**: every one of the 400 Sakana SFT rows
    (`datasets/sakana_sft.jsonl`) demonstrates `#include <torch/extension.h>`
    + `torch::Tensor run_kernel(torch::Tensor)` style. 400/400 grep hit on
    `torch::|at::Tensor|c10::|torch.nn`.
  - **Loose tolerance**: `_assert_close` ([eval_core.py:161](eval_service/eval_core.py#L161))
    uses `rtol=1e-3, atol=1e-3` — 1000x looser than torch default.
  - **anti_hack runtime checks are blind to this attack**: the candidate
    kernel can `return torch::relu(x)` or `return x.softmax(-1)` directly
    inside the .cu source; output is non-constant (real op), not
    passthrough (output differs from input element-wise), not no-op
    (>1μs runtime due to pybind + libtorch overhead), shapes match —
    every check passes vacuously.
  - **Smoking gun in the log**: 4 of 5 correct=True rollouts had
    `sv_eager < 1.0` (0.17, 0.77, 0.04, 0.77) — meaning the candidate
    was 6-23x SLOWER than eager PyTorch. A real hand-written CUDA
    kernel on simple ops shouldn't be 23x slower than eager. This is
    exactly the overhead profile of pybind + libtorch op dispatch
    wrapped in a custom extension.
- evidence:
  - EXP-015'-A (slurm 7040100) `logs/kf_stage1_7040100.out`
  - `grep scan_forbidden_symbols eval_service/eval_core.py` shows only
    one call site, at L356 inside `evaluate_kernel_impl` (the WCC
    path); zero references inside `evaluate_ops6k_kernel_impl`.
  - `grep -coE 'torch::|at::Tensor|c10::|torch.nn' datasets/sakana_sft.jsonl`
    returns 400 of 400 rows.
- conceptual family ruled out: **any ops6k eval path that (a) builds
  the candidate via `torch.utils.cpp_extension.load(..., with_cuda=True)`
  so libtorch is linked, AND (b) lacks a mechanism to prevent the
  candidate from calling libtorch ops at runtime**. Naive
  `scan_forbidden_symbols` on the .so is NOT a sufficient mechanism —
  every torch-extension .so links libtorch, so `nm -D` is guaranteed to
  show `torch::`/`at::`/`c10::` symbols regardless of whether the
  candidate is legitimate. The check would false-positive every kernel,
  which is why the ops6k path lacked the call in the first place — and
  why simply wiring it in does NOT fix the hole.
- conditions under which it could be revisited: the channel requires a
  structural anti-hack. Two known options:
  1. **C++ Dispatch interception**: inside the eval subprocess, register
     a high-priority `TORCH_LIBRARY_IMPL` fallback that captures all
     `aten::*` calls and throws unless whitelisted (only Tensor
     construction/accessors allowed). Then `dlopen` the candidate.
  2. **Raw `extern "C"` contract**: redesign ops6k to match WCC — raw
     pointers, no libtorch link, eval harness handles `cudaMalloc`/
     `Memcpy`. Forces the candidate to write actual CUDA. Cost: full
     SFT corpus rebuild (Sakana 400 rows all use torch::).
  Cheap stopgap (does NOT fully close): source-level regex scan for
  obvious calls (`torch::relu(`, `.softmax(`, `at::*` op names) — model
  can bypass via aliases / vendored algorithms but it forces hack
  attempts to be less trivial.
- master action taken: `research/live/master.json` rolled back to
  `hash=null` on 2026-05-28; `results.tsv` row renamed to
  `EXP-015'-A-INVALIDATED` with `promote=false`; cold-start gate
  re-opened.

## 2026-05-28 — Project history has NEVER produced a multi-step monotone mean_reward training curve

- what is observable from EXP-001 through EXP-015'-A: every recorded
  Stage 1 GRPO run on this codebase has either
  (a) collapsed to flat `reward=-1` with `reward_std=0, grad_norm=0`
  for the full run (EXP-001..EXP-008 family), or
  (b) produced a single non-(-1) data point at step 1 (EXP-009-B,
  one 22-min run, never extended), or
  (c) produced "correct=True" signal that was subsequently shown to be
  reward-hacked (EXP-015'-A, see 2026-05-28 entry above).
- what has NEVER been observed: a multi-step training curve where
  `mean_reward` increases monotonically (or even trends up across
  ≥ 3 steps) under a non-hacked reward path. The full project budget
  to date has gone into infrastructure surgery and reward-bypass
  removal. Whether the actual RL signal — once anti-hack is structural
  and SFT is contract-aligned — is dense enough to move the policy
  remains an open empirical question that has not been tested even
  once.
- generalizable rule: **before committing budget to ablations or
  scaling, run a cheap verification spike that demonstrates the
  pipeline produces a non-flat positive-trajectory training curve
  on a small task pool under the patched evaluator**. If that spike
  fails, none of the downstream investment (vLLM optimization,
  multi-GPU DDP, SFT corpus rebuild, Phase 2 Ablation 1/2/3) is
  justified. See `research/experiments/EXP-016p-spike.md` for the
  spike spec.
- conditions under which broader investment could be authorized: spike
  produces a training curve where `mean_reward` at the last 10 steps
  is strictly greater than the first 10 steps, with confidence
  surviving a per-step bootstrap. Until then, treat all multi-month
  / multi-thousand-dollar plans as conditional.

## 2026-05-27 — Treating a 7% correct rate (Wilson lower bound 3.6%) as a stable steady-state signal

- what NOT to do: cite EXP-015'-A's `pass_rate=0.066` (10/152 correct,
  Wilson 95% CI `[3.6%, 11.7%]`) as evidence of a *steady-state* policy
  capability, or use it as the comparability baseline for downstream
  knob ablations.
- why: this is a **phase-transition signature**, not a converged rate.
  Three independent fixes (commits 3b28c36, 61af7f5, 8989ac4) landed
  simultaneously over a 20-step run from a fresh SFT prior; the
  cold-start promotion was unconditional per the convention banner, not
  a quality assertion. The lower bound 3.6% is barely above zero, and
  the 7% point estimate sits on top of a high-variance Bernoulli over
  152 rollouts. Treating it as steady-state risks (a) premature
  declarations of "we cleared cold-start" leading to misallocated
  ablation budget, and (b) noise-driven false negatives on EXP-016+
  downstream tests because the parent baseline itself is sampling-noisy.
- evidence: EXP-015'-A (slurm 7040100). Bucket distribution heavily
  skewed (63% compile_fail, 30% compile_OK_wrong) — the policy is still
  failing at the compile gate two-thirds of the time. A 7% correctness
  rate observed during a transient unblock is consistent with a true
  underlying rate anywhere in `[3.6%, 11.7%]`, not "the rate".
- conditions under which the rate could be cited as a real baseline:
  EXP-016 (24-step extension) must show (a) the slope continuing
  upward (`correct/rollouts` increasing in the last 8 steps relative
  to the first 8), AND (b) Wilson CI tightening with more samples. Only
  after a second run reproduces correct>=7% across a non-overlapping
  rollout sample does the rate become a stable comparability anchor.

## 2026-05-18 — Probes that rely on `[ROLLOUT]` debug from `multi_turn_rollout.py`

- what was tried: every prior diagnostic that depended on per-rollout
  prints inside `training/multi_turn_rollout.py` (the `rollout_func`
  body): EXP-D1 (slurm 6885765), EXP-008 D (slurm 6886283), EXP-008
  D-retry (slurm 6886408).
- why they failed: **TRL 0.29 GRPOTrainer does not invoke `rollout_func`
  during training.** The construction tag `[ROLLOUT_FACTORY]` fires at
  init, but `[ROLLOUT_CALL]` never fires — confirmed in EXP-009-B
  (slurm 6888764) stdout AND stderr. TRL emits warning at init:
  `'rollout_func' is an experimental feature. This API may change or
  be removed at any time without prior notice.` The interpretation that
  env vars "failed to propagate" was partially correct (some did) and
  partially misdiagnosed: the print sites themselves were in dead code.
  Every Stage 1 / Stage 3 run in this codebase has actually been
  **single-turn** regardless of `KERNELFORGE_STAGE1_MAX_TURNS`.
- evidence: EXP-D1 (6885765), EXP-008-D (6886283), EXP-008-D-retry
  (6886408), EXP-009-B (6888764) — all show zero `[ROLLOUT_CALL]` or
  `[ROLLOUT prompt=...]` or `[VERIFIER prompt=...]` lines.
- conceptual family ruled out: **any probe that gates a diagnostic
  print on `KERNELFORGE_ROLLOUT_DEBUG` inside `training/multi_turn_rollout.py`,
  or on `KERNELFORGE_VERIFIER_DEBUG` inside the dead multi-turn path**.
  These tags will never fire under TRL 0.29.
- mitigation: commit 469f6a4 relocates the env-gated prints to the
  active `reward_from_env` path. New tags are `[ROLLOUT_INLINE]` and
  `[VERIFIER_INLINE]`. Future probes MUST use the `_INLINE` tags.
- conditions under which it could be revisited: if TRL bumps to a
  version that re-invokes user-provided `rollout_func`, or if we swap
  to a custom trainer subclass that explicitly drives multi-turn. Note
  also: the B5 "multi-turn attribution bug — concat prompt_ids across
  turns" hypothesis cannot bite under TRL 0.29 because the multi-turn
  loop never runs; do not re-propose B5 against this trainer.

## 2026-08-11 — Silent task-pool fallback: Ops-6K load failure downgrades Stage 1 to 3 WCC prompts

- what was tried: exp018c v1/v2 (slurm 7359525, 7365795) — Stage-1 GRPO
  launches intended to run on the ops6k task pool.
- why it failed: the launcher caught the Ops-6K load exception and
  substituted a different task pool without failing the job — stdout shows
  `Could not load Ops-6K for Stage 1: ...` immediately followed by
  `Using fallback Stage 1 prompts with live WCC evaluation support`, after
  which training proceeded on 3 WCC prompts. Two runs measured the wrong
  thing, and the substitution was only visible in stdout (v1's load error:
  `invalid literal for int() with base 10: 'trivial'`; v2's: `cannot mix
  list and non-list, non-null values`).
- evidence:
  `research/audits/evidence-june-018c/kf_exp018c_7359525.out` and
  `research/audits/evidence-june-018c/kf_exp018c_7365795.out`
  (results.tsv rows EXP-018c-v1-verify, EXP-018c-v2-verify).
- conceptual family ruled out: **any launcher path that continues after
  the intended task pool fails to load**.
- mitigation shipped: 2026-08-11 hard-fail in the Stage-1 loader unless
  `KERNELFORGE_ALLOW_POOL_FALLBACK=1` (a coordinator is landing this
  concurrently with this entry).
- conditions under which it could be revisited: none — hard-fails are
  strictly better here.
