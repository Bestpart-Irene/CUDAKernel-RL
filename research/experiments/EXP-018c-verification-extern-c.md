# EXP-018c — first clean verification run under unified extern C contract

⚠️ VERIFICATION PHASE — NOT MASTER-PROMOTION-ELIGIBLE. Never writes research/live/master.json regardless of metric values.

## History

(Recovered 2026-08-11: Explorer logs at `research/audits/evidence-june-018c/`
show the June "018c" was three attempts, not one.)

- v1 = slurm 7359525, 2026-06-01, VOID. Ops-6K load FAILED (`invalid
  literal for int() with base 10: 'trivial'`) and the launcher silently
  fell back to 3 WCC prompts — never ran the intended ops6k pool. 2/50
  steps completed at ~19 min/step (1149 s/it) before scancel; batch
  rewards in {-1.0, -0.5 (compiled but wcc_kernel symbol missing)}. See
  results.tsv row EXP-018c-v1-verify (corrected 2026-08-11) and notes.md
  2026-06-01 + 2026-08-11 entries.
- v2 = slurm 7365795, 2026-06-01, VOID. Same silent WCC fallback (load
  error: `cannot mix list and non-list, non-null values`); scancel'd by
  the operator at 17:43:51 EDT, ~4 min after start, 0/25 steps completed.
- v3 = slurm 7366090, 2026-06-01→06-02, COMPLETED 25/25 steps in 7h27m
  (train_runtime 26827s) but VOID for interpretation: Ops-6K pool loaded
  (16 prompts, post-51eb7c9 schema fix) yet the pre-migration prompts
  demanded pybind while the extern-C evaluator Source-rejected it —
  completions dominated by `Source rejected: #include <torch/` (the
  contradiction EXP-018d later fixed). Cold start from base
  Qwen3-Coder-30B-A3B-Instruct, eval_backend=local, v3-symbol-shaped,
  G=2, max_completion=2048, config max_turns=3 (TRL-0.29 dead code).
  Reward flat in [-1.0, -0.75] with no trend; terminal reward -1.0 /
  std 0.0 / grad_norm 0.0 (grad_norm=0.0 on 22/25 steps); entropy 0.35
  early → 0.25 terminal (peak 0.45 @step7). wandb exp018c-7366090 /
  4ocfqywn. Retro-recorded 2026-08-11.

## Rerun preconditions (from 2026-08-10 audit)

1. EXP-018c-p0 base pass@k probe decides cold-start vs EXP-018b.
   **SATISFIED 2026-08-12** — probe verdict COLD-START LICENSED
   (pooled p(parse+compile)=0.812 [0.723, 0.878] vs threshold 0.05),
   slurm 9080627, finished 2026-08-12T04:43Z. See results.tsv row
   EXP-018c-p0-verify and notes.md 2026-08-12 entry.
2. 5/5 hacked-fixture replay executed and recorded in EXP-018a.md Run
   section.
3. Launcher rewritten — pass criteria matched to actual step count,
   save_steps + slurm auto-resume, task pool = 3-task spike subset
   (vector_add_e2 / F.elu / F.softplus).
4. Pre-registered results.tsv row with -verify suffix and promote=false
   before dispatch.
5. Any correct=True rollout's .cu source must be persisted and pass a
   post-hoc deep anti-hack scan (script must exist first).
6. (added 2026-08-12, from probe finding a) E2 constant-check fix landed
   with a new evaluator_sha, and an E2 positive control passing on a
   CUDA host — the b182df91c482 evaluator's E2 second-invocation path
   passes broken/identical effective inputs and false-rejected all 14
   compile_ok vector_add_e2 candidates in the probe.
7. (added 2026-08-12, from probe ceiling caveat) planner decision on
   task-pool difficulty rebalance: E1 spike tasks are at ceiling for
   the base model (f_elu/f_softplus pass@8 ~= 1.0), so the current
   3-task pool would measure almost nothing; target tasks with low
   pass@1 but nonzero pass@32, or gate on E2/vector_add improvement
   after the harness fix.

## Run

(to be populated by experiment-worker; rerun not yet dispatched as of 2026-08-10)
