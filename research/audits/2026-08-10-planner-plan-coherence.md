# 2026-08-10 planner audit — plan coherence

Read-only audit at commit fc6c8f1. Scope: does the CAMPAIGN-018 plan and its
execution order follow from the evidence; where does the plan contradict itself.

---

## 1. DEPENDENCY VIOLATION — yes, direct and double

- The campaign makes 018b a hard blocker: `research/campaigns/CAMPAIGN-018-unify-contract.md:132` — "status: blocked on EXP-018a + EXP-018b."
- The launcher skips it: `scripts/cluster/exp018c.slurm:66-70` — `KERNELFORGE_STAGE1_INIT_CKPT=""` with the comment "Pure cold start from base Qwen3-Coder-30B-A3B" (rationale in commit 97a633d).
- The ledger rules this out: `research/do-not-repeat.md:62-79` (2026-05-16 / EXP-001) — "do not propose Stage 1 GRPO from a raw base checkpoint regardless of LR / G / temperature / max_completion_length sweeps," revisit condition "only after an SFT pass ... lifts p(valid_cuda_block) above ~0.05 per rollout."

The launcher's rationale (both existing SFT ckpts teach now-rejected contracts — true, per do-not-repeat.md:117-138 and 346-407) explains why 018b was inconvenient, not why cold start is legal. Skipping the blocker re-enters the exact ruled-out family: sparse-binary reward + raw base = zero reward variance = zero gradient. The one honest caveat: EXP-001's evidence was at max_completion=1024 with pre-migration prompts; the current setup (2048 tokens, post-fc6c8f1 extern-C prompts, a code-specialized base) is different enough that the revisit condition could be *tested* — but it must be tested, not assumed. **Minimal compliant path**: (a) a ~30-60 min single-rollout probe measuring p(parseable+compilable extern-C block) from base on the migrated prompts; if ≥5%, cold start is ledger-legal with documentation; (b) if <5%, EXP-018b as specced — rewrite Sakana 400 → ~300-320 extern-C rows (CAMPAIGN-018:83-90), respecting the length-filter lesson (do-not-repeat.md:140-158: bodies ≤~2500 chars or keep max_completion ≥2048), then a ~50-step Stage-2 SFT (EXP-002 precedent, notes.md:101-114: loss 2.5→0.5 in 50 steps, a few GPU-hours), then warm-start 018c.

## 2. GHOST RUN — yes, it violates the project's own conventions

Slurm 7359525 demonstrably ran and produced data: `exp018c.slurm:60` — "7359525 v1 measured 19min/step at G=2 + max_completion=2048"; git log shows launcher committed (97a633d) then "v2: cap max_steps to 25" (5f22f85) ~1.6h later. Yet: no `EXP-018c.md` exists, `results.tsv` has no row after 2026-05-27, and `notes.md` ends at the 2026-05-28 entry. This breaks all three verification-phase requirements at `notes.md:49-64` (spec file, banner, `-verify` results row) and the Format Per Entry at `notes.md:70-82`. Precedent is unambiguous: even init-time config failures got rows (results.tsv lines 15-20, EXP-008-C/F/D-FAILED). Worst detail: `EXP-016p-spike.md:113` says "See [EXP-018c](EXP-018c.md) for the actual verification result" — a dangling link to a file that never existed. **Ledger should say**: `EXP-018c-v1-verify | null | explorer-h200 | 7359525 | metrics null | promote=false | "VERIFICATION ONLY — VOID: task pool self-contradictory (15/16 prompts demanded pybind, evaluator hard-rejects pybind; fixed in fc6c8f1); ~19 min/step throughput measured; result uninterpretable"`. Relaunching before closing this loses the one datum the run produced — the 19 min/step figure that falsifies the 50-step/8h plan.

## 3. PASS-CRITERIA DRIFT — criteria are mechanically uncomputable as launched

- `CAMPAIGN-018:121-131` and `exp018c.slurm:10-13` both state: `mean(reward[40:50]) - mean(reward[0:10]) > 0.15`, linregress over 50 steps p<0.05. Then `exp018c.slurm:60` sets `MAX_STEPS=25`. Steps 40-50 do not exist in a 25-step run; criteria 1 and 2 as written are undecidable — the launcher contradicts its own header.
- Walltime math: 25 × 19 min = 475 min against a 480-min walltime (`exp018c.slurm:24`) before model load; the file contains zero `save_steps`/`resume`/`requeue`, despite `CAMPAIGN-018:132-134` specifying "2 × 8h slurm with auto-resume." The run likely SIGTERMs before step 25 with nothing checkpointed.
- Statistical resolution: G=2 with generation_batch_size=4 (do-not-repeat.md:244-260) yields ~4 rollouts/step, so a 10-step window is ~40 rollouts; on reward support {-1,-0.5,0,1,2,3} the window-mean SE is ~0.1, making the 0.15 threshold ~1.5 SE — marginal at 50 steps, meaningless at 25. EXP-016p's criteria (EXP-016p-spike.md:85-97) were statistically stronger (bootstrap CI + plausibility checks); 018c *weakened* them.
- Deep anti-hack scan script (CAMPAIGN-018:128-131, "script TBD as part of EXP-018a deliverables"): **does not exist**. Criterion 3 is also undecidable. Related: EXP-018a.md:155-163 — the Run section is all TBD, including the fixture-replay verdict (5/5 hack rejection required to proceed to 018b); `tests/` contains no replay of the 5 hacked EXP-015'-A completions. EXP-018a's own acceptance test has no recorded verdict.
- Bonus pool drift: `exp018c.slurm:64` comments "32 rows"; the live pool is 16 rows, effectively 14 (EXP-018d-contract-migration.md:44-60), and the campaign specifies a 3-5 task subset (CAMPAIGN-018:104-105; concrete 3-task pool at EXP-018a.md:121-141) — the launcher runs the full filter including 2 permanently dead tasks that pin reward at -1 and produce zero-variance G=2 groups.

## 4. NAMING COLLISION — yes, referential integrity is corrupted

CAMPAIGN-018:137-153 defines EXP-018d as the A100 promotion rerun, "blocked on EXP-018c PASSING." The shipped `research/experiments/EXP-018d-contract-migration.md` (commit fc6c8f1) is an unrelated prompt-hygiene fix that even declares "Parent: EXP-018a/c." Consequences: any future ledger row saying "EXP-018d" is ambiguous, and the campaign's gate now falsely appears violated ("018d" shipped while 018c never passed). **Fix** (memory-keeper, no code): renumber the campaign's A100 rerun (to EXP-018e/f) and annotate that the 018d identifier was consumed on 2026-08-03; do not rename the shipped file — its name is bound in the commit history. Second, worse integrity defect: CAMPAIGN-018:6-7 cites "A, B, D in `research/notes.md` 2026-05-30 audit" — **no such entry exists**; notes.md ends at 2026-05-28. The campaign's foundational motivation cites a nonexistent ledger anchor.

## 5. PLAN GAPS — ownerless decisions, sorted by whether they block the 018c rerun

**Block the rerun:**
- EXP-018a acceptance verdict (5/5 hacked-completion replay, EXP-018a.md:159-162) — unrecorded; the campaign's own chain conditions 018b→018c on it.
- Deep anti-hack scan script — nonexistent; without it (or a rewritten criterion 3) the rerun cannot pass or fail cleanly. Must also confirm the pipeline persists `.cu` sources of correct rollouts for post-hoc scanning.
- Launcher criteria/max_steps/auto-resume contradiction (answer 3) — must be rewritten before any relaunch.

**Do not block the rerun (but block promotion later):**
- E3 signature class vs relabel-unsupported (EXP-018d-contract-migration.md:49-60) — moot if the rerun is aligned to the campaign's 3-5 task subset; cheap safe default is relabel-unsupported now, E3 later.
- Frozen-evaluator / reward-gate / holdout items from the 2026-08-03 audit — the rerun is verification-phase, local, non-promotion (notes.md:49-64 explicitly decouples these), so none block it. All block the promotion-eligible A100 run. Exception: the reward-gate item should be settled first only if it would change v3-symbol-shaped semantics.

## 6. NEXT ACTION — one experiment: EXP-018c-p0, cold-start feasibility probe (no training)

- **Hypothesis**: base Qwen3-Coder-30B-A3B, given the migrated (fc6c8f1) extern-C prompts at max_new_tokens=2048, emits parseable + nvcc-compilable `extern "C"` kernels at p ≥ 0.05 — the exact revisit condition of do-not-repeat 2026-05-16.
- **Parent master hash**: null.
- **Config**: single-rollout diagnostic per the do-not-repeat.md:160-184 mandated pattern, driven through `scripts/probe_extern_c_eval.py` / `scripts/debug_eval_pipeline.py`; tasks = the EXP-018a 3-task spike pool (vector_add_e2, F.elu, F.softplus); 10-16 samples/task, temperature 1.0; log extract/compile/nm-symbol/correct buckets via `[ROLLOUT_INLINE]`/`[VERIFIER_INLINE]`. Runner: explorer-h200, ~0.5-1 h, $0.
- **Decision rule**: p ≥ 0.05 → cold-start 018c rerun is ledger-legal (document the satisfied revisit condition), rerun with fixed criteria on the 3-task pool with auto-resume; p < 0.05 → EXP-018b SFT rebuild is mandatory first, and every cold-start relaunch proposal dies. Either branch also delivers EXP-018a's unrecorded Day-6 smoke.
- **Not a duplicate**: no base-model output has ever been run through the extern-C evaluator with post-migration prompts.
- **Prerequisites (bookkeeping, zero GPU)**: close ghost 7359525 and renumber 018d before the probe's results are written.

---

**Overall judgment.** The campaign's *direction* is coherent and genuinely follows from the evidence: the 2026-05-28 invalidation proved the torch-extension channel is structural, the extern-C unification is the fix the ledger itself prescribed, and "one cheap verification spike before any further investment" is the right posture. But the *execution* has detached from the plan at four points — 018b silently skipped, 018c run and never recorded, the shipped 018d colliding with the planned 018d, and the campaign citing a notes.md audit entry that does not exist. The single most dangerous inconsistency is the 018c launcher itself (`scripts/cluster/exp018c.slurm:60` + `:66-70` vs its own header `:10-13`): it cold-starts from base in direct violation of the do-not-repeat ledger *and* caps at 25 steps while its pass criteria are defined over steps 40-50 with a deep-scan script that doesn't exist — meaning a rerun as-is would spend 8 GPU-hours to produce an outcome that can neither pass nor fail the campaign gate. That is a repeat of the project's own worst historical pattern (EXP-005/006/007: 1500 rollouts through a reward path that couldn't answer the question asked). Close the ghost run, fix the launcher's arithmetic, and run the 1-hour probe before anyone trains anything.
