# 2026-08-10 reviewer audit — governance & hack-channel integrity

Read-only audit at commit fc6c8f1. Scope: does the experimental governance hold
up — can any future result from this repo be trusted and compared, or are there
integrity holes that would invalidate the next run the way EXP-015'-A was
invalidated.

---

## Findings (ranked by severity)

**1. CRITICAL — An EXP-018c rerun today would be auto-promoted to master by the cold-start rule, because 018c carries zero of the three required verification-phase markings.**
- Evidence: `research/live/master.json` (`hash=null`); cold-start convention at `research/notes.md:22-27` ("first managed run that produces parseable mean_reward and pass_rate is **unconditionally promoted**"); verification-marking convention at notes.md:49-64 requires (1) filename `EXP-NNN-verification-*.md`, (2) header banner, (3) `-verify` tsv row. EXP-018c has **no experiment file at all**, no tsv row, and its only "NOT PROMOTION-ELIGIBLE" text lives in `scripts/cluster/exp018c.slurm:15` — not a governance surface memory-keeper reads.
- Why it invalidates: an H200/`local` verification result gets written as master, which under the backend-lock rule (notes.md:40-45) then locks comparability to `local` and blocks the planned A100 promotion — a repeat of the EXP-015'-A "promoted by cold-start technicality" failure shape.
- Minimal fix: create `research/experiments/EXP-018c-verification-extern-c.md` with the banner and pre-register the `-verify`/`promote=false` row before dispatch.

**2. CRITICAL — The extern "C" path closes the EXP-015'-A attack class but leaves three neighbor channels open.** Channel-by-channel:
- torch:: delegation (EXP-015'-A class): **BLOCKED** — source pre-scan before compile at `eval_service/eval_core.py:695` (`scan_source_forbidden`, patterns at `openenv_env/anti_hack.py:44-51`), raw nvcc with no libtorch link (eval_core.py:724), post-link `nm -D` scan (eval_core.py:749).
- Reference-output scraping: **OPEN**. The reference is computed in the *same process and CUDA context* immediately before the candidate runs: `ref_output_f` at eval_core.py:804, then `candidate_output = torch.empty_like(ref_output_f)` at :806 — with torch's caching allocator the two buffers sit at a predictable offset. `run_kernel` is *host* code invoked via ctypes (eval_core.py:776-779, 620-632); it can legally call `cudaMemcpy` device-to-device from the adjacent block into `out`. Output is input-dependent, non-constant across seeds, not a passthrough, shapes match, runtime >1μs — all five checks at eval_core.py:824-963 pass, likely scoring reward 2-3. Subprocess isolation (`openenv_env/eval_backend.py:158-219`) does not help: ref and candidate share the worker.
- Timing gaming via call-count state: **OPEN**. Correctness uses 5 calls (eval_core.py:790-810); timing then reuses the same dlsym'd function (eval_core.py:919-931) and **the benchmark output is never re-verified**. A static counter in the .so can compute correctly for 5 calls then launch a 1-element dummy kernel — beating `check_not_noop`'s 1μs floor (anti_hack.py:178) while producing reward 3.0.
- Precision/tolerance gaming: **OPEN, unchanged since the invalidation**. `_assert_close` still `rtol=1e-3, atol=1e-3` (eval_core.py:161) — the exact value master.json's invalidation comment cites — and the harness *appends `--use_fast_math` by default* when the candidate supplies no CU_FLAGS (eval_core.py:82).
- Constant-folding on fixed inputs: **PARTIALLY OPEN**. Eval inputs are deterministic forever (`torch.manual_seed(42+seed)`, seeds 0-4, eval_core.py:790-791; timing reuses seed 42 at :866). Hardcoding is blocked only by output size — i.e., blocked by accident, not by design.
- Also **UNCLEAR/weak**: anti-hack steps 9 and 10 are wrapped in `except Exception: pass` (eval_core.py:861-862, 962-963) — a crash inside the checks silently passes the candidate.
- Minimal fixes: verify one timed benchmark output against ref after timing; compute reference in a separate CUDA context/process or randomize allocation; drop default `--use_fast_math`; per-call random seeds.

**3. HIGH — Ledger integrity: slurm 7359525 (June 018c v1) ran, changed the project's config, and left no record.**
- Evidence: the only trace is `scripts/cluster/exp018c.slurm:60`. No results.tsv row (ledger ends 2026-05-27), no notes.md entry (ends 2026-05-28), no experiment file. Violates: AGENTS.md:75-76 hard rule, notes.md:70-82 per-entry format, notes.md:49-58 verification marking. Related: `research/experiments/EXP-018a.md:155-163` — the Run section is all TBD, meaning the campaign's own acceptance test ("replay EXP-015'-A's 5 hacked completions, 5/5 must be rejected", EXP-018a.md:151) has **no recorded verdict**. The claim "the channel is closed" currently has no benchmark evidence in the ledger.
- Minimal fix: retroactive tsv row + notes entry for 7359525; run and record the 5/5 fixture replay before the rerun.

**4. HIGH — The exp018c slurm re-enters a do-not-repeat family and cannot satisfy its own pass criteria.**
- `scripts/cluster/exp018c.slurm:70` sets `KERNELFORGE_STAGE1_INIT_CKPT=""` (pure cold start) vs `research/do-not-repeat.md:62-79`; campaign marks 018c "blocked on EXP-018a + EXP-018b" (CAMPAIGN-018:132). Header pass criteria assume 50 steps while line 60 sets `MAX_STEPS=25` — mechanically unable to pass or fail its own spec. A flat curve would be uninterpretable (base-model exploration failure vs design failure — the EXP-001 confound).
- Minimal fix: written amendment addressing the do-not-repeat condition (pre-registered), or wait for 018b; rewrite pass criteria for actual step count; restore 2×8h auto-resume.

**5. HIGH — "Frozen evaluator" is declared but unenforced and unversioned; no result can be attributed to an evaluator state.**
- AGENTS.md:64-65 and `.claude/skills/kernelbench-eval/SKILL.md:11` declare the evaluator frozen, yet EXP-018a's entire deliverable was rewriting `evaluate_ops6k_kernel_impl`, and audit commits (21e6852, cab09ef) touched evaluation/verification. No evaluator hash/version mechanism exists; results.tsv and master.json carry no evaluator-version field. A results row cannot be tied to the evaluator that produced it.
- Minimal fix: add an `evaluator_sha` (git blob hash of eval_core.py + anti_hack.py) column to results.tsv and field to master.json; amend AGENTS.md to say "frozen except via a campaign-authorized experiment that bumps evaluator_sha and re-opens cold start."

**6. MEDIUM-HIGH — The promotion gate's comparability fields are un-instrumented, and task-pool identity is not part of the contract at all.**
- Gate requires `eval_split`, `seed_count`, `max_turns`, `reward_version`, `eval_backend` to match (notes.md:35-37), but results.tsv has a column only for `reward_version`; nothing in code emits the rest (and hand-filling already failed once: EXP-015'-A's reward_version was recorded wrong). Task pool: effective pool is 14 (two 3+-tensor rows permanently error), while `exp018c.slurm:64` claims "32 rows" and the campaign assumed 16. Pool hash, dataset version, init-ckpt, and evaluator hash are all missing from the schema.
- Minimal fix: add `eval_split`/`seed_count`/`max_turns`/`eval_backend`/`task_pool_hash`/`init_ckpt` columns; resolve the two dead rows before the rerun.

**7. MEDIUM — Trainer-side vs evaluator-side signature inference can still diverge (the June-018c poison class is narrowed, not eliminated).**
- `training/task_support.py:168-222` `infer_signature_class` falls back to a static AST count on *any* exec failure, whereas `eval_service/eval_core.py:588-608` counts *tensors only* and has no fallback. Divergence cases: `[tensor, scalar]` returns (static says E2, eval says E1 — args misbound, the exact 018c-v1 poison); comprehension/variable returns. Migrated rows embed contract text frozen at migration time; `build_generation_prompt` (task_support.py:296-298) suppresses the fresh contract only on exact substring match.
- Minimal fix: one-shot consistency test asserting trainer-side class == eval-side class for all live rows, run in CI and before each dispatch.

**8. LOW-MEDIUM — Referential-integrity list for research/:**
- `EXP-016p-spike.md:113` (uncommitted): links nonexistent `EXP-018c.md`.
- `CAMPAIGN-018:6-7`: cites nonexistent "notes.md 2026-05-30 audit".
- "EXP-018d" means two different things (campaign §137 A100 rerun vs shipped contract-migration bugfix).
- `EXP-018a.md:26-27` specs `extern_c_signature` in the dataset; commit 51eb7c9 dropped it — spec/reality divergence with no amendment.
- Pool-size claims disagree: 32 (`exp018c.slurm:64`) vs 16 vs 14 effective.
- notes.md:390 / results.tsv:22 — EXP-009-B "parent master hash 117bb5d" while every other pre-promotion row records null.
- Stray untracked `datasets/combined_kernelforge.jsonl.bak`.

**Positive confirmations:** `KERNELFORGE_REWARD_VERSION=v3-symbol-shaped` is genuinely what `openenv_env/reward.py:88-113` computes; the −0.5 symbol bucket is correctly wired to the ops6k path; 4 of the 5 anti-hack checks plus the symbol probe do fire on the extern-C path.

## Verdict

**Conditional-no.** Dispatched today, the EXP-018c rerun would produce a number, but not a trustworthy or governable one: (a) with master=null and 018c unmarked, the cold-start rule would promote a local-H200 verification result to master; (b) the run violates the do-not-repeat cold-start condition and cannot mechanically satisfy its own criteria; (c) any `correct=True` rollout requires post-hoc audit because scraping/timing channels remain open and the 5/5 replay was never recorded; (d) the result could not be pinned to an evaluator version or pool composition. Trust becomes "yes" once five cheap zero-GPU steps land: verification file + pre-registered row; retro-record 7359525; run and record the fixture replay; amend/satisfy the cold-start condition and fix the criteria mismatch; add evaluator-hash and pool-hash fields (and resolve the 2 dead tasks). The timing/scraping channels should be closed before any speedup-bearing promotion, but do not block the binary "does the curve move" question if every correct rollout is source-audited.

## Residual measurement risk

Even after fixes: eval inputs are permanently fixed (seeds 42-46), so multi-step reward improvement partially measures overfitting to five deterministic input sets; H200-local timing feeds reward tiers 2/3 during verification despite the project's "never use H200 execution for performance reward" rule; the anti-hack `except: pass` wrappers mean a silent environment fault degrades hack detection to zero without signal.
