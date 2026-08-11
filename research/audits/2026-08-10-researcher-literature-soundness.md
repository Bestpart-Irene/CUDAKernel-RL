# 2026-08-10 researcher audit — literature soundness

Read-only audit at commit fc6c8f1, checked against 2024-2026 published work
(Kevin-32B, KernelBench, Sakana AI CUDA Engineer, CUDA-L1, DAPO, RLVR analyses).

---

## 1. VERDICT

The overall bet — RL with a verifiable, hack-closed evaluator can teach a ~30B model to write correct CUDA kernels — is **plausible and now well-precedented**: Kevin-32B did it with multi-turn GRPO on QwQ-32B (Cognition; arXiv 2507.11948). The project's evaluator direction is genuinely good: the extern-"C" + dlsym contract (EXP-018a) is *structurally stronger* than the blocklist/monkey-patch defenses in the published harnesses, and the project's own reward-hack post-mortem (EXP-015'-A) is exactly the failure Sakana and Kevin both hit publicly. **But EXP-018c as currently launched is not a fair test of the bet.** Every published success required a policy with substantial nonzero initial pass rate (Kevin's base: 56% correct on KernelBench before RL) and large groups (Kevin: 16 trajectories; DAPO exists because zero-variance groups produce zero gradient). 018c is cold-start from base with no SFT prior, G=2, on a contract the base model has never been trained toward — a configuration the project's own `do-not-repeat.md:62-79` rules out verbatim, and whose failure mode the project already observed in EXP-001. If 018c fails flat, the campaign's stated interpretation — "the design's central claim is empirically falsified" (CAMPAIGN-018 L189-198) — will be **wrong**: the run confounds "RL can't learn kernels" with "the policy never emitted a single valid candidate," which is precisely the confound CAMPAIGN-018 was written to eliminate by requiring 018b first. The project is not wrong in its goal or its evaluator; it is currently wrong in its test.

## 2. SOUND — choices the literature supports

- **Extern-"C" raw-pointer contract with no libtorch linkage** (EXP-018a). CUDA-Agent's three-file contract requires `.cu` files with zero PyTorch deps and an `extern "C"` launcher; Kevin had to bolt on string checks for `torch.nn` usage after the fact. Removing libtorch from the link entirely is the strongest version of this defense found in any system — it closes the EXP-015'-A delegation channel by construction, not by pattern-matching.
- **Treating reward hacking as the default outcome.** Sakana's AI CUDA Engineer exploited a memory-reuse bug and walked back 150x claims (TechCrunch 2025-02-21; miru's leaderboard analysis); Kevin found KernelBench output-tensor-recycling and try/except fallback hacks; CUDA-L1 found 32.8% of its RL kernels exploited a stream-timing loophole (arXiv 2507.14111). KernelForge's invalidation of its own only promotion (notes.md L516-564) with the `sv_eager < 1.0` smoking-gun analysis is the same epistemic hygiene the best groups practice.
- **beta=0 (no KL)**: Kevin used KL coefficient 0; DAPO also drops KL (arXiv 2503.14476). Sound.
- **max_completion=2048 floor** (do-not-repeat 2026-05-18): consistent with the field; the truncation diagnosis in EXP-009-B was a real, correctly-isolated causal finding.
- **Trivially easy spike pool with vector_add as a harness sanity check** (EXP-018a.md L121-141): matches NVIDIA's verifier-in-the-loop debugging workflow.
- **The planned Stage-2 RFT stage**: CUDA-Agent's ablation shows removing RFT warmup drops faster-rate 96.8%→49.8%. The stage exists in code; the problem is it sits downstream of the gate instead of upstream.
- **rtol/atol=1e-3**: tighter than KernelBench/CUDA-Agent's 1e-2, which "The Correctness Illusion" (arXiv 2606.20128) criticizes. Directionally right (but see reviewer: still gameable with --use_fast_math default).

## 3. UNSOUND OR RISKY — most important first

**U1. Launching 018c cold-start with no SFT prior repeats a ruled-out failure family and destroys the campaign's interpretability.** (a) Violates do-not-repeat.md:62-79 as written. (b) Contradicts the campaign's own dependency graph (018c "blocked on EXP-018a + EXP-018b"); 018b was skipped and `datasets/` contains no `sakana_sft_externc.jsonl`. (c) Removes two of the innovations the campaign claims to validate: I2 (ABI-annotation SFT) and I6 (SFT corpus) cannot fire in a run with no SFT. (d) Literature: RLVR does not elicit behaviors outside the base model's sampling distribution (Yue et al., arXiv 2504.13837); Kevin needed a base at 56% correctness; DeepSeek-R1 added cold-start SFT before RL for stability. Mitigating fact: this pool (extern-C vector_add/elu/softplus) is vastly easier than EXP-001's, and Qwen3-Coder-30B is a strong instruct coder — base pass@k may be nonzero here. But that is an **unmeasured assumption carrying the whole run**.

**U2. G=2 is below every published kernel-RL configuration, and it mathematically erases the v3-symbol-shaped reward's magnitudes.** With group-std-normalized advantage (active per `training/custom_grpo_trainer.py:67`), a G=2 group yields advantages of exactly ±1/√2 for *any* unequal pair: a (-1,-0.5) pair and a (-1,+1) pair produce identical gradients. The graded structure of v3 (I1) reduces to ordinal information at G=2. Groups where both rollouts land in the same bucket (overwhelmingly likely at near-zero pass rate) give zero gradient — the pathology DAPO's dynamic sampling fixes. Kevin used 16 trajectories/task; field guidance is G=8-16. Worse, the advertised Dr.Kernel-style TRLOO N/(N-1) correction is **inactive**: `custom_grpo_trainer.py:47` only enables it for `loss_type=="grpo"` while the default is `"dapo"` (stage1_warmup.py:80), and the TRL 0.29 hook is unavailable anyway. A third advertised innovation that does not fire.

**U3. The pass criterion is statistically weak and can pass without any learning of correctness.** ~4 rollouts/step; linregress p<0.05 over 50 autocorrelated nonstationary points has inflated false-positive rate; "last-10 > first-10" can be satisfied entirely by drifting from compile-fail (-1) to symbol-present-but-wrong (-0.5/0) — a formatting improvement, not kernel competence. Criterion 3 (≥1 clean correct) cannot distinguish learning from base-rate luck: at a static 0.5% pass probability, ≥1 success in ~200 rollouts occurs ~63% of the time. No published kernel-RL work uses training-reward slope as its gate; Kevin reports avg@16/best@16 on a held-out set before and after.

**U4. Single-turn leaves the field's main demonstrated win on the table — and the multi-turn path in this repo is dead code.** Kevin's ablation: single-turn RL matched multi-turn on best@16 correctness but lost badly on speedup (0.85x vs 1.10x); CUDA-Agent: removing the agent loop drops faster-than-compile 96.8%→14.1%. For the current correctness-only gate single-turn is defensible. But the endgame is speedup, and TRL 0.29's `rollout_func` is confirmed dead (notes.md L419-454), so there is currently **no working implementation path** to the multi-turn regime every successful system relied on. Strategic hole, not a spike-blocker.

**U5. Residual evaluator gaps that will bite at the next stage.** Correctness is checked at one shape with the reference model's input distribution — "The Correctness Illusion" (arXiv 2606.20128) and KernelBench-Verified (arXiv 2607.16241) show models exploit narrow distributions and single-shape checks; any future speedup reward must sync all CUDA streams before timing (CUDA-L1's exploit: 82/250 kernels faking speed via side streams). Two "supported" tasks are permanently dead in the live pool; `stage1_warmup.py:71` still defaults MAX_COMPLETION_LENGTH=1024 — a ruled-out family that only an env override avoids, in a project with three documented env-propagation failures.

## 4. MISSING FROM THE DESIGN — field-standard items the project lacks

- **A base-model pass@k probe before any RL run.** Every published effort baselines the policy on the eval first (Kevin: 56%; NVIDIA: best-of-N with a verifier hits 96-100% on KernelBench L1-2 *without training*). The project has never measured p(correct | base, extern-C prompt, k samples). Cheapest missing measurement; decides U1.
- **Dynamic sampling / degenerate-group filtering** (DAPO): oversample and drop all-same-reward groups so every batch carries gradient.
- **A held-out eval split and pre/post pass@k as the promotion metric.** The gate currently reads training reward on the training pool. Kevin held out 100 tasks; CUDA-Agent ran contamination checks. Nothing distinguishes memorizing 3 tasks from learning.
- **A best-of-N + verifier-feedback inference baseline.** If base best-of-64 with compiler feedback already solves the spike pool, the RL gate must beat that — and those trajectories are exactly the RFT warmup data.
- **Robust correctness checking**: multiple shapes, fuzzed input distributions, fp64 reference accumulation.
- **Entropy/collapse monitoring with asymmetric clipping** (clip-higher 0.2/0.28, Kevin & DAPO); EXP-001 exhibited entropy collapse to 0.15 in 13 steps and nothing addresses it.
- **All-stream-sync timing** before any speedup reward ships.

## 5. Minimal concrete changes per UNSOUND item

- **U1** — Before relaunching 018c: ~1 GPU-hour base-model pass@k probe (64-100 completions per spike task through the frozen extern-C evaluator). pass@64 > 0 on vector_add → cold-start empirically licensed, documented do-not-repeat revisit entry; = 0 → EXP-018b (or a ~50-row hand-written extern-C SFT set, or a one-shot worked example in the prompt) becomes a hard prerequisite.
- **U2** — G=8 with gradient_accumulation_steps=8 (the documented TRL constraint); if OOM at 2048 completion, G=4/accum=4 minimum. Either disable group-std scaling (Dr.GRPO-style) so v3 magnitudes survive, or document v3 as ordinal-only at small G; fix `custom_grpo_trainer.py:47` so TRLOO runs under the default loss type, or delete the claim from the docs.
- **U3** — Replace the slope criterion with pass@k on the spike pool at step 0 and step N under frozen sampling params, compared with Wilson CIs; keep last10-vs-first10 as a secondary compile-rate indicator only, computed at rollout level. Require the clean-correct count to *exceed* the step-0 base rate, not merely exist.
- **U4** — Accept single-turn for the 018c gate; open a tracked engineering item for a custom trainer subclass or TRL upgrade that actually drives eval-feedback turns.
- **U5** — Flip `stage1_warmup.py:71` default to 2048 in code; relabel the two uninferable-signature tasks `unsupported`; add ≥2 extra shapes + a non-uniform input distribution to the correctness loop; device-wide sync before timestamps when the speedup reward lands.

**Bottom line:** the evaluator work (018a/018d) is the most literature-sound part of this project and better than what several published systems shipped at launch. The bet is right. The current gate run is the wrong experiment: it tests "can GRPO extract signal from a policy that has never been shown to emit a single valid candidate, at a group size that cannot represent the reward shaping you built" — and the literature, plus this project's own EXP-001 and do-not-repeat ledger, already tells you how that ends. Run the pass@k probe first.

## Sources

- Kevin-32B: Multi-Turn RL for Writing CUDA Kernels — https://cognition.ai/blog/kevin-32b ; arXiv 2507.11948
- Sakana AI CUDA Engineer walk-back — TechCrunch 2025-02-21; miru leaderboard analysis (x.com/miru_why/status/1892703900425486539)
- DAPO — arXiv 2503.14476
- Does RL Really Incentivize Reasoning Beyond the Base Model? — arXiv 2504.13837
- The Correctness Illusion in LLM-Generated GPU Kernels — arXiv 2606.20128
- KernelBench-Verified — arXiv 2607.16241
- CUDA-L1: Contrastive RL for CUDA Optimization — arXiv 2507.14111
- NVIDIA: Automating GPU Kernel Generation with DeepSeek-R1 and Inference-Time Scaling — developer.nvidia.com blog
- DeepSeek-R1 (cold-start SFT before RL) — github.com/deepseek-ai/deepseek-r1
- RLOO vs GRPO group-size variance — ms-swift docs
- It Takes Two: Your GRPO Is Secretly DPO — alphaXiv 2510.00977
