# KernelForge Hackathon RL Environment Single-Source Notes

Last reviewed: March 5, 2026

## Purpose

This file is the audit note for the current KernelForge direction. It does not replace the PRD or GRPO doc. It exists to give another engineer one defensible reference for what is true in code today, what is good and should stay, what must change, and what belongs in future research rather than hackathon claims.

This note treats the **RL environment** as the hackathon center of gravity, while keeping it explicitly **complementary** to the broader recipe:

- OpenEnv-compatible RL environment
- SFT warmup
- small-budget GRPO pilot
- search / test-time compute
- GEPA / DSPy prompt-context evolution
- broader JEPA-style optimization ideas

## Review Protocol

To handle very large docs without pretending every sentence is equally novel:

- Every line in the PRD and GRPO docs inherits the verdict of its nearest parent section unless called out in a hotspot override below.
- Repeated task-list lines inherit the verdict of the execution section they belong to.
- Claim-heavy sections are reviewed separately from explanatory sections.
- Code wins over docs when they disagree.

## Inputs Reviewed

Conversation themes reviewed:

- RL environment is the weekend priority.
- The overall product is not only an environments project.
- GEPA / DSPy / JEPA / test-time compute / search remain part of the long-term recipe.
- The hackathon asks for an RL environment, so the environment must be real and defensible on its own.

Repo docs reviewed:

- `docs/KERNELFORGE_FINAL_PRD.md`
- `docs/GRPO_DEEP_DIVE.md`
- `docs/skills/DOUBLEGRAPH_A100.md`

Key code reviewed:

- `openenv_env/kernel_forge_env.py`
- `openenv_env/reward.py`
- `openenv_env/skill_builder.py`
- `training/task_support.py`
- `training/dataset_loader.py`
- `training/multi_turn_rollout.py`
- `training/custom_grpo_trainer.py`
- `training/stage3_grpo.py`
- `training/model_loader.py`
- `training/grpo_train.py`
- `modal_app.py`

Local checks run:

- `wc -l datasets/combined_kernelforge.jsonl datasets/doublegraph_sft.jsonl`
- `./.venv/bin/python -c 'from training.dataset_loader import load_training_dataset; from training.task_support import summarize_tasks; rows=load_training_dataset(stage="stage3", ops6k_max=128); print(len(rows)); print(summarize_tasks(rows))'`
- `./.venv/bin/python training/grpo_train.py --preflight-only`

Observed local facts:

- `datasets/combined_kernelforge.jsonl`: 224 rows
- `datasets/doublegraph_sft.jsonl`: 192 rows
- Stage-3 live-evaluable rows after filtering: 19 total
- Stage-3 live-evaluable split: 15 `ops6k`, 4 `wcc`
- Preflight currently fails on missing Modal auth, not on code import

External sources reviewed:

- OpenEnv: <https://github.com/meta-pytorch/OpenEnv>
- TRL OpenEnv docs: <https://huggingface.co/docs/trl/en/openenv>
- Qwen3-Coder-30B-A3B-Instruct: <https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct>
- DSPy: <https://github.com/stanfordnlp/dspy>
- CUDA-Agent project: <https://cuda-agent.github.io/>
- CUDA-Agent-Ops-6K dataset: <https://huggingface.co/datasets/BytedTsinghua-SIA/CUDA-Agent-Ops-6K>
- SkyDiscover: <https://skydiscover-ai.github.io/>
- DeepSeekMath / GRPO: <https://arxiv.org/abs/2402.03300>
- Dr. Kernel: <https://arxiv.org/abs/2602.05885>
- Process Supervision-Guided Policy Optimization for Code: <https://arxiv.org/abs/2410.17621>
- RLEF: <https://arxiv.org/abs/2410.02089>
- StepCoder: <https://arxiv.org/abs/2402.01391>
- Multi-Turn Code Generation Through Single-Step Rewards: <https://arxiv.org/abs/2502.20380>
- CUDA-L1: <https://arxiv.org/abs/2507.14111>
- GEPA: <https://arxiv.org/abs/2507.19457>
- AdaEvolve: <https://arxiv.org/abs/2602.20133>

## Executive Verdict

The repository is already credible as a **hackathon-scoped CUDA RL environment pilot**, but not as a credible claim of CUDA-Agent reproduction or broad one-GPU benchmark parity.

The strongest defensible story today is:

1. The environment path is real for a narrow live-evaluable slice.
2. The evaluator path is real for WCC and a stateless Ops subset.
3. The reward is real: discrete milestones `{-1, 1, 2, 3}` per CUDA-Agent ablation (96.8% vs 60.4% faster rate over continuous).
4. TRLOO exists as a subclass but the N/(N-1) correction is INACTIVE under TRL 0.29 (no `_compute_advantages` hook; loss-type gate); the trainer warns at init — 2026-08-10 audit.
5. SFT-first is still the correct near-term posture.
6. Search remains the strongest hedge and should stay complementary.

The weakest parts are:

1. Docs still overstate model defaults, task coverage, and future-scale RL maturity.
2. Only 19 of 224 combined rows are live-evaluable today.
3. MARS, CPPO, and MASPO are still mostly doc-level or stretch-goal material.
4. The code does not yet justify “same results as CUDA-Agent on one GPU” as a claim.

## What Stays The Same

These parts are directionally correct and should remain:

| Item | Verdict | Why it stays | Counterargument to preserve |
|---|---|---|---|
| RL environment as hackathon center | Keep | The weekend problem is environment credibility under real compile/correctness/timing feedback. | This is the one part judges can actually test. |
| SFT before RL | Keep | Data scarcity and sparse reward still make pure-RL-from-base wasteful. | External code-RL work still supports prior shaping before online RL. |
| Search as hedge, not replacement | Keep | Search covers RL instability and gives demo value fast. | Search plus eval is complementary to environment work, not a retreat from RL. |
| doubleGraph patterns as prompt prior | Keep | Real production A100 priors materially reduce dead-on-arrival generations. | Using expert priors is a strength, not cheating. |
| OpenEnv compatibility | Keep | It is the hackathon surface area. | Even if training bypasses the HTTP server, the environment contract is still strategically important. |
| Claim discipline | Keep | The docs improved significantly once they stopped pretending full CUDA-Agent parity this weekend. | This discipline should get stricter, not looser. |

## Current Code Truth

### Implemented

- `openenv_env/kernel_forge_env.py` implements a real OpenEnv `step/reset/state` contract.
- `modal_app.py` has a real WCC path and a real stateless Ops extension-eval path.
- `training/task_support.py` centralizes evaluator routing and payload construction.
- `openenv_env/reward.py` computes discrete milestone reward `{-1, 1, 2, 3}` with correctness gating.
- `training/custom_grpo_trainer.py` carries the TRLOO `N/(N-1)` advantage scaling, but it is INACTIVE under TRL 0.29 (no `_compute_advantages` hook; only gated to `loss_type=="grpo"`); the trainer emits a RuntimeWarning at init.
- `openenv_env/skill_builder.py` injects hardware-aware SKILL context plus 7 doubleGraph A100 patterns.

### Partially Implemented

- Multi-turn rollout exists, but supported live tasks are narrow.
- Stage-3 GRPO exists, but docstrings and actual defaults disagree.
- Topology context exists, but graph-task correctness/runtime coverage beyond WCC is not broad.
- The OpenEnv server exists, but the main training path largely goes through TRL rollout helpers and direct Modal calls instead of exercising the HTTP server as the authoritative runtime loop.

### Not Yet Implemented or Not Yet Defensible

- MARS return-to-go credit assignment
- CPPO completion pruning
- MASPO soft trust region
- full Nsight Compute structured reward
- broad Ops-6K live evaluation beyond the stateless subset
- broad graph-task harness beyond WCC
- one-GPU CUDA-Agent-match claim

## First-Principles Read on the Environment

The environment only matters if it converts expensive remote evaluation into a trustworthy training signal. For this weekend, that means:

1. `compile -> correctness -> timing -> reward` must be true.
2. correctness must gate reward, otherwise speedup hacking wins.
3. target-hardware measurement must happen on A100 if the target is A100.
4. unsupported tasks must be filtered aggressively rather than silently poisoning RL.
5. search and prompt/context evolution should consume the same evaluator signal, not invent a separate fake proxy loop.

The current repo mostly satisfies points 1 through 4 for a narrow slice. That is enough for a pilot environment. It is not enough for large-scope claims.

## PRD Section Matrix

Each row below is the section-level verdict for `docs/KERNELFORGE_FINAL_PRD.md`.

| PRD section | Verdict | Can defend today? | What stays | What must change |
|---|---|---:|---|---|
| Executive Summary | Keep with small edits | Yes | Hackathon-first framing and anti-parity language are good. | Mention the live-evaluable slice is 19 tasks today, not generic 6K coverage. |
| RL Feasibility (Hackathon-Scoped) | Keep | Yes | Correctly says this is a pilot and not CUDA-Agent reproduction. | Tie feasibility to actual task coverage and current model loader defaults. |
| 0. Locked Decisions | Revise | Partial | A100 target eval, H100 train, SFT-first, search hedge are good. | Primary model claim does not match `training/model_loader.py` default. |
| Training GPU Selection | Revise | Partial | H100 vs A100 split is directionally correct. | Remove implied certainty around the 30B model being the code default. |
| 1. Strategic Framing | Keep | Yes | Good discipline on what is and is not being claimed. | None beyond consistency. |
| 2.1 Hackathon Deliverable | Revise | Partial | Components A, C, D, E are directionally right. | Component B overstates breadth; the live harness is narrow, not full generic Ops-6K coverage. |
| 2.2 Long-Term Research Platform | Keep as future | Yes | Good to keep long-term ambition visible. | Tag more aggressively as future work. |
| 3. Scope Boundaries | Keep | Yes | Strong and correct. | None. |
| 4. Training Strategy | Keep with edits | Yes | Environment validation first, then SFT, then small-budget GRPO, then search is correct. | Reward uses discrete milestones `{-1,1,2,3}` per CUDA-Agent ablation. |
| 5. Hackathon Configuration | Revise | Partial | The posture and abort conditions are right. | Model section conflicts with code default; “short outputs” and `G=2` are true for Stage 3, not across the whole training stack. |
| 6. Realistic Compute Budget | Keep as estimate | Partial | Budget discipline is right. | Keep clearly labeled as estimate; do not imply measured runtime. |
| 7. Claim Discipline | Keep | Yes | This is one of the best parts of the doc. | Apply it harder to later sections. |
| 8. Repository Structure | Revise heavily | No | The conceptual decomposition is useful. | Large parts of this section describe planned structure, not the current repo layout. |
| 9. Complete Task List | Revise heavily | No | Useful as a planning artifact. | Not a source of truth; many subtasks are stale, superseded, or already implemented differently. |
| 10. Critical Path | Keep as planning | Partial | Prioritizing P0 evaluator plumbing is correct. | Update to actual remaining blockers rather than historical task sequencing. |
| 11. All Links | Keep | Yes | Good reference section. | Nothing major. |
| 12. Risk Matrix & Mitigations | Keep with hotspot fixes | Partial | Strongest part of the lower half because it surfaces failure modes. | Must stop mixing resolved, speculative, and outdated issues without a clearer status system. |
| 13. Future Work: Scale-Up Feasibility Assessment | Split or quarantine | No for current truth | Fine as a future-research appendix. | Remove it from any “single source of truth” framing; it is speculative and repeatedly overshoots current code reality. |

## PRD Hotspot Claim Matrix

These are the claims most likely to mislead another engineer if left unqualified.

| Claim | Verdict | Why | Counterargument if keeping it | Action |
|---|---|---|---|---|
| “Primary model is Qwen3-Coder-30B-A3B-Instruct” | Not true in code by default | `training/model_loader.py` still defaults to `Qwen/Qwen3-Coder-Next`. | Keep the architecture intent, not the current-code statement. | Either change code default or label the doc as target config. |
| “Real evaluation harness (CUDA Agent pipeline)” | Partial | Real for WCC + stateless Ops subset only. | The harness itself is real and worth highlighting. | Narrow the wording to supported families. |
| “6,000 operator tasks” as part of current evaluable system | Not defensible | Combined dataset is 224 rows here; only 19 are live-evaluable after filtering. | Keep Ops-6K as external data source, not current live-coverage claim. | Explicitly separate source dataset size from live harness coverage. |
| “Discrete reward: {-1,1,2,3}” | True | Matches `openenv_env/reward.py:compute_reward()`. | This is the actual code truth per CUDA-Agent ablation. | Keep. |
| “CUDA-Agent SKILL.md verbatim + doubleGraph pattern paste” | Partial | doubleGraph pattern injection is real; exact verbatim CUDA-Agent SKILL usage is not the main runtime truth. | Keep the broader “expert prompt prior” idea. | Phrase as “hardware-aware skill context with doubleGraph A100 priors.” |
| “SkyDiscover hedge” | Directionally true | Search is integrated conceptually and has repo support. | Keep as hedge, not central proof of RL success. | Avoid implying it validates RL by itself. |
| “doubleGraph baselines for reward calibration” | Partial | Valid only for the graph slice; not for generic Ops tasks. | Keep graph-slice calibration logic. | Make the graph-only scope explicit everywhere. |
| “20 turns / 150 steps / match config” | Not defensible for current repo truth | Stage-3 defaults and docs still disagree; environment docstrings are stale. | Keep long-horizon work as future target. | Move to future appendix only. |
| “Matches CUDA-Agent at 30-60x lower cost” | Not defensible | Requires MARS/CPPO/MASPO/future model scale that are not shipped. | Keep as research hypothesis only. | Remove from current-source-of-truth positioning. |
| “2.05x more effective signal than CUDA-Agent” | Not defensible | This is a speculative synthetic accounting argument, not measured evidence. | None. | Drop from operational docs. |

## GRPO Section Matrix

Each row below is the section-level verdict for `docs/GRPO_DEEP_DIVE.md`.

| GRPO section | Verdict | Can defend today? | What stays | What must change |
|---|---|---:|---|---|
| Hackathon Path header | Keep | Yes | Good reframing away from default B200/Coder-Next parity claims. | Align model default with code or label as target config. |
| Core Recommendation | Keep | Yes | SFT-first, small-budget GRPO, search hedge, avoid pretending MARS/MASPO/CPPO are done. | None beyond consistency. |
| Full Pedagogical Deep-Dive | Keep as theory appendix | Partial | Useful explanatory material on RL/GRPO/TRLOO. | Must stop blending pedagogy with statements that imply implementation. |
| TRLOO-GRPO Math Deep-Dive | Keep | Yes | TRLOO explanation is still relevant. | The code section should reflect the actual reward and trainer code. |
| GRPO-1 The Algorithm | Keep as background | Yes | Good background for another engineer. | Separate background math from current build claims. |
| GRPO-2 Memory Budget | Revise | Partial | It is useful to explain why 30B-on-H100 is attractive. | Code default still points to Coder-Next; future B200 analysis should be clearly future-only. |
| GRPO-3 TRL GRPOTrainer Exact Configuration | Revise | Partial | Good to explain reward plumbing and dataset schema. | “Exact” is too strong where code/doc differ. |
| GRPO-4 MARS+TRLOO Multi-Turn Extension | Future / stretch only | No | MARS is still a good research direction. | It is not implemented and should not be treated as core hackathon path. |
| GRPO-5 Compute Budget | Keep as estimate | Partial | Budget framing is helpful. | Keep estimated, not measured. |
| GRPO-6 Monitoring | Keep | Yes | Useful operational checklist. | None. |
| GRPO-7 Critical Implementation Notes | Keep with edits | Partial | Good collection of practical failure modes. | Unsloth/vLLM notes should match the actual chosen model/runtime path. |
| GRPO-8 Quick Reference Card | Keep after cleanup | Partial | Useful for engineer handoff. | Remove stale items and future-only claims. |
| GRPO-9 Nsight Compute Structured Rewards | Future / optional | No | Worth exploring later. | Current code does not implement full Nsight reward. |
| GRPO-10 Hybrid Eval | Keep | Partial | Local compile check + remote target eval is directionally correct and partly implemented. | Avoid overselling local/remote sophistication beyond what code does today. |
| GRPO-11 CPPO Completion Pruning | Future / optional | No | Interesting cost-saving idea. | Not implemented. |
| GRPO-12 MASPO Soft Trust Region | Future / optional | No | Fine as a research note. | Not hackathon-relevant unless implemented. |
| GRPO-13 Transformation Grammar | Deferred correctly | Yes as deferred | Correct to push out of v1. | Keep clearly non-core. |
| GRPO-14 Full Stacked Architecture | Split | No as current truth | Useful as a speculative architecture sketch. | It mixes real and unbuilt components too aggressively. |
| GRPO-15 Hackathon Config + Future Scale-Up | Split | Partial | Hackathon subsection is useful. | Future scale-up subsection should not live inside the operational core doc. |
| GRPO-16 Presentation Framing | Keep | Yes | Good for pitch discipline. | None. |
| GRPO-17 Abort Gates | Keep | Yes | Operationally useful. | None. |

## GRPO Hotspot Claim Matrix

| Claim | Verdict | Why | Counterargument if keeping it | Action |
|---|---|---|---|---|
| TRLOO is implemented | Not true in effect | `training/custom_grpo_trainer.py` defines the `G/(G-1)` override, but TRL 0.29 never calls it (no `_compute_advantages` hook) and the gate excludes the default `"dapo"` loss. | The subclass is still the right seam if TRL restores the hook. | Treat as inactive; trainer warns at init (2026-08-10 audit). |
| MARS is part of the current core | Not true | No MARS implementation is wired in the current trainer or rollout. | Keep only as stretch goal. | Relabel everywhere. |
| Current reward wraps OpenEnv exactly | Partial | The training path uses rollout helpers and direct Modal dispatch more than the OpenEnv server. | The reward still comes from the same evaluation logic family. | Phrase as “OpenEnv-aligned evaluator path,” not exact runtime path. |
| 30B-on-H100 is the active default | Not true in code by default | Code default is still Coder-Next. | Keep as target hackathon config. | Align code or docs. |
| Nsight structured rewards are active | Not true | Current code uses optional simple profiling bonuses, not full `ncu`-based structured reward. | Keep as future option. | Relabel as future. |
| CPPO and MASPO are part of the planned weekend stack | Not recommended | They increase complexity with no shipped support today. | Keep in appendix only. | Strip from core config narrative. |
| Multi-turn depth is the main weekend differentiator | Overstated | Current live harness coverage is too narrow to make long-horizon RL the primary weekend proof. | Keep multi-turn as future extension. | Make environment correctness and reward plumbing primary. |

## Broader Recipe Matrix

This matrix covers the conversation-level directions, not just the two docs.

| Direction | Verdict | Why it should or should not work here | Current gap | Better framing |
|---|---|---|---|---|
| RL environment as main hackathon focus | Correct | This is the actual hackathon category and the codebase already has the strongest grounding here. | Docs still dilute this with future-scale claims. | Keep as the weekend center. |
| Pure RL from scratch | Incorrect for this repo | Sparse reward plus low task coverage makes this wasteful. | Compile/correctness signal would be too thin. | Keep SFT-first. |
| SFT-first | Correct | doubleGraph priors + prompt priors help enormously under tight budget. | Need tighter alignment between SFT data story and live-evaluable RL slice. | Keep as default. |
| Small-budget GRPO pilot | Correct | Good for proving signal exists. | Must stay scoped to supported tasks. | Keep as pilot, not benchmark claim. |
| Deep multi-turn RL this weekend | Weak primary path | Interesting, but coverage and maturity are not there yet. | MARS not implemented, task slice too narrow. | Treat as secondary or future. |
| GEPA / DSPy prompt evolution | Correct as complement | Strong fit for evolving context/prompts without expensive weight updates. | Not yet integrated into one authoritative loop with the evaluator. | Keep as complementary to env + search. |
| JEPA-style context optimization | Reasonable long-term direction | Useful framing for structured latent / prompt-context evolution. | Still high-level in this repo. | Keep as broader recipe, not hackathon proof. |
| SkyDiscover / AdaEvolve / EvoX test-time search | Correct complement | Strong for hard cases and demo quality under budget. | Must use the same evaluator truth and not become a separate fake scoring loop. | Keep as hedge and inference-time compute layer. |
| Rule-based dense process rewards | Worth exploring | Supported by process-supervision literature, and cheaper than learned PRMs. | Current code uses discrete milestones, not dense stepwise reward. | Good v1.5 improvement if time allows. |
| Contrastive fast-vs-slow training pairs | Worth exploring later | CUDA-L1-style ranking can be more sample-efficient than scalar reward alone. | Not wired into the current trainer. | Good post-hackathon experiment. |
| “Same results as CUDA-Agent on one H100/B200” | Not a weekend claim | Current coverage and implementation do not justify it. | Too many unbuilt assumptions. | Keep only as future research target. |

## Additional External Directions Not Yet Fully Exploited

These are the external ideas worth considering beyond the current docs, but they should be ranked below “make the environment authoritative and measurable.”

| External idea | Source | Validity for this setup | Recommendation |
|---|---|---|---|
| Rule-based process supervision for code | Process Supervision-Guided Policy Optimization for Code, StepCoder | High | Good near-term improvement. Use cheap local syntax, compile, API-shape, and flag checks before remote eval. |
| Execution-grounded agent RL | RLEF | High | Reinforces the decision to keep real tool/execution feedback central. |
| Single-step reward approximation for multi-turn code | Multi-Turn Code Generation Through Single-Step Rewards | Medium-high | Supports simplifying the weekend path rather than over-investing in unimplemented per-turn credit machinery. |
| Contrastive CUDA optimization pairs | CUDA-L1 | Medium-high | Good post-hackathon research direction once more positive/negative kernel pairs exist. |
| GEPA prompt evolution over weight updates | GEPA, DSPy | High as complement | Strong match for prompt/context refinement when evaluator calls are expensive. |
| Inference-time evolutionary search | AdaEvolve, SkyDiscover | High as complement | Strong for the hard tail and for demo quality under budget. |

## Concrete Direction For The Next Engineer

### What to build now

1. Make the environment path the unquestioned source of truth.
2. Align code defaults with the hackathon doc defaults, especially the model default.
3. Keep filtering aggressively to supported tasks instead of pretending broader coverage.
4. Tighten documentation so reward, evaluator support, and model choice match the code.
5. Treat SFT + shallow GRPO + search as the weekend stack.

### What to defer

1. MARS
2. CPPO
3. MASPO
4. full Nsight reward shaping
5. full CUDA-Agent parity claims
6. B200 / Coder-Next match-stack positioning

### What to explicitly say in demos and handoff

- The environment is real.
- The evaluator is real.
- The reward is real.
- The current supported training slice is narrow.
- The repo is a credible pilot for data-efficient CUDA RL infra.
- The broader recipe includes GEPA / DSPy / search, but the environment is the central verified substrate.

## Bottom Line

The best single-source-of-truth position is:

KernelForge is currently a **credible hackathon RL environment pilot** with real target-hardware evaluation, real reward, real skill-context injection, and a defensible SFT-first + small-budget GRPO + search posture.

KernelForge is **not yet** a credible claim of CUDA-Agent parity, broad Ops-6K coverage, mature multi-turn credit assignment, or one-GPU reproduction of published large-scale results.

The correct move is not to abandon the current direction. The correct move is to **tighten it**:

- keep the RL environment as the weekend center,
- keep SFT-first,
- keep search and GEPA-style prompt/context evolution as complements,
- remove or quarantine speculative scale-up claims,
- and hand the next engineer a doc set where current truth, stretch goals, and future research are not mixed together.

## Environment Redesign Update (March 8, 2026)

Since this audit was written, the following changes were shipped:

1. **Task pool sampling** — `kernel_forge_env.py` now uses `TaskPool` instead of hardcoded WCC. Samples on `reset()`.
2. **max_turns=3** — down from 200, matching Dr. Kernel MAX_TURN=3.
3. **Anti-hack runtime checks** — 5 Dr. Kernel-inspired checks wired into `eval_core.py`.
4. **Reference code in observations** — model sees PyTorch reference code + interface contract.
5. **`_dispatch()` replaces `_modal()`** — backend-neutral.
6. **Dependency inversion fixed** — `task_routing.py` re-exports from training/.
7. **Benchmark tooling** — `run_benchmark.py` (fast_p metrics) + `compare_results.py` (before/after).
8. **Task pool builder** — `build_task_pool.py` curates Ops-6K from HuggingFace.

**The centralized single-source-of-truth for the environment design is now `docs/SYSTEM_TRUTH.md`.**
