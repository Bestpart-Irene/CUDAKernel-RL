# training/ — 3-Stage RL Pipeline

## GPU Split

> **H200** ($4.54/hr, 141GB HBM3e) = model weights + generation + gradient updates + local nvcc compile checks (fast-fail).
> **A100 (CoreWeave via Northflank)** = all performance reward (speedup timing, execution-based correctness, Nsight profiling). Set `KERNELFORGE_EVAL_BACKEND=modal` for Modal fallback.
> H200 timing ≠ A100 timing. Never use H200 execution for performance reward.

## Model

- **Primary**: unsloth/Qwen3-Coder-30B-A3B-Instruct (30.5B MoE, 3.3B active, bf16 on H200 141GB)
- **Loaded via**: Unsloth `FastLanguageModel` + `PatchFastRL("GRPO")` (Faster MOE 2026 update)
- **Env var**: `KERNELFORGE_MODEL` in `model_loader.py`
- **Note**: bf16 (no quantization) on H200. ~61GB model → ~80GB free for vLLM + GRPO.

## Stage Configs

| Param | Stage 1 (`stage1_warmup.py`) | Stage 2 (`stage2_rft.py`) | Stage 3 (`stage3_grpo.py`) |
|-------|------------------------------|---------------------------|----------------------------|
| **Type** | GRPO warm-up | RFT (SFT on filtered) | TRLOO-augmented GRPO + curriculum **(optimized per GRPO-15)** |
| **LR** | 2e-6 | 5e-6 | 3e-6 |
| **Temperature** | 1.0 | 0.7 (generation) | 0.7 |
| **G (generations)** | 8 (default since 2026-08-10; OOM fallback G=4/accum=4) | — | 2 |
| **Max turns** | 3 | — | **3** (matches Dr. Kernel MAX_TURN=3; env var `KERNELFORGE_MAX_TURNS`) |
| **Steps/Epochs** | 100 steps | 3 epochs | **50 steps** (hackathon pilot; 150 is future target) |
| **Batch** | 1 × 4 grad_accum | 1 × 4 grad_accum | 1 × 4 grad_accum |
| **Optimizer** | paged_adamw_8bit | — | paged_adamw_8bit |
| **Output** | `outputs/kernelforge-stage1` | `outputs/kernelforge-stage2` | `outputs/kernelforge-stage3` |

**Shared**: bf16=True, max_prompt_length=512, max_completion_length=2048 (do-not-repeat 2026-05-18: ≤1024 is a ruled-out family), top_k=50, top_p=0.95, repetition_penalty=1.05, use_vllm=False (disabled for hackathon; set KERNELFORGE_USE_VLLM=1 to enable)

## LoRA Config (`model_loader.py`)

| Param | Value |
|-------|-------|
| r | 16 |
| alpha | 16 |
| dropout | 0 |
| max_seq_length | 8192 |
| quant | bf16 (no quant on H200 141GB) |
| **targets** | q/k/v/o_proj, gate/up/down_proj (Unsloth handles MoE routing) |
| **gradient_checkpointing** | True (not "unsloth" — MoE limitation) |

## Curriculum (`curriculum.py`)

| Phase | Name | Target Reward | Problems |
|-------|------|---------------|----------|
| 0 | `single_ops` | 1.0 | vector_add, relu, softmax, matmul, gelu |
| 1 | `fusion_2op` | 2.0 | LayerNorm+ReLU, MatMul+BiasAdd, Softmax+Dropout, Conv2d+BatchNorm |
| 2 | `arch_specific` | 2.0 | WCC, reduce_sum, GEMM, attention |
| 3 | `advanced` | 3.0 | LayerNorm+GELU+Linear, Flash-Attention, Batched SpMV |

- **Window**: 10 rewards
- **Promote**: >50% hit target reward
- **Demote**: <20% positive

### Key class: `CurriculumManager`
- `get_problem() -> dict` — next problem from current phase
- `record_reward(reward) -> str | None` — returns event ("promoted"/"demoted") or None
- `status() -> dict` — phase index, name, history, rates

## Multi-Turn Rollout (`multi_turn_rollout.py`)

### `make_multi_turn_rollout(max_turns=3, skill_md_gpu=None) -> Callable`
Returns TRL-compatible `rollout_func(prompt_ids, ...) -> dict` with keys:
`prompt_ids`, `completion_ids`, `logprobs`, `env_reward`. Stage 3 uses `max_turns=3-5` per GRPO-15.1 hackathon config.

### Flow per completion:
1. Generate response on H200 → `extract_cuda_code(text)` extracts ```cuda blocks
2. **`_local_compile_check(code)`** — nvcc -arch=sm_80 -c syntax check (**IMPLEMENTED**, saves ~50% eval cost)
3. If compiles locally → `evaluate_code_remote(code, task_row)` — **A100 execution via eval_backend:** routes to `evaluate_ops6k_kernel` (Ops-6K) or `evaluate_kernel` (WCC)
4. `_compute_reward_from_result(result)` → discrete milestone {-1, 1, 2, 3}
5. `_format_feedback(result, reward, turn)` → feedback text for next turn
6. **Early exit** at reward >= 3.0 (beats torch.compile)

### `reward_from_env(completions, **kwargs) -> list[float]`
Single-turn wrapper — extracts env_reward from rollout kwargs.

## RFT Filter (`rft_filter.py`)

### `TrajectoryCollector(modal_app_name, model_path)`
Legacy constructor name is retained for compatibility, but evaluation now routes through the shared backend adapter rather than directly through Modal.
- `collect_trajectories(num_trajectories=100) -> list[dict]`
- `filter_trajectories(min_reward=0.0) -> list[dict]` (revised per GRPO-15: any speedup is useful signal)
- `save_rft_dataset(filtered, output_path)` → HF messages format
- Generation: max_new_tokens=2048, temperature=0.7, top_p=0.95

## TRLOO Custom Trainer (`custom_grpo_trainer.py`) — INACTIVE under TRL 0.29

`TRLOOGRPOTrainer(GRPOTrainer)` — the N/(N-1) correction CANNOT run under TRL 0.29: the `_compute_advantages` hook does not exist (advantages are computed inline), and the old gate only fired for `loss_type=="grpo"` while the default is `"dapo"`. The trainer now emits a loud RuntimeWarning at init under every loss_type when the correction is inactive (2026-08-10 audit U2). The real small-G mitigation is G=8 (now the Stage-1 default). Do not cite TRLOO as an active mitigation.

Also provides factory: `create_trloo_trainer(model, tokenizer, reward_funcs, train_dataset, config, rollout_func)`.

## CUDA Agent Integration (`cuda_agent_integration.py`)

- Dataset: `BytedTsinghua-SIA/CUDA-Agent-Ops-6K`
- `load_cuda_agent_prompt_dataset(max_samples=1024, seed=42) -> Dataset`
  - Returns columns: `prompt`, `task_code`, `ops`, `data_source`
  - `task_code` carries reference PyTorch model for `evaluate_ops6k_kernel`
- `load_cuda_agent_prompt_texts(max_samples=128, seed=42) -> list[str]`
- `_build_cuda_prompt(example, max_code_chars=6000) -> str | None`

## Unified Dataset Pipeline (`datasets/build_combined_dataset.py`, `training/dataset_loader.py`)

- **Builder**: `datasets/build_combined_dataset.py`
  - Merges doubleGraph manifest tasks + Ops-6K tasks into one problem schema
  - Output: `datasets/combined_kernelforge.jsonl`
  - Main APIs:
    - `build_combined_dataset(...) -> list[dict]`
    - `inject_into_curriculum(cm, dataset) -> dict[str, int]`
- **Loader**: `training/dataset_loader.py`
  - `load_training_dataset(stage="stage1"|"stage2"|"stage3", ...)`
  - Stage 1: prompt Dataset (easy Ops-6K + doubleGraph base variants)
  - Stage 2: SFT JSONL rows from `datasets/doublegraph_sft.jsonl`
  - Stage 3: full combined rows; optional `CurriculumManager` injection

## doubleGraph A100 Expert Demonstrations (`datasets/`)

Source-of-truth metadata feed:
- `docs/research/doublegraph/doublegraph_a100_manifest.jsonl` (192 kernels)

Derived training artifacts:
- `doublegraph_sft.jsonl` — HF messages format for Stage 2 SFT (192 entries)
- `combined_kernelforge.jsonl` — merged Ops-6K + doubleGraph curriculum pool (224 entries: 192 doubleGraph + 32 ops_local)
- Legacy files (`doublegraph_a100_kernels.jsonl`, `doublegraph_grpo_prompts.jsonl`) archived to `archive/datasets_legacy/`

Categories: traversal, components, community, centrality, link_analysis, link_prediction, cores, tree.
Topology-aware prompts: power-law, sparse-islands, dense-regular, dense-community, bipartite.
Per-kernel compilation flags included in training signal (--maxrregcount, --rdc, etc.).

Harvester: `python3 datasets/extract_doublegraph_a100.py`

## Evolutionary Search — AdaEvolve + EvoX (`skydiscover_integration/`)

Implements SkyDiscover's algorithms natively (no external dependency).
- `evaluator.py`: `KernelForgeEvaluator` — cascade eval (stage1 local compile → stage2 remote A100 via `eval_backend`)
- `adaevolve.py`: `AdaEvolve` — multi-island evolutionary search with UCB1 scheduling
- `evox_strategies.py`: `EvoXStrategyManager` — self-evolving mutation strategies (LogWindowScorer)
- Seed kernels in `initial_kernels/` (5 A100 graph kernels from doubleGraph)
- Launch: `./skydiscover_integration/run_evolution.sh`

## Stacked Mitigations

| Technique | Status | Location |
|-----------|--------|----------|
| TRLOO advantage scaling (N/(N-1)) | **INACTIVE** (no TRL 0.29 hook; loud warning at init; use G=8 instead) | `custom_grpo_trainer.py` |
| Local compile fast-path | **DONE** | `multi_turn_rollout.py:_local_compile_check()` |
| Ops-6K evaluation | **DONE** | `eval_service/eval_core.py:evaluate_ops6k_kernel_impl()` via `eval_backend` |
| Discrete reward {-1,1,2,3} | **DONE** | `reward.py:compute_reward()` |
| Nsight lightweight profiling | **DONE** | `eval_service/eval_core.py:evaluate_kernel_impl()` (with `modal_app.py` as fallback wrapper) |
| doubleGraph A100 expert dataset | **DONE** | `datasets/doublegraph_*.jsonl` (192 entries) |
| Real A100 patterns in SKILL.md | **DONE** | `skill_builder.py:_append_a100_patterns()` |
| Topology-aware curriculum | **DONE** | `curriculum.py` Phase 2-3 (5 graph problems with graph_properties) |
| Graph topology in RL observations | **DONE** | `kernel_forge_env.py`, `curriculum.py`, `multi_turn_rollout.py` |
| Task pool sampling on reset() | **DONE** | `openenv_env/task_pool.py`, `kernel_forge_env.py` |
| Anti-hack runtime checks (Dr. Kernel) | **DONE** | `openenv_env/anti_hack.py` (5 checks), wired into `eval_core.py` |
| Benchmark runner (fast_p metrics) | **DONE** | `scripts/run_benchmark.py` |
| Before/after comparison | **DONE** | `scripts/compare_results.py` |
| Task pool builder (Ops-6K curation) | **DONE** | `tasks/build_task_pool.py` |
| Evaluator bridge (cascade eval) | **DONE** | `skydiscover_integration/evaluator.py` |
| AdaEvolve multi-island search | **DONE** | `skydiscover_integration/adaevolve.py` |
| EvoX self-evolving strategies | **DONE** | `skydiscover_integration/evox_strategies.py` |
| MARS return-to-go credit | **DROPPED** (degenerates with outcome rewards) | — (GRPO-4) |
| CPPO completion pruning | **DROPPED** (harmful for exploration at G=2) | — (GRPO-11) |
| MASPO soft trust region | DEFERRED (future research) | — (GRPO-12) |
| ~~Transformation grammar~~ | DEFERRED | v2 (GRPO-13 line 1849) |

## Abort Conditions

- reward = -1 consistently after 30 steps → dataset/prompt issue
- reward std = 0 → model collapsed
- NaN loss → LR too high or gradient explosion
- OOM → reduce G, batch, or seq_length
- Gate G-0.8 failure (non-zero gradients + >1.2x on 2/20 kernels) → abandon GRPO, use SFT + AdaEvolve only

## Deep Dive Pointers

- Quick reference card: `GRPO_DEEP_DIVE.md` line 1573 (GRPO-8)
- Full stacked architecture: `GRPO_DEEP_DIVE.md` line 1978 (GRPO-14)
- Monitor checklist: `GRPO_DEEP_DIVE.md` line 1446 (GRPO-6)


<claude-mem-context>

</claude-mem-context>