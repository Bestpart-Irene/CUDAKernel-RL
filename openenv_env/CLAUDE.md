# openenv_env/ — OpenEnv Environment

## GPU Split

> All performance reward (speedup, correctness) executes on **A100 via CoreWeave (Northflank)**. Set `KERNELFORGE_EVAL_BACKEND=modal` to use Modal instead.
> **H200** ($4.54/hr) for training + local compile checks only.
> Judges call `step()` first — cache model on startup, warm CUDA context, pre-load tokenizer.

## OpenEnv Protocol

- **NOT Gymnasium**. HTTP client-server protocol.
- Install: `uv add "openenv-core[core]>=0.2.1"`
- Server: FastAPI app created via `create_app(KernelForgeEnv, KernelForgeAction, KernelForgeObservation, env_name="kernelforge")` in `openenv_env/server/app.py`

## Upstream Dataset Contract (training -> env)

The environment now receives prompts originating from a unified dataset pipeline:
- `datasets/build_combined_dataset.py`
- `training/dataset_loader.py`

Unified problem fields expected upstream:
- `prompt`, `ops`, `difficulty`, `data_source`
- optional: `task_code`, `topology`, `graph_properties`, `kernel_id`, `compile_flags`

This keeps OpenEnv episode payloads consistent across Ops-6K dense ops and doubleGraph topology-aware tasks.

## KernelForgeEnv (`kernel_forge_env.py`)

### Types

```python
class KernelForgeAction(Action):
    cuda_code: str

class KernelForgeObservation(Observation):
    text: str
    baseline_original_ms: float | None = None
    baseline_doublegraph_ms: float | None = None
    hardware: dict[str, Any]
    turn: int = 0
    best_reward: float = -1.0
    info: dict[str, Any]
    graph_properties: dict[str, Any] | None = None  # Topology context (degree dist, density, etc.)
    topology_type: str | None = None  # "power-law", "sparse-islands", "dense-regular", etc.
```

### Constructor
```python
KernelForgeEnv(task_pool: TaskPool | None = None)
```
- `self.task_pool` — loaded from `TaskPool.load()` if not provided (fallback: combined_kernelforge.jsonl → builtin ELU task)
- `self.target_gpu` from env `KERNELFORGE_TARGET_GPU` (default "a100")
- `self.max_turns = int(os.getenv("KERNELFORGE_MAX_TURNS", "3"))` — matches Dr. Kernel MAX_TURN=3
- remote evaluation dispatch via `openenv_env/eval_backend.py` using `_dispatch()` method

### Contract
```python
def reset(self, seed=None, episode_id=None, task_id=None, **kwargs) -> KernelForgeObservation
def step(self, action: KernelForgeAction, timeout_s=None, **kwargs) -> KernelForgeObservation
```

`reset()` samples a task from `TaskPool` (or uses specific `task_id`), builds observation with SKILL.md + reference code + interface contract.

`step()` builds a task-specific payload via `training/task_support.py` and dispatches through `_dispatch()` → `openenv_env/eval_backend.py` to the active backend. Default path: CoreWeave/Northflank HTTP service. Fallback path: Modal wrapper.

### Task Pool (`task_pool.py`)

```python
class TaskPool:
    @classmethod
    def load(cls, pool_path=None) -> TaskPool   # Fallback chain: pool_v0.jsonl → combined → builtin
    def sample(self, task_id=None, seed=None, backend=None) -> dict
    def get_cached_baselines(self, task_id) -> dict | None
    def cache_baselines(self, task_id, baselines) -> None
```

### Task Routing (`task_routing.py`)

Re-exports from `training/task_support.py` to fix dependency inversion (env should not import training). Exports: `build_modal_payload`, `normalize_eval_result`, `normalize_task_row`, `task_interface_contract`.

## Reward (`reward.py`)

```python
def validate_eval_result(result: dict) -> dict  # Assert evaluator contract keys, clamp NaN/inf, safe defaults

def compute_reward(compiled, correct, speedup_vs_eager, speedup_vs_compile,
                   occupancy=None, mem_coalescing=None, warp_efficiency=None) -> float
```

Reward shape is selected at module-load time by `KERNELFORGE_REWARD_VERSION`. Default `v3-symbol-shaped` (EXP-010). Always record the active version in `research/live/master.json` on every promotion — rows under different versions are **not comparable**.

| Return | v3-symbol-shaped (default, EXP-010) | v2-shaped (EXP-005) | v1-discrete-milestone (legacy) |
|--------|-------------------------------------|---------------------|--------------------------------|
| -1.0 | not compiled | not compiled | not compiled OR compiled-but-wrong |
| -0.5 | compiled, **symbol missing** (`undefined symbol` from verifier dlsym) | (n/a) | (n/a) |
| 0.0 | compiled, symbol OK, numerically wrong | compiled but wrong (any reason) | (never returned) |
| 1.0 | correct, not faster than baselines | correct, not faster | correct, not faster |
| 2.0 | correct, > eager PyTorch (>5%) | correct, > eager | correct, > eager |
| 3.0 | correct, > torch.compile (>5%) | correct, > torch.compile | correct, > torch.compile |

**Version history**:
- v1-discrete-milestone (initial): binary success / fail. Caused grad_norm=0 collapses on compile-fail-dominated groups (EXP-004 v3).
- v2-shaped (EXP-005): split compile-fail from compiled-but-wrong so GRPO has variance. Confirmed unlocks signal but cannot push the model toward the verifier contract — both "symbol missing" and "numerically wrong" gave the same 0.0 (job 6920683, 11 GRPO steps, 100% of compile=True hit `undefined symbol: wcc_kernel`).
- v3-symbol-shaped (EXP-010, current default): adds a -0.5 bucket between -1 and 0 specifically for "compiled cleanly but verifier could not dlsym the canonical entry". Gives GRPO a directional gradient toward `extern "C" void wcc_kernel(...)` without any prompt edits or post-extract bypass.

`symbol_loaded` is detected in `training/task_support._detect_symbol_loaded` by substring-matching the eval result's `verifier_msg` for `"undefined symbol"` AND `"FFI verification failed"`. v1 / v2 ignore this argument.

```python
def trloo_post_process(advantages: list[float], n: int) -> list[float]
```
Scales GRPO advantages by N/(N-1) to correct Dr. Kernel gradient shrinkage.

## Nsight Integration

Nsight metrics (occupancy, mem_coalescing, warp_efficiency) are accepted as optional kwargs in `compute_reward()` for API compatibility but are **unused in discrete reward mode**. The discrete milestone scheme {-1, 1, 2, 3} does not incorporate profiling bonuses — it rewards based on correctness and speedup tiers only.

## Anti-Hack (`anti_hack.py`)

### Static Checks

#### `extract_cu_flags(cuda_code) -> list[str]`
Extracts from `// CU_FLAGS: ...` comment lines.

**Allowed flags**: `--use_fast_math`, `--extra-device-vectorization`, `--rdc=true`, `--maxrregcount=*`

#### `scan_forbidden_symbols(so_path) -> str | None`
Uses `nm -D` to scan compiled .so. Returns failure reason or None.

**Forbidden symbols**: `torch`, `at::Tensor`, `c10::`, `torch::autograd`, `triton`, `torch.compile`, `torch.nn.functional`

### Runtime Anti-Hack Checks (Dr. Kernel-inspired)

These are wired into `eval_service/eval_core.py:evaluate_ops6k_kernel_impl()` lines 732-773.

| Function | What It Catches |
|----------|----------------|
| `check_shapes_match(candidate, reference)` | Output tensor count/shape mismatch |
| `check_output_not_constant(output1, output2)` | Decoy kernel returning hardcoded values regardless of input |
| `check_not_passthrough(output, inputs)` | Kernel returning input unchanged (identity function) |
| `check_not_noop(runtime_ms)` | Suspiciously fast kernel (< 1μs) — skips computation |
| `run_anti_hack_suite(...)` | Orchestrates all checks in sequence |

## GPU Registry (`gpu_registry.py`)

### `get_gpu_spec(gpu_name) -> dict[str, Any]`

| Key | A100 | H100 | B200 |
|-----|------|------|------|
| arch | sm_80 | sm_90a | sm_100a |
| SMs | 108 | 132 | 192 |
| L2 cache | 40 MB | 50 MB | 96 MB |
| SMEM/SM | 164 KB | 228 KB | 228 KB |
| HBM BW | 2.0 TB/s | 3.35 TB/s | 8.0 TB/s |
| Regs/SM | 65,536 | 65,536 | 65,536 |
| TMA | no | yes | yes |
| DSMEM | no | yes | yes |
| async_copy | yes | yes | yes |

**A100 features**: cp.async, l2_persistence, bf16_tensor_core, tf32_tensor_core, structured_sparsity_2_4

## Skill Builder (`skill_builder.py`)

### `build_skill_md(gpu_name="a100") -> str`
Priority chain:
1. Env var `KERNELFORGE_SKILL_FILE` (file path)
2. Static file `skill_{gpu_name}.md` (from project root)
3. `_generate_skill_md(gpu_name)` — dynamic from GPU_REGISTRY

### Generated SKILL.md priorities:
1. Algorithmic reduction (>50% impact)
2. Memory hierarchy (20-50%) — arch-specific patterns
3. Compute optimization (10-20%) — warp primitives, TF32
4. Library integration — cuBLAS, cuDNN, cuSPARSE

## CachePool (`cache_pool.py`)

```python
class GPUCachePool:
    def __init__(self, max_entries: int = 8)
    def get_or_create(self, key: str, factory: Callable[[], Any], metadata=None) -> Any
    def get(self, key: str, default=None) -> Any
    def clear(self) -> None
```

LRU eviction. Calls `close()` or `release()` on evicted entries.

Entry type: `GPUCacheEntry(key: str, value: Any, metadata: dict)`

## Nsight Integration — API COMPAT ONLY

Nsight metric kwargs (occupancy, mem_coalescing, warp_efficiency) are accepted by `compute_reward()` but **unused in discrete reward mode**. Discrete milestones normalize across problem difficulty without profiling bonuses. No separate `reward_nsight.py` needed.

## Transform Grammar — DEFERRED TO v2

Transformation grammar (12-40 rules) was planned but deferred. No production system has shipped this for CUDA kernels as of March 2026. For v1, use CUDA-Agent's SKILL.md verbatim + doubleGraph pattern paste.

See GRPO-13, `GRPO_DEEP_DIVE.md` line 1849 for reference design.

## Deep Dive Pointers

- Nsight rewards: `GRPO_DEEP_DIVE.md` line 1623 (GRPO-9)
- Transform grammar (deferred): `GRPO_DEEP_DIVE.md` line 1849 (GRPO-13)
- Reward function spec: `GRPO_DEEP_DIVE.md` line 557 (GRPO-3, Sec 3.2)


<claude-mem-context>

</claude-mem-context>