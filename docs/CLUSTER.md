# KernelForge on Northeastern Explorer

Runbook for running KernelForge on the Northeastern Explorer / EL9 cluster
(`rc-docs.northeastern.edu/en/explorer-main/`).

This project shares the cluster with `Reinforcement-Casino`. The two projects
**must not** share environments, scratch namespaces, or job-name patterns.
This doc is the contract that keeps them apart.

## Cost Posture (read this first)

Discovery is free for NU netids. Modal and CoreWeave/Northflank cost real
money. Default cluster posture for this project is **zero external cost**:

- Training and reward-bearing eval run inside the **same slurm allocation** on
  one A100 node.
- `KERNELFORGE_EVAL_BACKEND=local` is exported by `cluster_paths.sh` — this
  routes `dispatch_eval(...)` to `eval_service.eval_core` in-process instead
  of HTTP / Modal. See [openenv_env/eval_backend.py](../openenv_env/eval_backend.py).
- `modal_train.py` / `modal_app.py` / `eval_service/app.py` (FastAPI) are
  preserved but **dormant on this cluster path**. Do not deploy them unless
  you have an explicit budget reason.

When you outgrow one node (multi-GPU, larger replicas), the migration path is
either Discovery's `multigpu` partition (still free, needs reservation) or
re-enabling the Modal / CoreWeave backends. Both swaps are a single env-var
flip.

## Identity

| | Value |
|--|--|
| Cluster | Northeastern Explorer (EL9) |
| Netid | `xie.yiyi` (override with `KF_NETID`) |
| Login host | `login.discovery.neu.edu` (override with `KF_DISCOVERY_HOST`; confirm against current NU RC docs) |
| Shared conda base | `/shared/EL9/explorer/miniconda3/24.11.1/miniconda3` (not used by KernelForge — kept here for reference) |

## Path Layout (strict isolation)

```
/home/xie.yiyi/CUDAKernel-RL/                  <-- repo (mirrors local repo)
/scratch/xie.yiyi/kernelforge/                 <-- all KernelForge writes live here
  ├── .venv/                                   <-- uv-managed Python env
  ├── uv/bin/uv                                <-- uv binary (avoids $HOME quota)
  ├── hf_cache/                                <-- HF_HOME, HF_HUB_CACHE, HF_DATASETS_CACHE
  ├── checkpoints/                             <-- training outputs
  ├── eval_artifacts/                          <-- compiled kernels from eval
  ├── logs/                                    <-- runtime logs (slurm output is in repo logs/)
  ├── torch_cache/                             <-- TORCH_HOME
  ├── triton_cache/                            <-- TRITON_CACHE_DIR
  ├── pip_cache/                               <-- PIP_CACHE_DIR
  └── uv_cache/                                <-- UV_CACHE_DIR
```

For comparison, `Reinforcement-Casino` (DO NOT TOUCH from this project) owns:
- `/scratch/xie.yiyi/conda_envs/rl_casino`
- `/scratch/xie.yiyi/conda_envs/rl_casino_eval`
- `/scratch/xie.yiyi/rl_casino_outputs/`
- `/scratch/xie.yiyi/hf_cache/` (rl-casino's own HF cache root)

The single source of truth for KernelForge paths is
[`scripts/cluster/cluster_paths.sh`](../scripts/cluster/cluster_paths.sh).
That script has a hard-fail collision check that aborts if any KernelForge
variable accidentally points into rl-casino's namespace.

## Job Naming

Every KernelForge slurm job name starts with `kf_*`:

- `kf_setup` — one-time env install
- `kf_smoke` — env health check
- `kf_stage1`, `kf_stage2`, `kf_stage3` — training stages
- `kf_eval`, `kf_bench` — evaluation / benchmarking

This way `squeue -u $USER` lets you tell KernelForge jobs from rl-casino jobs
at a glance.

## Partitions

- `gpu` — default GPU partition, A100 via `--gres=gpu:a100:1`
- `multigpu` — multi-GPU jobs, needs RC reservation request
- `short` — CPU batch (there is NO `cpu` partition on Explorer)

## Tooling: uv (not conda)

KernelForge uses `uv` against the existing `pyproject.toml` + `uv.lock`.
`requirements.txt` is preserved as a fallback only.

We deliberately do not use the shared conda installation. Reasons:

- `uv.lock` is the only artifact that captures the exact dep graph that
  passes locally. Re-solving via conda+pip would discard it.
- The torch version is pulled by `trl[vllm]`. uv handles this fine.
- uv installs faster and to a known scratch path; no `$HOME` quota pressure.

## First-Time Setup

On your laptop:

```bash
# 1. Push the repo
bash scripts/cluster/sync_to_discovery.sh
```

On the cluster (after the rsync completes):

```bash
ssh xie.yiyi@login.discovery.neu.edu
cd /home/$USER/CUDAKernel-RL

# 2. Install the env (one-time, ~10–20 min depending on wheel cache)
sbatch scripts/cluster/setup_env.slurm

# 3. Watch the log
tail -f logs/kf_setup_*.out

# 4. Once setup_env finishes, run the smoke job
sbatch scripts/cluster/smoke.slurm

# 5. Watch
tail -f logs/kf_smoke_*.out
```

## Day-to-Day Workflow

```bash
# On your laptop: push changes
bash scripts/cluster/sync_to_discovery.sh

# On the cluster: submit jobs
ssh xie.yiyi@login.discovery.neu.edu
cd ~/CUDAKernel-RL
sbatch scripts/cluster/smoke.slurm
# ... or whichever stage script you've built next

# Inspect
squeue -u $USER -o "%.10i %.20j %.8T %.10M %R"
tail -f logs/kf_*_${SLURM_JOB_ID}.out
```

## Activating the Env in an Interactive Shell

```bash
# After ssh'ing to the cluster:
cd ~/CUDAKernel-RL
source scripts/cluster/cluster_paths.sh
source "${KF_VENV}/bin/activate"

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Always `source cluster_paths.sh` first so `HF_HOME`, `TORCH_HOME`,
`TRITON_CACHE_DIR`, etc. all point at `/scratch`, not `/home`.

## Secrets

Login is via NU SSO + Duo. The cluster does not store API keys. For runs
that need them:

```bash
# On the cluster, one-time per shell or per slurm script:
hf auth login      # HuggingFace
wandb login        # Weights & Biases
```

Or set via env vars:

```bash
export HF_TOKEN=...
export WANDB_API_KEY=...
```

Do **not** commit these to the repo.

## When You Edit `pyproject.toml`

`uv.lock` is the source of truth on the cluster. After changing deps locally:

```bash
# locally:
uv lock                    # regenerate uv.lock
git add pyproject.toml uv.lock
git commit -m "deps: ..."

# push to cluster:
bash scripts/cluster/sync_to_discovery.sh

# on the cluster (in an interactive session or a small slurm job):
cd ~/CUDAKernel-RL
source scripts/cluster/cluster_paths.sh
uv sync --frozen
```

## Failure Modes & Fallbacks

| Symptom | Likely cause | Fix |
|---|---|---|
| `setup_env.slurm` fails on `uv sync` with a torch wheel mismatch | Wheel for current CUDA/python combo missing | Re-run the job; if persistent, pin `torch` explicitly in `pyproject.toml` and `uv lock` again |
| `smoke.slurm` says `venv not found` | `setup_env.slurm` did not finish or failed silently | Re-submit `setup_env.slurm`, check `logs/kf_setup_*.err` |
| `nvidia-smi` works but `torch.cuda.is_available() == False` | torch was installed without CUDA support (CPU wheel) | Verify `pyproject.toml` did not regress to a CPU-only torch; re-run `uv sync` |
| Out of disk in `$HOME` | uv or HF cache leaked into `$HOME` | Confirm `cluster_paths.sh` was sourced before any python work — that's what redirects caches to `/scratch` |
| Collision-check abort on job start | A `KF_*` variable was overridden to point into rl-casino | Unset the variable; check your shell rc |
| `RuntimeError: KERNELFORGE_EVAL_URL must be set ...` at training start | Something reset `KERNELFORGE_EVAL_BACKEND` to `coreweave` (the legacy default) | Re-source `cluster_paths.sh`; check the slurm script does not unset the var |
| `evaluate_kernel_impl` ImportError on training start | The slurm node has no nvcc or torch.cuda | Confirm `--gres=gpu:a100:1` is in your `#SBATCH` directives; rerun on a GPU node |
| Baselines recomputed every job, slowing eval | `TaskPool` baseline cache lives on `/scratch`, but a new physical A100 node has slightly different timing | Acceptable noise; warm-start by re-using the same node id when possible, or commit cached baselines to `research/live/baselines.json` |

## Why a Separate Path Doc

The single biggest source of cross-project bugs on shared HPC is sloppy
namespacing: one project's run mutates the other's checkpoint or HF cache.
This doc + `cluster_paths.sh` enforce strict separation by construction so
neither project depends on the other to behave.
