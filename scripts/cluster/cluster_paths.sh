# shellcheck shell=bash
# KernelForge cluster path contract (Northeastern Explorer / EL9).
#
# Single source of truth for all paths used by slurm scripts. Every other
# scripts/cluster/*.slurm sources this file. Override any variable via
# environment if you are running on a different netid or layout.
#
# Strict-isolation contract: NEVER touch rl-casino paths. The rl-casino
# project owns /scratch/${KF_NETID}/conda_envs/rl_casino*,
# /scratch/${KF_NETID}/rl_casino_outputs, and /scratch/${KF_NETID}/hf_cache.
# KernelForge writes under /scratch/${KF_NETID}/kernelforge ONLY.

set -euo pipefail

# --- Identity --------------------------------------------------------------
# Default to the value the rl-casino setup uses; override if different.
KF_NETID="${KF_NETID:-${USER:-xie.yiyi}}"

# --- Repo location on cluster ---------------------------------------------
KF_HOME_REPO="${KF_HOME_REPO:-/home/${KF_NETID}/CUDAKernel-RL}"

# --- Scratch namespace (everything KernelForge writes lives here) ---------
KF_SCRATCH_ROOT="${KF_SCRATCH_ROOT:-/scratch/${KF_NETID}/kernelforge}"
KF_VENV="${KF_VENV:-${KF_SCRATCH_ROOT}/.venv}"
KF_UV_INSTALL_DIR="${KF_UV_INSTALL_DIR:-${KF_SCRATCH_ROOT}/uv}"
KF_HF_HOME="${KF_HF_HOME:-${KF_SCRATCH_ROOT}/hf_cache}"
KF_HF_HUB_CACHE="${KF_HF_HUB_CACHE:-${KF_HF_HOME}/hub}"
KF_HF_DATASETS_CACHE="${KF_HF_DATASETS_CACHE:-${KF_HF_HOME}/datasets}"
KF_CHECKPOINTS="${KF_CHECKPOINTS:-${KF_SCRATCH_ROOT}/checkpoints}"
KF_LOGS="${KF_LOGS:-${KF_SCRATCH_ROOT}/logs}"
KF_EVAL_ARTIFACTS="${KF_EVAL_ARTIFACTS:-${KF_SCRATCH_ROOT}/eval_artifacts}"

# --- CUDA toolkit (nvcc) on PATH for compile / eval --------------------------
# Slurm shell does not inherit login-shell module state, and our previous
# slurm scripts never loaded the cuda module, so eval_service/eval_core.py
# subprocess.run(["nvcc", ...]) failed with FileNotFoundError. EXP-004
# debug (slurm 6874203) caught this — _local_compile_check was silently
# swallowing the same FileNotFoundError as "success", masking the bug for
# every GRPO run in EXP-001..EXP-003. Load the module here so every slurm
# job that sources cluster_paths.sh gets nvcc on PATH and CUDA_HOME set.
KF_CUDA_MODULE="${KF_CUDA_MODULE:-cuda/12.8.0}"
if command -v module >/dev/null 2>&1; then
    module load "${KF_CUDA_MODULE}" 2>/dev/null || true
fi
# Defensive fallback if `module` is unavailable (e.g. interactive non-Lmod shell).
export CUDA_HOME="${CUDA_HOME:-/shared/EL9/explorer/cuda/12.8.0}"
export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
case ":${PATH}:" in
    *":${CUDA_HOME}/bin:"*) ;;
    *) export PATH="${CUDA_HOME}/bin:${PATH}" ;;
esac
case ":${LD_LIBRARY_PATH:-}:" in
    *":${CUDA_HOME}/lib64:"*) ;;
    *)
        # Only append ':'$LD_LIBRARY_PATH when non-empty — a trailing ':'
        # is an empty component, which the linker treats as CWD.
        if [ -n "${LD_LIBRARY_PATH:-}" ]; then
            export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
        else
            export LD_LIBRARY_PATH="${CUDA_HOME}/lib64"
        fi
        ;;
esac

# --- WandB / job naming ---------------------------------------------------
KF_WANDB_PROJECT="${KF_WANDB_PROJECT:-kernelforge}"
# Slurm job-name prefix; every job in this project starts with kf_*
KF_JOB_PREFIX="kf"

# --- Eval backend ---------------------------------------------------------
# On Discovery / Explorer we run training AND eval on the same slurm
# allocation (single A100 node). That means eval_service.eval_core is
# imported in-process, no external HTTP / Modal call.
# Override with KERNELFORGE_EVAL_BACKEND=coreweave (and KERNELFORGE_EVAL_URL)
# if you want to dispatch to an external eval host.
export KERNELFORGE_EVAL_BACKEND="${KERNELFORGE_EVAL_BACKEND:-local}"

# --- Export everything ----------------------------------------------------
export KF_NETID KF_HOME_REPO KF_SCRATCH_ROOT KF_VENV KF_UV_INSTALL_DIR
export KF_HF_HOME KF_HF_HUB_CACHE KF_HF_DATASETS_CACHE
export KF_CHECKPOINTS KF_LOGS KF_EVAL_ARTIFACTS
export KF_WANDB_PROJECT KF_JOB_PREFIX

# HuggingFace + Torch caches redirected to scratch (avoid $HOME quota)
export HF_HOME="${KF_HF_HOME}"
export HF_HUB_CACHE="${KF_HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${KF_HF_DATASETS_CACHE}"
export TRANSFORMERS_CACHE="${KF_HF_HUB_CACHE}"
export TORCH_HOME="${KF_SCRATCH_ROOT}/torch_cache"
export TRITON_CACHE_DIR="${KF_SCRATCH_ROOT}/triton_cache"
export PIP_CACHE_DIR="${KF_SCRATCH_ROOT}/pip_cache"
export UV_CACHE_DIR="${KF_SCRATCH_ROOT}/uv_cache"
export UV_PROJECT_ENVIRONMENT="${KF_VENV}"

# Put scratch-installed uv on PATH so `uv` resolves without using $HOME
export PATH="${KF_UV_INSTALL_DIR}/bin:${PATH}"

# --- Sanity: collision check vs rl-casino ---------------------------------
# Fail loudly if any KernelForge path accidentally points into rl-casino.
case "${KF_SCRATCH_ROOT}:${KF_VENV}:${KF_CHECKPOINTS}" in
  *rl_casino*|*rl-casino*|*Reinforcement-Casino*)
    echo "[cluster_paths.sh] FATAL: KernelForge path collides with rl-casino namespace" >&2
    echo "  KF_SCRATCH_ROOT=${KF_SCRATCH_ROOT}" >&2
    echo "  KF_VENV=${KF_VENV}" >&2
    exit 99
    ;;
esac

# --- Ensure scratch dirs exist (cheap; idempotent) ------------------------
mkdir -p \
  "${KF_SCRATCH_ROOT}" \
  "${KF_HF_HUB_CACHE}" \
  "${KF_HF_DATASETS_CACHE}" \
  "${KF_CHECKPOINTS}" \
  "${KF_LOGS}" \
  "${KF_EVAL_ARTIFACTS}" \
  "${TORCH_HOME}" \
  "${TRITON_CACHE_DIR}" \
  "${PIP_CACHE_DIR}" \
  "${UV_CACHE_DIR}"
