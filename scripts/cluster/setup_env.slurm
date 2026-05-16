#!/bin/bash
# One-time KernelForge environment setup on Northeastern Explorer.
#
# Submit from the repo root on the cluster:
#   cd /home/$USER/CUDAKernel-RL
#   sbatch scripts/cluster/setup_env.slurm
#
# What this does:
#   - sources scripts/cluster/cluster_paths.sh
#   - installs uv into /scratch/$USER/kernelforge/uv (NOT into $HOME — quota)
#   - runs `uv sync --frozen` against pyproject.toml + uv.lock
#   - verifies torch + CUDA on an A100 node
#
# Strict isolation: every path written below lives under
#   /scratch/$USER/kernelforge/...
# This script will refuse to run if it would touch rl-casino paths.

#SBATCH --job-name=kf_setup
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/kf_setup_%j.out
#SBATCH --error=logs/kf_setup_%j.err

set -euo pipefail

# --- Locate repo + load path contract ---
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/cluster/cluster_paths.sh"

mkdir -p "${REPO_ROOT}/logs"  # local logs/ (slurm output target above)

echo "========================================"
echo "kf_setup job_id : ${SLURM_JOB_ID:-local}"
echo "node            : ${SLURMD_NODENAME:-$(hostname)}"
echo "started         : $(date)"
echo "repo root       : ${REPO_ROOT}"
echo "venv target     : ${KF_VENV}"
echo "uv install dir  : ${KF_UV_INSTALL_DIR}"
echo "scratch root    : ${KF_SCRATCH_ROOT}"
echo "========================================"

# --- Install uv into /scratch (avoids $HOME quota) ---
if [ ! -x "${KF_UV_INSTALL_DIR}/bin/uv" ]; then
  echo ""
  echo "=== Installing uv into ${KF_UV_INSTALL_DIR} ==="
  mkdir -p "${KF_UV_INSTALL_DIR}/bin"
  # The official installer respects UV_INSTALL_DIR
  export UV_INSTALL_DIR="${KF_UV_INSTALL_DIR}/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  hash -r
else
  echo "uv already present at ${KF_UV_INSTALL_DIR}/bin/uv"
fi

echo "uv version: $("${KF_UV_INSTALL_DIR}/bin/uv" --version)"

# --- Sync the project ---
echo ""
echo "=== uv sync (frozen against uv.lock) ==="
# UV_PROJECT_ENVIRONMENT (exported by cluster_paths.sh) points uv at /scratch
"${KF_UV_INSTALL_DIR}/bin/uv" sync --frozen --no-progress --extra train

# --- Install unsloth WITHOUT its deps ---
# pyproject.toml says: unsloth is NOT in [project.optional-dependencies.train]
# because its declared dep `trl<=0.24.0` would downgrade our pinned trl==0.29.0.
# The `--no-deps` flag is the project's required install path.
# Reference: pyproject.toml lines 53-54
echo ""
echo "=== Install unsloth + unsloth_zoo (--no-deps) ==="
# uv pip install does NOT honor UV_PROJECT_ENVIRONMENT — point at the venv
# explicitly with --python, or it errors out with "No virtual environment found".
"${KF_UV_INSTALL_DIR}/bin/uv" pip install \
  --python "${KF_VENV}/bin/python" \
  --no-deps unsloth unsloth_zoo

# --- Verify CUDA + torch ---
echo ""
echo "=== Verify torch + CUDA on this node ==="
"${KF_VENV}/bin/python" - <<'PY'
import sys
print("python      :", sys.version.split()[0])
try:
    import torch
    print("torch       :", torch.__version__)
    print("cuda avail  :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device      :", torch.cuda.get_device_name(0))
        print("cuda compiled:", torch.version.cuda)
        print("capability  :", torch.cuda.get_device_capability(0))
except Exception as e:
    print("torch import FAILED:", repr(e))

for mod in ("transformers", "trl", "vllm", "triton", "datasets", "accelerate", "unsloth", "unsloth_zoo"):
    try:
        m = __import__(mod)
        print(f"{mod:13s}:", getattr(m, "__version__", "?"))
    except Exception as e:
        print(f"{mod:13s}: import failed -> {e}")
PY

echo ""
echo "========================================"
echo "kf_setup complete: $(date)"
echo "  venv: ${KF_VENV}"
echo "  activate manually with:"
echo "    source ${REPO_ROOT}/scripts/cluster/cluster_paths.sh"
echo "    source \${KF_VENV}/bin/activate"
echo "========================================"
