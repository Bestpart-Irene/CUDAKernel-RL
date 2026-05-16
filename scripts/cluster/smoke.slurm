#!/bin/bash
# KernelForge smoke job — proves the cluster env is alive end-to-end.
#
# Submit from the repo root on the cluster:
#   cd /home/$USER/CUDAKernel-RL
#   sbatch scripts/cluster/smoke.slurm
#
# What this does:
#   - activates the KernelForge venv on /scratch
#   - prints torch / CUDA / GPU info
#   - runs scripts/smoke_test.py (existing repo smoke test)
#   - (optional) pings the eval service if KERNELFORGE_EVAL_URL is set
#
# This is a 30-minute job on one A100. Cost-cheap on Explorer.

#SBATCH --job-name=kf_smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/kf_smoke_%j.out
#SBATCH --error=logs/kf_smoke_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/cluster/cluster_paths.sh"

mkdir -p "${REPO_ROOT}/logs"

echo "========================================"
echo "kf_smoke job_id  : ${SLURM_JOB_ID:-local}"
echo "node             : ${SLURMD_NODENAME:-$(hostname)}"
echo "started          : $(date)"
echo "repo             : ${REPO_ROOT}"
echo "venv             : ${KF_VENV}"
echo "scratch          : ${KF_SCRATCH_ROOT}"
echo "========================================"

if [ ! -d "${KF_VENV}" ]; then
  echo "FATAL: venv not found at ${KF_VENV}"
  echo "Run: sbatch scripts/cluster/setup_env.slurm first."
  exit 1
fi

# Activate the venv
# shellcheck disable=SC1091
source "${KF_VENV}/bin/activate"

echo ""
echo "=== Environment sanity ==="
python - <<'PY'
import torch
print("torch        :", torch.__version__)
print("cuda avail   :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device       :", torch.cuda.get_device_name(0))
    print("capability   :", torch.cuda.get_device_capability(0))
    print("cuda runtime :", torch.version.cuda)
PY

echo ""
echo "=== nvidia-smi ==="
nvidia-smi | head -20 || echo "nvidia-smi unavailable"

echo ""
echo "=== Repo smoke test ==="
python scripts/smoke_test.py

# --- Optional: eval service ping ---
if [ -n "${KERNELFORGE_EVAL_URL:-}" ]; then
  echo ""
  echo "=== Eval service ping: ${KERNELFORGE_EVAL_URL} ==="
  curl -sS --max-time 15 "${KERNELFORGE_EVAL_URL}/health" || echo "eval service ping failed"
else
  echo ""
  echo "(KERNELFORGE_EVAL_URL not set — skipping eval service ping)"
fi

echo ""
echo "========================================"
echo "kf_smoke complete: $(date)"
echo "========================================"
