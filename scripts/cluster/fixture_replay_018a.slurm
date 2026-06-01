#!/bin/bash
# EXP-018a Day 5 acceptance gate: run fixture replay through production
# evaluate_ops6k_kernel_impl(). 5 hack variants must be source-rejected,
# 1 positive control (clean elu) must compile + correct.

#SBATCH --job-name=kf_fixture_018a
#SBATCH --partition=sharing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/kf_fixture_018a_%j.out
#SBATCH --error=logs/kf_fixture_018a_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/cluster/cluster_paths.sh"
mkdir -p "${REPO_ROOT}/logs"

if [ ! -d "${KF_VENV}" ]; then
  echo "FATAL: venv not found at ${KF_VENV}." >&2
  exit 1
fi
source "${KF_VENV}/bin/activate"

module load cuda/12.8.0 || { echo "FATAL: cuda module load failed" >&2; exit 1; }
echo "[env] $(nvcc --version | head -4 | tail -1)"
echo "[env] $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"

# A100 architecture
export KERNELFORGE_CUDA_ARCH="${KERNELFORGE_CUDA_ARCH:-sm_80}"

python3 scripts/fixture_replay_018a.py
