#!/bin/bash
# EXP-018a step C: validate extern "C" + dlsym eval contract end-to-end.
# Compiles 5 hand-written candidates (correct / wrong / 2 hacks /
# passthrough) for a vector_add task, runs each through the new harness,
# verifies bucket attribution matches expectation.
# Expected runtime: ~3-5 min (5 nvcc + 5 evals, no model load).

#SBATCH --job-name=kf_probe_externc
#SBATCH --partition=sharing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/kf_probe_externc_%j.out
#SBATCH --error=logs/kf_probe_externc_%j.err

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

# A100 = sm_80. Override via env if running on h200.
export EXTERN_C_SM_ARCH="${EXTERN_C_SM_ARCH:-sm_80}"

python3 scripts/probe_extern_c_eval.py
