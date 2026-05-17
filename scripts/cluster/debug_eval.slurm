#!/bin/bash
# Diagnostic: 1-rollout probe of the EXP-003 reward chain.
# Loads EXP-002 SFT adapter, generates one completion, walks the full
# extract → compile → eval chain printing what each layer sees.
# Expected runtime: ~10 min (mostly model load).

#SBATCH --job-name=kf_debug_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/kf_debug_eval_%j.out
#SBATCH --error=logs/kf_debug_eval_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/cluster/cluster_paths.sh"
mkdir -p "${REPO_ROOT}/logs"

export KERNELFORGE_QUANT_BITS="${KERNELFORGE_QUANT_BITS:-0}"
export KERNELFORGE_SKIP_UNSLOTH=1
export KERNELFORGE_EVAL_BACKEND=local
export KERNELFORGE_EVAL_URL="http://disabled:0"

if [ ! -d "${KF_VENV}" ]; then
  echo "FATAL: venv not found at ${KF_VENV}." >&2
  exit 1
fi
source "${KF_VENV}/bin/activate"

echo "========================================"
echo "kf_debug_eval job_id : ${SLURM_JOB_ID:-local}"
echo "node                 : ${SLURMD_NODENAME:-$(hostname)}"
echo "started              : $(date)"
echo "========================================"

python -u scripts/debug_eval_pipeline.py

echo "========================================"
echo "kf_debug_eval done: $(date)"
echo "========================================"
