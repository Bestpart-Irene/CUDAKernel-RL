#!/bin/bash
# EXP-004 diagnostic: 1 rollout per WCC-filtered stage1 task (4 tasks total).
# Loads EXP-002 SFT adapter, generates one completion per task, walks the
# full extract → compile → eval chain, and dumps per-task JSON into
# outputs/exp004_debug/ for offline bucket attribution.
# Expected runtime: ~15-20 min (model load ~5min + 4 rollouts × ~1-2min each).

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
# EXP-004: align dataset selection with EXP-003 D6 stage1 (WCC-only).
export KERNELFORGE_STAGE1_BACKEND_FILTER=wcc
# Persistent JSON dump location (NOT /tmp on the compute node).
export KERNELFORGE_EXP004_OUTDIR="${REPO_ROOT}/outputs/exp004_debug"
mkdir -p "${KERNELFORGE_EXP004_OUTDIR}"

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
