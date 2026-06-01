#!/bin/bash
# EXP-018c — First verification run under the unified extern "C" eval
# contract (CAMPAIGN-018).
#
# 50-step Stage-1 GRPO, ops6k task pool (now extern "C" contract),
# cold start from base Qwen3-Coder-30B-A3B (no SFT prior — old Sakana
# SFT teaches the obsolete torch::-extension contract that gets
# rejected at source scan now).
#
# Pass criteria (mechanically computable post-run):
#   1. mean(reward[40:50]) - mean(reward[0:10]) > 0.15
#   2. linregress over 50 steps: slope > 0, p < 0.05
#   3. >= 1 unique correct=True rollout (sanity floor)
#
# NOT PROMOTION-ELIGIBLE: eval_backend=local on H200.

#SBATCH --job-name=kf_exp018c
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/kf_exp018c_%j.out
#SBATCH --error=logs/kf_exp018c_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/cluster/cluster_paths.sh"
mkdir -p "${REPO_ROOT}/logs"

# Scratch-backed checkpoint outputs (avoid /home quota)
export KERNELFORGE_STAGE1_OUTPUT="${KF_CHECKPOINTS}/stage1_exp018c"
mkdir -p "${KERNELFORGE_STAGE1_OUTPUT}"

# Graceful shutdown on SIGTERM
on_term() {
  echo "=== Caught SIGTERM at $(date) — preserving in-progress state ==="
  if [ -d "${KERNELFORGE_STAGE1_OUTPUT}" ]; then
    INTERRUPT_DIR="${KF_CHECKPOINTS}/stage1_exp018c_interrupted_${SLURM_JOB_ID:-local}"
    cp -r "${KERNELFORGE_STAGE1_OUTPUT}" "${INTERRUPT_DIR}" 2>/dev/null && \
      echo "Snapshot saved to ${INTERRUPT_DIR}"
  fi
  tail -100 "${REPO_ROOT}/logs/kf_exp018c_${SLURM_JOB_ID}.out" 2>/dev/null || true
  exit 143
}
trap on_term SIGTERM SIGINT

# bf16 on H200
export KERNELFORGE_QUANT_BITS=0

# === EXP-018c config: 50-step verification, cold-start, ops6k pool ===
export KERNELFORGE_ROLLOUT_DEBUG=1
export KERNELFORGE_VERIFIER_DEBUG=1
export KERNELFORGE_STAGE1_NUM_GENERATIONS=2
export KERNELFORGE_STAGE1_BETA=0
export KERNELFORGE_STAGE1_MAX_STEPS=25   # 7359525 v1 measured 19min/step at G=2 + max_completion=2048; 25 fits the 8h walltime with margin
export KERNELFORGE_STAGE1_MAX_COMPLETION_LENGTH=2048
export KERNELFORGE_GRPO_LOSS_TYPE=dapo
export KERNELFORGE_REWARD_VERSION=v3-symbol-shaped
export KERNELFORGE_STAGE1_BACKEND_FILTER=ops6k   # restrict to ops_local_fallback (32 rows, new extern "C" contract)

# No SFT warm-start. Sakana SFT taught the OLD torch::-extension contract
# which is now rejected at source scan. Old doublegraph SFT taught WCC
# void contract — also incompatible with extern "C" run_kernel(float*).
# Pure cold start from base Qwen3-Coder-30B-A3B.
export KERNELFORGE_STAGE1_INIT_CKPT=""

# Verification-phase, local subprocess eval (H200 timing != A100 timing
# — NOT promotion-eligible; only the binary "did policy move" question)
export KERNELFORGE_EVAL_BACKEND=local
export KERNELFORGE_EVAL_URL="http://disabled:0"

# wandb
export WANDB_PROJECT="${KF_WANDB_PROJECT:-kernelforge}"
export WANDB_NAME="exp018c-${SLURM_JOB_ID:-local}"
export WANDB_DIR="${KF_LOGS}/wandb"
mkdir -p "${WANDB_DIR}"

if [ -f "${REPO_ROOT}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

echo "================================================="
echo "EXP-018c first verification run under unified extern \"C\" contract"
echo "job_id        : ${SLURM_JOB_ID:-local}"
echo "node          : ${SLURMD_NODENAME:-$(hostname)}"
echo "started       : $(date)"
echo "max_steps     : ${KERNELFORGE_STAGE1_MAX_STEPS}"
echo "G             : ${KERNELFORGE_STAGE1_NUM_GENERATIONS}"
echo "beta          : ${KERNELFORGE_STAGE1_BETA}"
echo "reward        : ${KERNELFORGE_REWARD_VERSION}"
echo "task filter   : ${KERNELFORGE_STAGE1_BACKEND_FILTER}"
echo "SFT init ckpt : '${KERNELFORGE_STAGE1_INIT_CKPT}' (empty = cold start)"
echo "eval backend  : ${KERNELFORGE_EVAL_BACKEND}"
echo "output        : ${KERNELFORGE_STAGE1_OUTPUT}"
echo "================================================="

if [ ! -d "${KF_VENV}" ]; then
  echo "FATAL: venv not found at ${KF_VENV}." >&2
  exit 1
fi
source "${KF_VENV}/bin/activate"
module load cuda/12.8.0 || { echo "FATAL: cuda module load failed" >&2; exit 1; }

python3 -m training.stage1_warmup
