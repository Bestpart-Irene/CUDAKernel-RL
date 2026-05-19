#!/bin/bash
# Stage 2 — RFT/SFT on filtered trajectories, H200 train, A100 eval (HTTP).

#SBATCH --job-name=kf_stage2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
# NU policy caps training walltime at 8h.
#SBATCH --output=logs/kf_stage2_%j.out
#SBATCH --error=logs/kf_stage2_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/cluster/cluster_paths.sh"
mkdir -p "${REPO_ROOT}/logs"

# Redirect stage outputs to scratch — /home is quota-limited (40GB on
# Explorer) and a single SFT checkpoint is ~10GB. Filling /home breaks
# slurm log writes and silently kills new jobs with RaisedSignal:53.
export KERNELFORGE_STAGE1_OUTPUT="${KERNELFORGE_STAGE1_OUTPUT:-${KF_CHECKPOINTS}/stage1}"
export KERNELFORGE_STAGE2_OUTPUT="${KERNELFORGE_STAGE2_OUTPUT:-${KF_CHECKPOINTS}/stage2}"
export KERNELFORGE_STAGE3_OUTPUT="${KERNELFORGE_STAGE3_OUTPUT:-${KF_CHECKPOINTS}/stage3}"
mkdir -p "${KERNELFORGE_STAGE1_OUTPUT}" "${KERNELFORGE_STAGE2_OUTPUT}" "${KERNELFORGE_STAGE3_OUTPUT}"

export KERNELFORGE_QUANT_BITS="${KERNELFORGE_QUANT_BITS:-0}"
export KERNELFORGE_EVAL_BACKEND=coreweave
# EXP-002: SFT-first reorder. Skips broken RFT trajectory collection
# (requires non-existent Stage 1 checkpoint) and trains SFT directly on
# datasets/doublegraph_sft.jsonl. See research/experiments/EXP-002-stage2-sft-first.md.
export KERNELFORGE_SKIP_RFT_COLLECTION="${KERNELFORGE_SKIP_RFT_COLLECTION:-1}"
# Explicit re-export so sbatch --export=ALL,VAR=val values reach Python's
# os.getenv. Every KERNELFORGE_* knob the trainer reads must be listed —
# missing exports silently fall back to module-level defaults.
export KERNELFORGE_SFT_DATA_SOURCE="${KERNELFORGE_SFT_DATA_SOURCE:-doublegraph}"
export KERNELFORGE_STAGE2_EPOCHS="${KERNELFORGE_STAGE2_EPOCHS:-3}"
export KERNELFORGE_RFT_TRAJECTORIES="${KERNELFORGE_RFT_TRAJECTORIES:-50}"
export KERNELFORGE_RFT_MIN_REWARD="${KERNELFORGE_RFT_MIN_REWARD:-1.0}"
export KERNELFORGE_REWARD_VERSION="${KERNELFORGE_REWARD_VERSION:-v2-shaped}"

if [ "${KERNELFORGE_SKIP_RFT_COLLECTION}" = "1" ]; then
  echo "KERNELFORGE_SKIP_RFT_COLLECTION=1 → SFT-only run; skipping eval server wait."
  export KERNELFORGE_EVAL_URL="http://disabled:0"
else
  ADDRESS_FILE="${KF_SCRATCH_ROOT}/eval_server.address"
  echo "Waiting for eval server address at ${ADDRESS_FILE}..."
  for i in $(seq 1 60); do
    if [ -s "${ADDRESS_FILE}" ]; then break; fi
    sleep 10
  done
  if [ ! -s "${ADDRESS_FILE}" ]; then
    echo "FATAL: eval server address not published after 10 min." >&2
    exit 2
  fi
  EVAL_ADDR="$(cat "${ADDRESS_FILE}")"
  export KERNELFORGE_EVAL_URL="http://${EVAL_ADDR}"

  echo "Waiting for eval server health..."
  for i in $(seq 1 60); do
    if curl -sS -m 5 "${KERNELFORGE_EVAL_URL}/health" | grep -q '"ok"'; then break; fi
    sleep 10
  done
fi

export WANDB_PROJECT="${KF_WANDB_PROJECT:-kernelforge}"
export WANDB_NAME="stage2-${SLURM_JOB_ID:-local}"
export WANDB_DIR="${KF_LOGS}/wandb"
mkdir -p "${WANDB_DIR}"

if [ -f "${REPO_ROOT}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

echo "========================================"
echo "kf_stage2 job_id  : ${SLURM_JOB_ID:-local}"
echo "node              : ${SLURMD_NODENAME:-$(hostname)}"
echo "started           : $(date)"
echo "eval backend      : ${KERNELFORGE_EVAL_BACKEND}"
echo "eval URL          : ${KERNELFORGE_EVAL_URL}"
echo "quant bits        : ${KERNELFORGE_QUANT_BITS}"
echo "wandb project     : ${WANDB_PROJECT}"
echo "wandb name        : ${WANDB_NAME}"
echo "checkpoints       : ${KF_CHECKPOINTS}/stage2"
echo "========================================"

if [ ! -d "${KF_VENV}" ]; then
  echo "FATAL: venv not found at ${KF_VENV}. Run setup_env.slurm first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${KF_VENV}/bin/activate"

echo ""
echo "=== Stage 2 RFT/SFT ==="
python -m training.grpo_train --stage stage2

echo ""
echo "========================================"
echo "kf_stage2 complete: $(date)"
echo "========================================"
