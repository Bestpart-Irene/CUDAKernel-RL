#!/bin/bash
# Stage 3 — TRLOO-augmented GRPO with curriculum, H200 train, A100 eval (HTTP).

#SBATCH --job-name=kf_stage3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
# NU policy caps training walltime at 8h. Stage3 targets 50 steps; if
# unfinished, resume from checkpoint in a follow-up job.
#SBATCH --output=logs/kf_stage3_%j.out
#SBATCH --error=logs/kf_stage3_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/cluster/cluster_paths.sh"
mkdir -p "${REPO_ROOT}/logs"

export KERNELFORGE_QUANT_BITS="${KERNELFORGE_QUANT_BITS:-0}"
export KERNELFORGE_MAX_TURNS="${KERNELFORGE_MAX_TURNS:-3}"
export KERNELFORGE_EVAL_BACKEND=coreweave
# Explicit re-export so sbatch --export=ALL,VAR=val values reach Python's
# os.getenv. Every KERNELFORGE_* knob the trainer reads must be listed.
export KERNELFORGE_STAGE3_NUM_GENERATIONS="${KERNELFORGE_STAGE3_NUM_GENERATIONS:-2}"
export KERNELFORGE_STAGE3_MAX_STEPS="${KERNELFORGE_STAGE3_MAX_STEPS:-50}"
export KERNELFORGE_GRPO_LOSS_TYPE="${KERNELFORGE_GRPO_LOSS_TYPE:-dapo}"
export KERNELFORGE_REWARD_VERSION="${KERNELFORGE_REWARD_VERSION:-v2-shaped}"
export KERNELFORGE_ROLLOUT_DEBUG="${KERNELFORGE_ROLLOUT_DEBUG:-0}"
export KERNELFORGE_VERIFIER_DEBUG="${KERNELFORGE_VERIFIER_DEBUG:-0}"

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

export WANDB_PROJECT="${KF_WANDB_PROJECT:-kernelforge}"
export WANDB_NAME="stage3-${SLURM_JOB_ID:-local}"
export WANDB_DIR="${KF_LOGS}/wandb"
mkdir -p "${WANDB_DIR}"

if [ -f "${REPO_ROOT}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

echo "========================================"
echo "kf_stage3 job_id  : ${SLURM_JOB_ID:-local}"
echo "node              : ${SLURMD_NODENAME:-$(hostname)}"
echo "started           : $(date)"
echo "eval backend      : ${KERNELFORGE_EVAL_BACKEND}"
echo "eval URL          : ${KERNELFORGE_EVAL_URL}"
echo "quant bits        : ${KERNELFORGE_QUANT_BITS}"
echo "max turns         : ${KERNELFORGE_MAX_TURNS}"
echo "wandb project     : ${WANDB_PROJECT}"
echo "wandb name        : ${WANDB_NAME}"
echo "checkpoints       : ${KF_CHECKPOINTS}/stage3"
echo "========================================"

if [ ! -d "${KF_VENV}" ]; then
  echo "FATAL: venv not found at ${KF_VENV}. Run setup_env.slurm first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${KF_VENV}/bin/activate"

echo ""
echo "=== Stage 3 GRPO + curriculum ==="
python -m training.grpo_train --stage stage3

echo ""
echo "========================================"
echo "kf_stage3 complete: $(date)"
echo "========================================"
