#!/bin/bash
# Stage 1 — GRPO warm-up (100 steps), trained on H200, eval HTTP-dispatched to A100.
#
# Prereq: kf_eval_server.slurm must be running (publishes address to
# ${KF_SCRATCH_ROOT}/eval_server.address). Use scripts/cluster/launch_run.sh
# to coordinate, or sbatch this manually after the eval server is up.

#SBATCH --job-name=kf_stage1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/kf_stage1_%j.out
#SBATCH --error=logs/kf_stage1_%j.err
# Note: NU policy caps training walltime at 8h regardless of partition.
# Stage1 = 100 GRPO steps. If it doesn't finish in 8h, the run will be
# truncated; checkpoints should still be saved periodically by the trainer.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/cluster/cluster_paths.sh"
mkdir -p "${REPO_ROOT}/logs"

# --- Graceful shutdown on slurm SIGTERM (walltime / scancel) ---
# Slurm sends SIGTERM 60s before SIGKILL. Use that window to copy any
# in-progress checkpoint to a "interrupted" subdirectory so the next job
# can resume from the latest step instead of losing everything.
on_term() {
  echo "=== Caught SIGTERM at $(date) — preserving in-progress state ==="
  if [ -d "${KF_CHECKPOINTS}/stage1" ]; then
    INTERRUPT_DIR="${KF_CHECKPOINTS}/stage1_interrupted_${SLURM_JOB_ID:-local}"
    cp -r "${KF_CHECKPOINTS}/stage1" "${INTERRUPT_DIR}" 2>/dev/null && \
      echo "Snapshot saved to ${INTERRUPT_DIR}"
  fi
  echo "=== Tail of recent stdout for context ==="
  tail -50 "${REPO_ROOT}/logs/kf_stage1_${SLURM_JOB_ID}.out" 2>/dev/null || true
  exit 143
}
trap on_term SIGTERM SIGINT

# H200 141GB fits Qwen3-Coder-30B-A3B at bf16 with vLLM/GRPO headroom.
export KERNELFORGE_QUANT_BITS="${KERNELFORGE_QUANT_BITS:-0}"

# EXP-003: warm-start from Stage 2 SFT adapter checkpoint.
# Set to empty string to load base model instead.
export KERNELFORGE_STAGE1_INIT_CKPT="${KERNELFORGE_STAGE1_INIT_CKPT:-outputs/kernelforge-stage2/checkpoint-50}"

# Override cluster_paths.sh's default (local) — we are using remote A100 eval.
export KERNELFORGE_EVAL_BACKEND=coreweave

# Resolve EVAL_URL from the eval server's published address.
ADDRESS_FILE="${KF_SCRATCH_ROOT}/eval_server.address"
echo "Waiting for eval server address at ${ADDRESS_FILE}..."
for i in $(seq 1 60); do
  if [ -s "${ADDRESS_FILE}" ]; then break; fi
  sleep 10
done
if [ ! -s "${ADDRESS_FILE}" ]; then
  echo "FATAL: eval server address not published after 10 min." >&2
  echo "  Did you sbatch scripts/cluster/kf_eval_server.slurm first?" >&2
  exit 2
fi
EVAL_ADDR="$(cat "${ADDRESS_FILE}")"
export KERNELFORGE_EVAL_URL="http://${EVAL_ADDR}"

echo "Waiting for eval server health at ${KERNELFORGE_EVAL_URL}/health..."
for i in $(seq 1 60); do
  if curl -sS -m 5 "${KERNELFORGE_EVAL_URL}/health" | grep -q '"ok"'; then
    echo "Eval server healthy."
    break
  fi
  sleep 10
done

export WANDB_PROJECT="${KF_WANDB_PROJECT:-kernelforge}"
export WANDB_NAME="stage1-${SLURM_JOB_ID:-local}"
export WANDB_DIR="${KF_LOGS}/wandb"
mkdir -p "${WANDB_DIR}"

if [ -f "${REPO_ROOT}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

echo "========================================"
echo "kf_stage1 job_id  : ${SLURM_JOB_ID:-local}"
echo "node              : ${SLURMD_NODENAME:-$(hostname)}"
echo "started           : $(date)"
echo "eval backend      : ${KERNELFORGE_EVAL_BACKEND}"
echo "eval URL          : ${KERNELFORGE_EVAL_URL}"
echo "quant bits        : ${KERNELFORGE_QUANT_BITS} (0 = bf16 on H200)"
echo "wandb project     : ${WANDB_PROJECT}"
echo "wandb name        : ${WANDB_NAME}"
echo "checkpoints       : ${KF_CHECKPOINTS}/stage1"
echo "========================================"

if [ ! -d "${KF_VENV}" ]; then
  echo "FATAL: venv not found at ${KF_VENV}. Run setup_env.slurm first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${KF_VENV}/bin/activate"

echo ""
echo "=== Preflight ==="
python -m training.grpo_train --preflight-only

echo ""
echo "=== Stage 1 GRPO warm-up ==="
python -m training.grpo_train --stage stage1

echo ""
echo "========================================"
echo "kf_stage1 complete: $(date)"
echo "========================================"
