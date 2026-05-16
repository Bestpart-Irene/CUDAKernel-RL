#!/bin/bash
# Eval server — FastAPI evaluation service on a dedicated A100 node.
#
# This is the A100 half of the H200-train / A100-eval split. Reward timing
# is measured on A100 to keep speedup_vs_orig comparable to the project
# baseline. Submit BEFORE any kf_stage*.slurm — training jobs read this
# server's address from ${KF_EVAL_SERVER_ADDRESS_FILE}.
#
# Submit from the repo root on the cluster:
#   cd /home/$USER/CUDAKernel-RL
#   sbatch scripts/cluster/kf_eval_server.slurm

#SBATCH --job-name=kf_eval_server
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/kf_eval_server_%j.out
#SBATCH --error=logs/kf_eval_server_%j.err
# Note: gpu partition has 8h time limit. Stage1 (~4-6h) fits inside, with
# ~2h headroom. If you switch to stage3 (12h target), promote this back to
# multigpu/24h and accept the longer queue wait.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/cluster/cluster_paths.sh"
mkdir -p "${REPO_ROOT}/logs"

# Eval server is stateless — on SIGTERM, just drop the address file so
# any in-flight stage1 immediately sees the disconnect instead of timing out.
on_term() {
  echo "=== Caught SIGTERM at $(date) — eval server going down ==="
  rm -f "${KF_SCRATCH_ROOT}/eval_server.address"
  exit 143
}
trap on_term SIGTERM SIGINT

# Pick a high-numbered port to avoid clashes with other users on the node.
EVAL_PORT="${KF_EVAL_PORT:-$(( 18000 + (SLURM_JOB_ID % 1000) ))}"
EVAL_HOST="$(hostname -s)"
EVAL_ADDR="${EVAL_HOST}:${EVAL_PORT}"
ADDRESS_FILE="${KF_SCRATCH_ROOT}/eval_server.address"

# Source .env if present (HF_TOKEN, WANDB key — eval server only needs HF for
# loading reference models from Ops-6K).
if [ -f "${REPO_ROOT}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

echo "========================================"
echo "kf_eval_server job_id : ${SLURM_JOB_ID:-local}"
echo "node                  : ${EVAL_HOST}"
echo "port                  : ${EVAL_PORT}"
echo "address               : ${EVAL_ADDR}"
echo "address file          : ${ADDRESS_FILE}"
echo "started               : $(date)"
echo "========================================"

if [ ! -d "${KF_VENV}" ]; then
  echo "FATAL: venv not found at ${KF_VENV}. Run setup_env.slurm first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${KF_VENV}/bin/activate"

echo ""
echo "=== GPU info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Publish address atomically (write tmp + rename) so the training job can poll.
mkdir -p "$(dirname "${ADDRESS_FILE}")"
echo "${EVAL_ADDR}" > "${ADDRESS_FILE}.tmp"
mv -f "${ADDRESS_FILE}.tmp" "${ADDRESS_FILE}"
echo "Published address to ${ADDRESS_FILE}"

# Clean up on exit so stale addresses do not mislead future training jobs.
trap 'rm -f "${ADDRESS_FILE}"; echo "Removed ${ADDRESS_FILE}"' EXIT TERM INT

echo ""
echo "=== Starting FastAPI eval service ==="
echo "URL: http://${EVAL_ADDR}"
exec python -m uvicorn eval_service.app:app \
  --host 0.0.0.0 \
  --port "${EVAL_PORT}" \
  --workers 1 \
  --log-level info
