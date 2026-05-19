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

# Redirect ALL stage outputs to scratch by default — /home is quota-limited
# (40GB on Explorer) and a single Stage 2 checkpoint is ~10GB. Jobs filling
# /home break slurm's ability to write logs and silently get killed with
# RaisedSignal:53. Override only if you know what you're doing.
export KERNELFORGE_STAGE1_OUTPUT="${KERNELFORGE_STAGE1_OUTPUT:-${KF_CHECKPOINTS}/stage1}"
export KERNELFORGE_STAGE2_OUTPUT="${KERNELFORGE_STAGE2_OUTPUT:-${KF_CHECKPOINTS}/stage2}"
export KERNELFORGE_STAGE3_OUTPUT="${KERNELFORGE_STAGE3_OUTPUT:-${KF_CHECKPOINTS}/stage3}"
mkdir -p "${KERNELFORGE_STAGE1_OUTPUT}" "${KERNELFORGE_STAGE2_OUTPUT}" "${KERNELFORGE_STAGE3_OUTPUT}"

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

# EXP-008 fix: explicit re-export so sbatch --export=ALL,VAR=val values
# actually reach Python's os.getenv. Without this the comma-list values
# silently dropped (verified twice with ROLLOUT_DEBUG and VERIFIER_DEBUG).
# Add every KERNELFORGE_* knob the trainer reads here — missing exports
# silently fall back to module-level defaults.
export KERNELFORGE_ROLLOUT_DEBUG="${KERNELFORGE_ROLLOUT_DEBUG:-0}"
export KERNELFORGE_VERIFIER_DEBUG="${KERNELFORGE_VERIFIER_DEBUG:-0}"
export KERNELFORGE_STAGE1_NUM_GENERATIONS="${KERNELFORGE_STAGE1_NUM_GENERATIONS:-2}"
export KERNELFORGE_STAGE1_BETA="${KERNELFORGE_STAGE1_BETA:-0.04}"
export KERNELFORGE_STAGE1_MAX_STEPS="${KERNELFORGE_STAGE1_MAX_STEPS:-100}"
export KERNELFORGE_GRPO_LOSS_TYPE="${KERNELFORGE_GRPO_LOSS_TYPE:-dapo}"
export KERNELFORGE_REWARD_VERSION="${KERNELFORGE_REWARD_VERSION:-v2-shaped}"

# EXP-003 D6 (verification): warm-start from doubleGraph Stage 2 ckpt AND
# filter Stage 1 dataset to wcc-backend tasks only so the SFT contract
# (void wcc_kernel(...)) matches the eval contract end-to-end.
# Two coupled env vars implement the single logical change "align Stage 1
# domain with SFT contract":
#   KERNELFORGE_STAGE1_INIT_CKPT       = doubleGraph SFT ckpt path
#   KERNELFORGE_STAGE1_BACKEND_FILTER  = wcc (drop ops6k tasks at load)
# Override to ops6k contract later by switching ckpt to
# outputs/kernelforge-stage2-sakana/checkpoint-50 and unsetting the filter.
# Set INIT_CKPT to empty to skip SFT warm-start entirely.
export KERNELFORGE_STAGE1_INIT_CKPT="${KERNELFORGE_STAGE1_INIT_CKPT:-outputs/kernelforge-stage2/checkpoint-50}"
export KERNELFORGE_STAGE1_BACKEND_FILTER="${KERNELFORGE_STAGE1_BACKEND_FILTER:-wcc}"

# =============================================================================
# VERIFICATION PHASE — DIAGNOSTIC ONLY, NOT MASTER-PROMOTION-ELIGIBLE
# =============================================================================
# Default backend is `local` (eval runs in-process on the SAME H200 as training)
# because A100 capacity on NU Explorer is unreliable. Reward speedup numbers
# from local-H200 eval are NOT comparable to A100 baselines:
#   - H200 (sm_90) ≠ A100 (sm_80) — different bandwidth, SM count, compile target
#   - doubleGraph expert kernels were tuned for A100
#   - PyTorch eager/compile is also faster on H200, so speedup ratios shift
# This run answers ONLY two binary questions:
#   1. Does SFT prior give GRPO non-zero reward variance? (reward_std > 0)
#   2. Does grad_norm > 0 (model is actually learning)?
# Any speedup_vs_orig / pass_rate / fast_p numbers reported here are NOT to
# be promoted into research/live/master.json. The A100-comparable cold-start
# is a separate future run with KERNELFORGE_EVAL_BACKEND=coreweave.
# Set KERNELFORGE_EVAL_BACKEND=coreweave externally to override for that run.
# =============================================================================
export KERNELFORGE_EVAL_BACKEND="${KERNELFORGE_EVAL_BACKEND:-local}"

if [ "${KERNELFORGE_EVAL_BACKEND}" = "local" ]; then
  echo "VERIFICATION PHASE: KERNELFORGE_EVAL_BACKEND=local — skipping eval server wait."
  export KERNELFORGE_EVAL_URL="http://disabled:0"
else
  # Production (A100-comparable) path: wait for the FastAPI eval server.
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
fi

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
