#!/bin/bash
# EXP-018c-p0 — base-model cold-start feasibility probe (NO training).
#
# Planner spec: research/audits/2026-08-10-planner-plan-coherence.md §6.
# Measures p(parseable + nvcc-compilable extern "C" kernel) from BASE
# Qwen3-Coder-30B-A3B on the migrated (fc6c8f1) prompts for the 3-task
# spike pool (vector_add_e2 / F.elu / F.softplus), 32 samples per task,
# temperature 1.0, max_new_tokens 2048, through the frozen extern-C
# evaluator. This is the do-not-repeat 2026-05-16 revisit-condition test
# that must run BEFORE any 018c training relaunch.
#
# Verdict (printed + outputs/probe_018c_p0/summary.json):
#   "COLD-START LICENSED (p>=0.05)"  -> 018c rerun is ledger-legal
#   "EXP-018B REQUIRED (p<0.05)"     -> SFT rebuild is a hard prerequisite
#
# Every completion is fsync'd to outputs/probe_018c_p0/completions.jsonl
# as it finishes — SIGTERM loses nothing, and re-submitting resumes.
# Supersedes the deleted scripts/cluster/exp018c.slurm (cold-started in
# violation of the ledger; capped steps below its own pass criteria).

#SBATCH --job-name=kf_probe_018c_p0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/kf_probe_018c_p0_%j.out
#SBATCH --error=logs/kf_probe_018c_p0_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/cluster/cluster_paths.sh"
mkdir -p "${REPO_ROOT}/logs"

# bf16 on H200; no Unsloth (Qwen3 MoE fast path is upstream-broken).
export KERNELFORGE_QUANT_BITS="${KERNELFORGE_QUANT_BITS:-0}"
export KERNELFORGE_SKIP_UNSLOTH=1

# Probe compiles AND runs candidates on this H200 (sm_90). Correctness-only
# question — H200-vs-A100 timing incomparability does not apply here.
export KERNELFORGE_TARGET_ARCH="${KERNELFORGE_TARGET_ARCH:-sm_90}"
export KERNELFORGE_EVAL_BACKEND=local
export KERNELFORGE_EVAL_URL="http://disabled:0"

# Probe knobs (spec defaults; override via sbatch --export=ALL,VAR=val).
export PROBE_018C_P0_SAMPLES="${PROBE_018C_P0_SAMPLES:-32}"
export PROBE_018C_P0_MAX_NEW_TOKENS="${PROBE_018C_P0_MAX_NEW_TOKENS:-2048}"
export PROBE_018C_P0_TEMPERATURE="${PROBE_018C_P0_TEMPERATURE:-1.0}"
export PROBE_018C_P0_GEN_BATCH="${PROBE_018C_P0_GEN_BATCH:-8}"
export PROBE_018C_P0_OUTDIR="${PROBE_018C_P0_OUTDIR:-${REPO_ROOT}/outputs/probe_018c_p0}"
mkdir -p "${PROBE_018C_P0_OUTDIR}"

if [ ! -d "${KF_VENV}" ]; then
  echo "FATAL: venv not found at ${KF_VENV}. Run setup_env.slurm first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${KF_VENV}/bin/activate"
module load cuda/12.8.0 || { echo "FATAL: cuda module load failed" >&2; exit 1; }

echo "================================================="
echo "EXP-018c-p0 cold-start feasibility probe"
echo "job_id        : ${SLURM_JOB_ID:-local}"
echo "node          : ${SLURMD_NODENAME:-$(hostname)}"
echo "started       : $(date)"
echo "samples/task  : ${PROBE_018C_P0_SAMPLES}"
echo "max_new_tokens: ${PROBE_018C_P0_MAX_NEW_TOKENS}"
echo "temperature   : ${PROBE_018C_P0_TEMPERATURE}"
echo "target arch   : ${KERNELFORGE_TARGET_ARCH}"
echo "outdir        : ${PROBE_018C_P0_OUTDIR}"
echo "nvcc          : $(nvcc --version | head -4 | tail -1)"
echo "gpu           : $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
echo "================================================="

# Preflight: the venv must already carry torch/transformers (setup_env.slurm).
# If a dependency is missing, install it with uv pip AND AN EXPLICIT --python:
# `uv pip install` does NOT inherit UV_PROJECT_ENVIRONMENT (it errors with
# "No virtual environment found" or, worse, targets another env). The
# project-required form, exactly as scripts/cluster/setup_env.slurm uses:
#   "${KF_UV_INSTALL_DIR}/bin/uv" pip install --python "${KF_VENV}/bin/python" <pkg>
python - <<'PY'
import importlib.util
missing = [m for m in ("torch", "transformers", "peft") if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(
        f"FATAL: missing packages in venv: {missing}. Run setup_env.slurm, or "
        "install with: uv pip install --python \"${KF_VENV}/bin/python\" <pkg>"
    )
print("preflight imports OK")
PY

# Run the probe in the background and forward SIGTERM to it so its own
# handler fires (it fsyncs each finished sample and writes a partial
# summary). A foreground child would swallow the trap until exit.
python -u scripts/probe_pass_at_k.py &
PROBE_PID=$!
on_term() {
  echo "=== Caught SIGTERM at $(date) — forwarding to probe (pid ${PROBE_PID}) ==="
  kill -TERM "${PROBE_PID}" 2>/dev/null || true
  wait "${PROBE_PID}" || true
  echo "Probe exited after SIGTERM; finished samples are preserved in ${PROBE_018C_P0_OUTDIR}/completions.jsonl"
  exit 143
}
trap on_term SIGTERM SIGINT
# `wait` under set -e: capture the exit code without killing the script.
RC=0
wait "${PROBE_PID}" || RC=$?

echo "================================================="
echo "EXP-018c-p0 done: $(date)  rc=${RC}"
echo "summary: ${PROBE_018C_P0_OUTDIR}/summary.json"
echo "================================================="
exit "${RC}"
