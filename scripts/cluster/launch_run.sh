#!/usr/bin/env bash
# Coordinated launch: A100 eval server first, then H200 training job.
#
# Usage (on the cluster, from repo root):
#   bash scripts/cluster/launch_run.sh stage1
#   bash scripts/cluster/launch_run.sh stage2
#   bash scripts/cluster/launch_run.sh stage3
#
# This submits kf_eval_server.slurm, waits until it publishes its address,
# then submits the requested kf_stage*.slurm. The training job's own waiting
# logic adds a second safety net.

set -euo pipefail

STAGE="${1:-stage1}"
case "${STAGE}" in
  stage1|stage2|stage3) ;;
  *) echo "Usage: $0 {stage1|stage2|stage3}" >&2; exit 1 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/cluster/cluster_paths.sh"

# The kf_*.slurm scripts declare #SBATCH --output=logs/... relative to the
# submit dir; slurm needs that directory to exist at job START or the job
# dies immediately on a fresh tree. Create it before any sbatch.
mkdir -p "${REPO_ROOT}/logs" "${KF_LOGS}"

ADDRESS_FILE="${KF_SCRATCH_ROOT}/eval_server.address"

# Clear any stale address from a prior crashed run.
rm -f "${ADDRESS_FILE}"

echo "Submitting eval server..."
EVAL_OUT=$(sbatch --parsable scripts/cluster/kf_eval_server.slurm)
EVAL_JOBID="${EVAL_OUT}"
echo "  eval server job: ${EVAL_JOBID}"

echo "Waiting up to 10 min for eval server to publish address..."
for i in $(seq 1 60); do
  if [ -s "${ADDRESS_FILE}" ]; then
    echo "  address: $(cat "${ADDRESS_FILE}")"
    break
  fi
  sleep 10
done
if [ ! -s "${ADDRESS_FILE}" ]; then
  echo "Eval server did not publish an address. Check logs/kf_eval_server_${EVAL_JOBID}.out" >&2
  echo "Cancelling eval server job ${EVAL_JOBID} so it does not burn allocation." >&2
  scancel "${EVAL_JOBID}" || true
  exit 2
fi

echo ""
echo "Submitting kf_${STAGE}.slurm..."
TRAIN_OUT=$(sbatch --parsable "scripts/cluster/kf_${STAGE}.slurm")
TRAIN_JOBID="${TRAIN_OUT}"
echo "  training job: ${TRAIN_JOBID}"

echo ""
echo "Both jobs submitted:"
echo "  eval server : ${EVAL_JOBID}   (logs/kf_eval_server_${EVAL_JOBID}.out)"
echo "  ${STAGE}      : ${TRAIN_JOBID}  (logs/kf_${STAGE}_${TRAIN_JOBID}.out)"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER -o '%.10i %.20j %.8T %.10M %R'"
echo ""
echo "When training finishes, cancel the eval server to free the A100:"
echo "  scancel ${EVAL_JOBID}"
