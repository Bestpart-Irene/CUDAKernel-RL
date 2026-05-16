#!/usr/bin/env bash
# Push the local CUDAKernel-RL repo to the Northeastern Explorer cluster.
#
# Run this from your laptop, NOT from the cluster:
#   bash scripts/cluster/sync_to_discovery.sh
#
# Override these via environment if your netid or host differ:
#   KF_NETID=xie.yiyi
#   KF_DISCOVERY_HOST=login.discovery.neu.edu   # confirm with NU RC docs
#
# Strict isolation: this script will only ever rsync TO
#   /home/${KF_NETID}/CUDAKernel-RL/
# and refuses to touch rl-casino paths.

set -euo pipefail

LOCAL_REPO="${LOCAL_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
KF_NETID="${KF_NETID:-xie.yiyi}"
# Default to Explorer (the active NU RC cluster). Override with
# KF_DISCOVERY_HOST=login.discovery.neu.edu if you really mean Discovery.
KF_DISCOVERY_HOST="${KF_DISCOVERY_HOST:-login.explorer.northeastern.edu}"
KF_REMOTE_REPO="${KF_REMOTE_REPO:-/home/${KF_NETID}/CUDAKernel-RL}"

# --- Isolation check ---
case "${KF_REMOTE_REPO}" in
  *Reinforcement-Casino*|*rl-casino*|*rl_casino*)
    echo "FATAL: KF_REMOTE_REPO points into rl-casino namespace: ${KF_REMOTE_REPO}" >&2
    exit 99
    ;;
esac

echo "Local  : ${LOCAL_REPO}"
echo "Remote : ${KF_NETID}@${KF_DISCOVERY_HOST}:${KF_REMOTE_REPO}"
echo ""

# Confirm host is reachable
if ! ssh -o BatchMode=yes -o ConnectTimeout=10 \
       "${KF_NETID}@${KF_DISCOVERY_HOST}" "true" 2>/dev/null; then
  echo "WARN: passwordless SSH to ${KF_DISCOVERY_HOST} not configured."
  echo "      You will be prompted for your NU password / Duo push."
  echo ""
fi

# Make sure the remote repo dir exists
ssh "${KF_NETID}@${KF_DISCOVERY_HOST}" "mkdir -p '${KF_REMOTE_REPO}'"

# rsync. Excludes match .gitignore-style: don't push venvs, caches, datasets,
# checkpoints, logs, big uv lock side-files, .git internals if you want fast.
rsync -avz --delete-after \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  --exclude='.pytest_cache/' \
  --exclude='.mypy_cache/' \
  --exclude='.ruff_cache/' \
  --exclude='outputs/' \
  --exclude='checkpoints/' \
  --exclude='logs/' \
  --exclude='research/live/*.log' \
  --exclude='datasets/*.tar' \
  --exclude='datasets/*.parquet' \
  --exclude='archive/' \
  "${LOCAL_REPO}/" \
  "${KF_NETID}@${KF_DISCOVERY_HOST}:${KF_REMOTE_REPO}/"

echo ""
echo "Done."
echo ""
echo "Next on cluster:"
echo "  ssh ${KF_NETID}@${KF_DISCOVERY_HOST}"
echo "  cd ${KF_REMOTE_REPO}"
echo "  sbatch scripts/cluster/setup_env.slurm     # first time only"
echo "  sbatch scripts/cluster/smoke.slurm         # verify env"
