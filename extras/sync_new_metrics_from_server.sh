#!/usr/bin/env bash
# Pull new_metrics CSV/JSON from Salman server → local extras/output/ (rsync).
#
# Prefer rsync over git for large per-vehicle CSVs.
#
# Usage (Mac, from repo root):
#   chmod +x extras/sync_new_metrics_from_server.sh
#   export SERVER=madina@salmansserver.tail8a0d62.ts.net   # or your SSH host
#   ./extras/sync_new_metrics_from_server.sh
#
# One agent/network/run only:
#   AGENT=dqn NETWORK=sumo4x4 RUN_NAME=seed42_steps3600 ./extras/sync_new_metrics_from_server.sh
#
# Dry run:
#   DRY_RUN=1 ./extras/sync_new_metrics_from_server.sh

set -euo pipefail

SERVER="${SERVER:-madina@salmansserver.tail8a0d62.ts.net}"
REMOTE_REPO="${REMOTE_REPO:-~/LibSignalFork}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_ROOT="${REPO_ROOT}/extras/output"
REMOTE_ROOT="${REMOTE_REPO}/extras/output"

AGENT="${AGENT:-}"
NETWORK="${NETWORK:-}"
RUN_NAME="${RUN_NAME:-}"

RSYNC_FLAGS=(-avz --progress)
if [[ -n "${DRY_RUN:-}" ]]; then
  RSYNC_FLAGS+=(-n)
fi

mkdir -p "${LOCAL_ROOT}"

if [[ -n "${AGENT}" && -n "${NETWORK}" && -n "${RUN_NAME}" ]]; then
  remote="${REMOTE_ROOT}/${AGENT}/${NETWORK}/${RUN_NAME}/"
  local="${LOCAL_ROOT}/${AGENT}/${NETWORK}/${RUN_NAME}/"
  mkdir -p "${local}"
  echo "rsync ${remote} -> ${local}"
  rsync "${RSYNC_FLAGS[@]}" "${SERVER}:${remote}" "${local}"
else
  echo "rsync ${SERVER}:${REMOTE_ROOT}/ -> ${LOCAL_ROOT}/"
  rsync "${RSYNC_FLAGS[@]}" "${SERVER}:${REMOTE_ROOT}/" "${LOCAL_ROOT}/"
fi

echo ""
echo "Done. Local files:"
find "${LOCAL_ROOT}" -name 'new_metrics.csv' 2>/dev/null | head -20
