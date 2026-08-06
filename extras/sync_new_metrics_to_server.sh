#!/usr/bin/env bash
# Push local new_metrics outputs → Salman server extras/output/ (rsync).
#
# Usage (from repo root):
#   export SERVER=madina@salmansserver.tail8a0d62.ts.net
#   ./extras/sync_new_metrics_to_server.sh
#
# One run only:
#   AGENT=maxpressure NETWORK=sumo4x4 RUN_NAME=seed42_steps3600 ./extras/sync_new_metrics_to_server.sh

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

if [[ -n "${AGENT}" && -n "${NETWORK}" && -n "${RUN_NAME}" ]]; then
  remote="${REMOTE_ROOT}/${AGENT}/${NETWORK}/${RUN_NAME}/"
  local="${LOCAL_ROOT}/${AGENT}/${NETWORK}/${RUN_NAME}/"
  echo "rsync ${local} -> ${SERVER}:${remote}"
  rsync "${RSYNC_FLAGS[@]}" "${local}" "${SERVER}:${remote}"
else
  echo "rsync ${LOCAL_ROOT}/ -> ${SERVER}:${REMOTE_ROOT}/"
  rsync "${RSYNC_FLAGS[@]}" "${LOCAL_ROOT}/" "${SERVER}:${REMOTE_ROOT}/"
fi

echo "Done."
