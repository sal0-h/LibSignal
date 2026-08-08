#!/usr/bin/env bash
# Push Ingolstadt 1x21 fixes to gpujobs (run from your Mac, repo root).
#
# Usage:
#   chmod +x extras/sync_ingolstadt_to_gpujobs.sh
#   ./extras/sync_ingolstadt_to_gpujobs.sh
#
# Then on gpujobs:
#   bash extras/smoke_crossing_proxy_1x21.sh
#   export MCS_LABEL=15288 && ./extras/resubmit_ingolstadt_axes_remainder.sh

set -euo pipefail

SERVER="${SERVER:-mmirzata@172.20.48.59}"
REMOTE_DIR="${REMOTE_DIR:-LibSignalFork}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Syncing Ingolstadt fixes -> ${SERVER}:~/${REMOTE_DIR}/"

FILES=(
  world/world_sumo.py
  configs/sim/sumo1x21.cfg
  data/raw_data/ingolstadt21/ingolstadt21.sumocfg
  extras/submit_ingolstadt_1x21_chained.sh
  extras/resubmit_ingolstadt_axes_remainder.sh
  extras/smoke_crossing_proxy_1x21.sh
  extras/prepare_ingolstadt_1x21_assets.sh
)

for f in "${FILES[@]}"; do
  echo "  ${f}"
  scp "${REPO_ROOT}/${f}" "${SERVER}:${REMOTE_DIR}/${f}"
done

echo ""
echo "Synced. On gpujobs:"
echo "  ssh ${SERVER}"
echo "  cd ~/${REMOTE_DIR}"
echo "  bash extras/smoke_crossing_proxy_1x21.sh"
echo "  export MCS_LABEL=15288 && ./extras/resubmit_ingolstadt_axes_remainder.sh"
