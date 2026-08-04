#!/usr/bin/env bash
# Copy extras/ to gpujobs from your Mac (run locally, NOT on the server).
#
# Usage:
#   chmod +x extras/copy_to_server.sh
#   ./extras/copy_to_server.sh
#
# Optional env vars:
#   SERVER=mmirzata@172.20.48.59
#   REMOTE_DIR=~/LibSignalFork

set -euo pipefail

SERVER="${SERVER:-mmirzata@172.20.48.59}"
REMOTE_DIR="${REMOTE_DIR:-LibSignalFork}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Copying extras/ -> ${SERVER}:${REMOTE_DIR}/extras/"
ssh "${SERVER}" "mkdir -p ${REMOTE_DIR}/extras"
scp -r "${REPO_ROOT}/extras/"* "${SERVER}:${REMOTE_DIR}/extras/"

echo ""
echo "Copied. On the server, run:"
echo "  ssh ${SERVER}"
echo "  conda activate libsignal"
echo "  cd ~/${REMOTE_DIR}"
echo "  python extras/run_vehicle_wait_logs.py --agent maxpressure --network sumo4x4 --seed 42 --test-steps 3600"
echo "  # -> extras/output/maxpressure/sumo4x4/seed42_steps3600/vehicle_waiting_times.csv"
