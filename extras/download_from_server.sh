#!/usr/bin/env bash
# Download vehicle wait logs from gpujobs to local extras/output/.
#
# Usage (from repo root on your Mac):
#   chmod +x extras/download_from_server.sh
#   ./extras/download_from_server.sh
#
# Optional:
#   SERVER=mmirzata@172.20.48.59
#   NETWORK=sumo7x28
#   RUN_NAME=seed42_steps3600

set -euo pipefail

SERVER="${SERVER:-mmirzata@172.20.48.59}"
AGENT="${AGENT:-maxpressure}"
NETWORK="${NETWORK:-sumo7x28}"
RUN_NAME="${RUN_NAME:-seed42_steps3600}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_DIR="LibSignalFork/extras/output/${AGENT}/${NETWORK}/${RUN_NAME}"
LOCAL_DIR="${REPO_ROOT}/extras/output/${AGENT}/${NETWORK}/${RUN_NAME}"

mkdir -p "${LOCAL_DIR}"

echo "Downloading ${SERVER}:${REMOTE_DIR}/ -> ${LOCAL_DIR}/"
scp "${SERVER}:${REMOTE_DIR}/vehicle_waiting_times.csv" "${LOCAL_DIR}/"
scp "${SERVER}:${REMOTE_DIR}/vehicle_waiting_times_meta.json" "${LOCAL_DIR}/"

echo ""
echo "Saved locally:"
echo "  ${LOCAL_DIR}/vehicle_waiting_times.csv"
echo "  ${LOCAL_DIR}/vehicle_waiting_times_meta.json"
echo ""
echo "Downloaded. Plot in your own Jupyter notebook using custom_wait_s (see extras/README.md)."
