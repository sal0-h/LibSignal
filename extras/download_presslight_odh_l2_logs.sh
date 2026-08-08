#!/usr/bin/env bash
# Download PressLight OD-hub L2 4x4 experiment logs from gpujobs to Mac.
#
# Usage (Mac):
#   chmod +x extras/download_presslight_odh_l2_logs.sh
#   ./extras/download_presslight_odh_l2_logs.sh

set -euo pipefail

SERVER="${SERVER:-mmirzata@172.20.48.59}"
REMOTE_DIR="${REMOTE_DIR:-LibSignalFork}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_LOGS="${REMOTE_DIR}/data/output_data/tsc/sumo_presslight_odh_l2/sumo4x4"
LOCAL_LOGS="${REPO_ROOT}/data/output_data/tsc/sumo_presslight_odh_l2/sumo4x4"

mkdir -p "${LOCAL_LOGS}"

echo "Downloading ${SERVER}:${REMOTE_LOGS}/ -> ${LOCAL_LOGS}/"
scp -r "${SERVER}:${REMOTE_LOGS}/odh_l2_es/logger" "${LOCAL_LOGS}/odh_l2_es/" 2>/dev/null || true
scp -r "${SERVER}:${REMOTE_LOGS}/odh_l2/logger" "${LOCAL_LOGS}/odh_l2/" 2>/dev/null || true

echo "Local copies:"
ls -la "${LOCAL_LOGS}/odh_l2_es/logger/" 2>/dev/null || true
ls -la "${LOCAL_LOGS}/odh_l2/logger/" 2>/dev/null || true
