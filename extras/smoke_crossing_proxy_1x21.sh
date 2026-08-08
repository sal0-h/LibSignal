#!/usr/bin/env bash
# Quick smoke: fixedtime_crossing_proxy on sumo1x21 (~2 min).
# Fails fast if Ingolstadt routed demand or crossing_proxy route fix is broken.
#
# Usage (gpujobs):
#   cd ~/LibSignalFork
#   bash extras/smoke_crossing_proxy_1x21.sh

set -euo pipefail
cd "$(dirname "$0")/.."

if ! grep -q 'or self.crossing_proxy' world/world_sumo.py; then
  echo "ERROR: world/world_sumo.py missing crossing_proxy route fix." >&2
  echo "From Mac: ./extras/sync_ingolstadt_to_gpujobs.sh" >&2
  exit 1
fi

if [[ ! -f data/raw_data/ingolstadt21/ingolstadt21_routed.rou.xml ]]; then
  echo "Generating Ingolstadt assets..."
  bash extras/prepare_ingolstadt_1x21_assets.sh
fi

CONDA_PREFIX="${CONDA_PREFIX:-/data1/mmirzata/.conda/envs/libsignal}"
export SUMO_HOME="${SUMO_HOME:-${CONDA_PREFIX}/share/sumo}"
export PATH="${CONDA_PREFIX}/bin:${SUMO_HOME}/bin:${PATH}"

echo "SUMO_HOME=${SUMO_HOME}"
echo "Smoke: fixedtime_crossing_proxy / sumo1x21 (expect [CrossingProxy] Ingolstadt routed demand)"

python run.py \
  -a fixedtime_crossing_proxy \
  -w sumo \
  -n sumo1x21 \
  --seed 42 \
  --ngpu -1 \
  --interface libsumo \
  --prefix smoke_cp_1x21

echo "OK — crossing_proxy smoke passed."
