#!/usr/bin/env bash
# Generate Ingolstadt (sumo1x21) realism + OD-hub assets.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${SUMO_HOME:-}" ]]; then
  SUMO_HOME="$(python -c 'import os,sumo; print(os.path.dirname(sumo.__file__))' 2>/dev/null || true)"
fi
if [[ -z "${SUMO_HOME:-}" ]]; then
  echo "Set SUMO_HOME"; exit 1
fi
export SUMO_HOME
export PATH="${SUMO_HOME}/bin:${PATH}"
echo "SUMO_HOME=${SUMO_HOME}"

# Trip file → vehicles with fixed routes (needed for crossing_proxy lane halts)
duarouter \
  -n data/raw_data/ingolstadt21/ingolstadt21.net.xml \
  -r data/raw_data/ingolstadt21/ingolstadt21.rou.xml \
  -o data/raw_data/ingolstadt21/ingolstadt21_routed.rou.xml \
  --ignore-errors true --repair true --no-warnings true

python extras/gen_hetero_routes.py
python extras/gen_slow_start_routes.py
python extras/gen_crossing_proxy_lanes.py --network ingolstadt21
python extras/gen_ingolstadt21_taz.py
python extras/gen_od_hub_demand_ingolstadt21.py
echo "Assets ready."
