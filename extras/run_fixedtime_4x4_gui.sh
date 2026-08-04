#!/usr/bin/env bash
# Open SUMO-GUI with fixed-time control on the 4x4 grid (sumo4x4 / grid4x4 data).
#
# Usage (repo root, display required — run on your Mac desktop, not headless server):
#   chmod +x extras/run_fixedtime_4x4_gui.sh
#   ./extras/run_fixedtime_4x4_gui.sh
#
# In SUMO-GUI: use Play, Delay slider, zoom/pan.

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${SUMO_HOME:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/share/sumo" ]]; then
    export SUMO_HOME="${CONDA_PREFIX}/share/sumo"
  elif [[ -d "/opt/homebrew/opt/sumo/share/sumo" ]]; then
    export SUMO_HOME="/opt/homebrew/opt/sumo/share/sumo"
  fi
fi
export PATH="${SUMO_HOME:-}/bin:${PATH}"

INTERFACE="${INTERFACE:-traci}"
if python -c "import libsumo" 2>/dev/null; then
  INTERFACE="${INTERFACE:-libsumo}"
fi

echo "SUMO_HOME=${SUMO_HOME:-unset}"
echo "interface=${INTERFACE}"
echo "Running fixed-time on grid4x4 with GUI (900 sim-seconds)..."

python run.py \
  --agent fixedtime_gui \
  --world sumo \
  --network sumo4x4_gui \
  --interface "${INTERFACE}" \
  --prefix gui_view
