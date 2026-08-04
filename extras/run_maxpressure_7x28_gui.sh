#!/usr/bin/env bash
# MaxPressure + SUMO-GUI on 7x28 Manhattan grid (~392 signals — VERY slow in GUI).
#
# Run on your Mac (needs display). Start with 600 sim-seconds; increase in
# configs/tsc/maxpressure_gui.yml if needed.
#
#   ./extras/run_maxpressure_7x28_gui.sh

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

INTERFACE=traci
python -c "import libsumo" 2>/dev/null && INTERFACE=libsumo

echo "WARNING: 7x28 has ~392 intersections — still CPU-heavy."
echo "Using gui_step_length=10 (10 sim-seconds per step). In SUMO-GUI set Delay(ms)=0."
echo "SUMO_HOME=${SUMO_HOME:-unset}  interface=${INTERFACE}"

python run.py \
  --agent maxpressure_gui \
  --world sumo \
  --network sumo7x28_gui \
  --interface "${INTERFACE}" \
  --prefix gui_view
