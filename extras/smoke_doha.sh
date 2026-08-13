#!/usr/bin/env bash
# Smoke: SUMO-only, then LibSignal FixedTime and MaxPressure on sumo_doha.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export SUMO_HOME="${SUMO_HOME:-$(python3 -c 'import os,sumo; print(os.path.dirname(sumo.__file__))' 2>/dev/null || true)}"
: "${SUMO_HOME:?set SUMO_HOME}"
export PATH="${SUMO_HOME}/bin:${PATH}"

echo "== SUMO-only 600s =="
sumo -c data/raw_data/doha_corniche/doha_corniche.sumocfg --end 600 \
  --duration-log.statistics --no-step-log

echo "== TLS validation =="
python extras/validate_sumo_tls.py --net data/raw_data/doha_corniche/doha_corniche.net.xml

echo "== FixedTime smoke =="
python run.py -a fixedtime_doha_smoke -w sumo -n sumo_doha --seed 42 --ngpu -1 --prefix doha_ft_smoke

echo "== MaxPressure smoke =="
python run.py -a maxpressure_doha_smoke -w sumo -n sumo_doha --seed 42 --ngpu -1 --prefix doha_mp_smoke

echo "OK"
