#!/usr/bin/env bash
# Baseline early-stop protocol on sumo4x4 (grid4x4).
#
# Agents:
#   maxpressure, fixedtime  — classical baselines (1 eval episode)
#   dqn, presslight, colight, ppo_pfrl (IPPO) — RL with min=20 / max=2000 / patience=20
#
# Usage (local sequential):
#   ./extras/run_baseline_early_stop_4x4.sh              # all 6
#   ./extras/run_baseline_early_stop_4x4.sh baselines    # FT + MP only
#   ./extras/run_baseline_early_stop_4x4.sh rl           # DQN + PressLight + CoLight + IPPO
#   ./extras/run_baseline_early_stop_4x4.sh dqn          # single agent
#
# Env overrides:
#   SEED=42 PREFIX=baseline_early_stop NGPU=-1 NETWORK=sumo4x4 INTERFACE=libsumo

set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-all}"
NETWORK="${NETWORK:-sumo4x4}"
SEED="${SEED:-42}"
PREFIX="${PREFIX:-baseline_early_stop}"
NGPU="${NGPU:--1}"
INTERFACE="${INTERFACE:-libsumo}"
CHECK_ONLY=0
if [[ "${MODE}" == "--check" || "${MODE}" == "check" ]]; then
  CHECK_ONLY=1
  MODE="${2:-all}"
fi

# Prefer libsignal (has torch_scatter for CoLight), then traffic, then .venv.
if [[ -f /opt/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /opt/miniconda3/etc/profile.d/conda.sh
  if conda env list | grep -qE '^libsignal\s'; then
    conda activate libsignal
  elif conda env list | grep -qE '^traffic\s'; then
    conda activate traffic
  fi
elif [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if [[ -z "${SUMO_HOME:-}" ]]; then
  if [[ -d "$(brew --prefix sumo 2>/dev/null)/share/sumo" ]]; then
    export SUMO_HOME="$(brew --prefix sumo)/share/sumo"
  else
    export SUMO_HOME="$(python -c 'import os,sumo; print(os.path.dirname(sumo.__file__))' 2>/dev/null || true)"
  fi
fi

BASELINES=(maxpressure fixedtime)
RL_AGENTS=(dqn presslight colight ppo_pfrl)

pick_agents() {
  case "$MODE" in
    all) echo "${BASELINES[*]} ${RL_AGENTS[*]}" ;;
    baselines) echo "${BASELINES[*]}" ;;
    rl) echo "${RL_AGENTS[*]}" ;;
    maxpressure|fixedtime|dqn|presslight|colight|ppo_pfrl) echo "$MODE" ;;
    *)
      echo "Usage: $0 {all|baselines|rl|maxpressure|fixedtime|dqn|presslight|colight|ppo_pfrl}" >&2
      exit 1
      ;;
  esac
}

AGENTS=($(pick_agents))

echo "========== Baseline early-stop readiness =========="
echo "network=${NETWORK} seed=${SEED} prefix=${PREFIX} interface=${INTERFACE}"
echo "budget: min_episodes=20 max_episodes=2000 patience=20 metric=test"
echo "agents: ${AGENTS[*]}"
echo "python: $(python -c 'import sys; print(sys.executable)')"
echo "SUMO_HOME=${SUMO_HOME:-UNSET}"

MISSING=0
for agent in "${AGENTS[@]}"; do
  case "$agent" in
    colight)
      if ! python -c "import torch_scatter" >/dev/null 2>&1; then
        echo "[NOT READY] colight — torch_scatter missing in this env"
        echo "  fix: conda activate libsignal  (or pip install torch_scatter for your torch)"
        MISSING=1
      else
        echo "[ready] colight (torch_scatter ok)"
      fi
      ;;
    ppo_pfrl)
      if ! python -c "import pfrl, torch" >/dev/null 2>&1; then
        echo "[NOT READY] ppo_pfrl (IPPO) — pfrl/torch missing"
        MISSING=1
      else
        echo "[ready] ppo_pfrl (IPPO)"
      fi
      ;;
    dqn|presslight)
      if ! python -c "import torch" >/dev/null 2>&1; then
        echo "[NOT READY] ${agent} — torch missing"
        MISSING=1
      else
        echo "[ready] ${agent}"
      fi
      ;;
    maxpressure|fixedtime)
      echo "[ready] ${agent} (classical baseline)"
      ;;
  esac
done

if [[ ! -f "configs/sim/${NETWORK}.cfg" ]]; then
  echo "[NOT READY] missing configs/sim/${NETWORK}.cfg"
  MISSING=1
fi

if [[ -z "${SUMO_HOME:-}" || ! -d "${SUMO_HOME}" ]]; then
  echo "[NOT READY] SUMO_HOME unset or invalid"
  MISSING=1
fi

if [[ "$MISSING" -ne 0 ]]; then
  echo "Fix the issues above, then re-run."
  exit 1
fi

echo "All selected agents ready."
echo "===================================================="

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  echo "Check-only mode: not starting runs."
  exit 0
fi

LOGDIR="data/output_data/tsc/baseline_early_stop_runs"
mkdir -p "${LOGDIR}"

run_one() {
  local agent="$1"
  local outfile="${LOGDIR}/${agent}_${NETWORK}_seed${SEED}.log"
  echo "========== ${agent} =========="
  echo "logging to ${outfile}"
  python run.py \
    --agent "${agent}" \
    --world sumo \
    --network "${NETWORK}" \
    --seed "${SEED}" \
    --prefix "${PREFIX}" \
    --ngpu "${NGPU}" \
    --interface "${INTERFACE}" \
    2>&1 | tee "${outfile}"
}

for agent in "${AGENTS[@]}"; do
  run_one "${agent}"
done

echo "Done. Per-agent logs: ${LOGDIR}/"
echo "LibSignal outputs under data/output_data/tsc/sumo_<agent>_${PREFIX}/${NETWORK}/${PREFIX}/logger/"
