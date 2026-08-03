#!/usr/bin/env bash
# Train all supported RL agents with the standard early-stop episode budget.
# Usage:
#   ./extras/run_all_rl_early_stop.sh [network] [seed] [prefix]
# Defaults: sumo1x1, seed 42, prefix early_stop

set -euo pipefail
cd "$(dirname "$0")/.."

NETWORK="${1:-sumo1x1}"
SEED="${2:-42}"
PREFIX="${3:-early_stop}"
NGPU="${NGPU:--1}"

# Activate local venv if present (Cursor Cloud / setup.sh layouts).
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate traffic 2>/dev/null || true
fi

if [[ -z "${SUMO_HOME:-}" ]]; then
  export SUMO_HOME="$(python -c 'import os,sumo; print(os.path.dirname(sumo.__file__))')"
fi

# CoLight needs torch_scatter; skip if unavailable.
AGENTS=(dqn presslight frap mplight maddpg_v2 magd ppo_pfrl)
if python -c "import torch_scatter" >/dev/null 2>&1; then
  AGENTS+=(colight)
else
  echo "[skip] colight (torch_scatter not installed)"
fi

# ppo (non-pfrl) has a stub train(); skip by default.
echo "Network=${NETWORK} seed=${SEED} prefix=${PREFIX}"
echo "Agents: ${AGENTS[*]}"
echo "Budget: min_episodes/max_episodes/patience from configs/tsc/base.yml"

LOGDIR="data/output_data/tsc/rl_early_stop_runs"
mkdir -p "${LOGDIR}"

for agent in "${AGENTS[@]}"; do
  echo "========== ${agent} =========="
  outfile="${LOGDIR}/${agent}_${NETWORK}_seed${SEED}.log"
  python run.py \
    --agent "${agent}" \
    --world sumo \
    --network "${NETWORK}" \
    --seed "${SEED}" \
    --prefix "${PREFIX}" \
    --ngpu "${NGPU}" \
    2>&1 | tee "${outfile}"
done

echo "All RL runs finished. Logs in ${LOGDIR}/"
