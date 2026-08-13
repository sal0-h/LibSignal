#!/usr/bin/env bash
# Submit M0 journal cell: movie demand · all realism axes OFF · 3600s · 200 eps.
#
# Agents: FixedTime, MaxPressure, DQN, PressLight, CoLight
# Networks: sumo4x4, sumo1x21 (Ingolstadt)
#
# Submit from the gpujobs LOGIN node (same as other LibSignal TSC jobs):
#
#   cd ~/LibSignalFork
#   git pull   # ensure master has trip-metrics + this script
#   export MCS_LABEL=crs-XXXX          # or QSIURP_Salman
#   ./extras/submit_m0_homo.sh all
#
# Modes:
#   ./extras/submit_m0_homo.sh all         # 10 jobs (5 agents × 2 nets)
#   ./extras/submit_m0_homo.sh baselines   # FT + MP × both nets (4 jobs)
#   ./extras/submit_m0_homo.sh rl          # DQN + PressLight + CoLight × both nets
#   ./extras/submit_m0_homo.sh 4x4         # all 5 agents on sumo4x4 only
#   ./extras/submit_m0_homo.sh 1x21        # all 5 agents on sumo1x21 only
#
# Optional env:
#   SEED=42 PREFIX=m0_homo CONDA_ENV=libsignal REPO_DIR=$HOME/LibSignalFork
#   TIME_BASELINE=04:00:00 TIME_RL=48:00:00
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/m0_homo_<jobid>.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-${HOME}/LibSignalFork}"
cd "${REPO_DIR}"
mkdir -p logs

if [[ -z "${MCS_LABEL:-}" ]]; then
  echo "Set MCS_LABEL first, e.g.:"
  echo "  export MCS_LABEL=crs-XXXX"
  echo "  # or: export MCS_LABEL=QSIURP_Salman"
  exit 1
fi

MODE="${1:-all}"
SEED="${SEED:-42}"
PREFIX="${PREFIX:-m0_homo}"
CONDA_ENV="${CONDA_ENV:-libsignal}"
TIME_BASELINE="${TIME_BASELINE:-04:00:00}"
TIME_RL="${TIME_RL:-48:00:00}"

BASELINES=(fixedtime maxpressure)
RL=(dqn presslight colight)

is_rl() {
  case "$1" in
    dqn|presslight|colight) return 0 ;;
    *) return 1 ;;
  esac
}

submit_one() {
  local agent="$1"
  local network="$2"
  local wall
  if is_rl "${agent}"; then
    wall="${TIME_RL}"
  else
    wall="${TIME_BASELINE}"
  fi
  local job_name="m0_${agent}_${network}"
  echo "sbatch ${job_name}  (time=${wall}, prefix=${PREFIX})"
  # Export only what the job needs. Do NOT use --export=ALL (login CONDA_PREFIX
  # is often /opt/anaconda3, which is missing on compute nodes → exit 127).
  sbatch \
    --job-name="${job_name}" \
    --time="${wall}" \
    --export="ALL,AGENT=${agent},NETWORK=${network},SEED=${SEED},PREFIX=${PREFIX},CONDA_ENV=${CONDA_ENV},REPO_DIR=${REPO_DIR},CONDA_PREFIX_OVERRIDE=/data1/mmirzata/.conda/envs/${CONDA_ENV}" \
    --mcs-label="${MCS_LABEL}" \
    extras/slurm_m0_homo.sh
}

submit_net() {
  local network="$1"
  shift
  local agents=("$@")
  for agent in "${agents[@]}"; do
    submit_one "${agent}" "${network}"
  done
}

ALL_AGENTS=("${BASELINES[@]}" "${RL[@]}")

case "${MODE}" in
  all)
    submit_net sumo4x4 "${ALL_AGENTS[@]}"
    submit_net sumo1x21 "${ALL_AGENTS[@]}"
    ;;
  baselines)
    submit_net sumo4x4 "${BASELINES[@]}"
    submit_net sumo1x21 "${BASELINES[@]}"
    ;;
  rl)
    submit_net sumo4x4 "${RL[@]}"
    submit_net sumo1x21 "${RL[@]}"
    ;;
  4x4)
    submit_net sumo4x4 "${ALL_AGENTS[@]}"
    ;;
  1x21|ingolstadt)
    submit_net sumo1x21 "${ALL_AGENTS[@]}"
    ;;
  *)
    echo "Usage: $0 {all|baselines|rl|4x4|1x21}"
    exit 1
    ;;
esac

echo ""
echo "Submitted M0 (axes OFF / homo). Monitor: squeue -u \$USER"
echo "Outputs: data/output_data/tsc/sumo_<agent>/<network>/${PREFIX}/logger/"
echo "  new_metrics.csv (+ new_metrics_best.csv for RL)"
