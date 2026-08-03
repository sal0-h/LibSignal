#!/usr/bin/env bash
# Submit baseline early-stop 4x4 jobs on gpujobs (login node).
#
# Usage:
#   export MCS_LABEL=crs-XXXX
#   ./extras/submit_baseline_early_stop_4x4.sh              # all 5
#   ./extras/submit_baseline_early_stop_4x4.sh baselines    # FT + MP
#   ./extras/submit_baseline_early_stop_4x4.sh rl           # DQN + PressLight + CoLight
#   ./extras/submit_baseline_early_stop_4x4.sh dqn          # single agent

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

if [[ -z "${MCS_LABEL:-}" ]]; then
  echo "Set MCS_LABEL first, e.g. export MCS_LABEL=crs-XXXX"
  exit 1
fi

MODE="${1:-all}"
NETWORK="${NETWORK:-sumo4x4}"
SEED="${SEED:-42}"
PREFIX="${PREFIX:-baseline_early_stop}"
SCRIPT="extras/slurm_baseline_early_stop_4x4.sh"
chmod +x "${SCRIPT}" extras/run_baseline_early_stop_4x4.sh 2>/dev/null || true

BASELINES=(maxpressure fixedtime)
RL_AGENTS=(dqn presslight colight)

pick_agents() {
  case "$MODE" in
    all) echo "${BASELINES[*]} ${RL_AGENTS[*]}" ;;
    baselines) echo "${BASELINES[*]}" ;;
    rl) echo "${RL_AGENTS[*]}" ;;
    maxpressure|fixedtime|dqn|presslight|colight) echo "$MODE" ;;
    *)
      echo "Usage: $0 {all|baselines|rl|maxpressure|fixedtime|dqn|presslight|colight}" >&2
      exit 1
      ;;
  esac
}

AGENTS=($(pick_agents))
echo "Submitting agents: ${AGENTS[*]}"
echo "network=${NETWORK} seed=${SEED} prefix=${PREFIX}"

for agent in "${AGENTS[@]}"; do
  sbatch \
    --mcs-label="${MCS_LABEL}" \
    --job-name="bes_${agent}_4x4" \
    --export=ALL,AGENT="${agent}",NETWORK="${NETWORK}",SEED="${SEED}",PREFIX="${PREFIX}" \
    "${SCRIPT}"
done

echo "Submitted ${#AGENTS[@]} job(s). Monitor: squeue -u \$USER"
echo "Outputs: data/output_data/tsc/sumo_<agent>_${PREFIX}/${NETWORK}/${PREFIX}/logger/"
