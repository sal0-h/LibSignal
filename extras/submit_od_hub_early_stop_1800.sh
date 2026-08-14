#!/usr/bin/env bash
# Submit OD-hub 1800s early-stop runs on sumo4x4 (L1 axes-off + L2 realism_full).
#
# 10 jobs:
#   L1 (odh_l1_es): fixedtime, maxpressure, dqn, presslight, colight
#   L2 (odh_l2_es): fixedtime, maxpressure, dqn, presslight, colight
#
# Usage:
#   export MCS_LABEL=crs-XXXX
#   ./extras/submit_od_hub_early_stop_1800.sh all
#   ./extras/submit_od_hub_early_stop_1800.sh l1
#   ./extras/submit_od_hub_early_stop_1800.sh l2
#
# Requires branch feat/early-stop-episodes (cycle-mean median-wait early-stop).

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs extras/_slurm_generated

if [[ -z "${MCS_LABEL:-}" ]]; then
  echo "Set MCS_LABEL first, e.g. export MCS_LABEL=crs-XXXX"
  exit 1
fi

MODE="${1:-all}"
SEED="${SEED:-42}"
NETWORK="${NETWORK:-sumo4x4}"
PREFIX_L1="${PREFIX_L1:-odh_l1_es}"
PREFIX_L2="${PREFIX_L2:-odh_l2_es}"

L1_AGENTS=(fixedtime_odh_l1 maxpressure_odh_l1 dqn_odh_l1 presslight_odh_l1 colight_odh_l1)
L2_AGENTS=(fixedtime_odh_l2 maxpressure_odh_l2 dqn_odh_l2 presslight_odh_l2 colight_odh_l2)

write_job() {
  local agent="$1"
  local prefix="$2"
  local tag="${agent}"
  local out="extras/_slurm_generated/${tag}_${prefix}.sh"
  cat > "${out}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${tag}
#SBATCH --output=logs/${tag}_%j.out
#SBATCH --error=logs/${tag}_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
cd ~/LibSignalFork

CONDA_PREFIX="/data1/mmirzata/.conda/envs/libsignal"
export SUMO_HOME="\${CONDA_PREFIX}/share/sumo"
export PATH="\${CONDA_PREFIX}/bin:\${SUMO_HOME}/bin:\${PATH}"

echo "Host: \$(hostname)"
echo "Agent: ${agent}  Network: ${NETWORK}  Seed: ${SEED}  Prefix: ${prefix}"
echo "OD-hub early-stop: cycle_mean median wait, max=500 min=30 patience=3, 5% or 1s"

if [[ "${agent}" == colight_* ]]; then
  python -c "import torch_scatter" 2>/dev/null || {
    TV="\$(python -c 'import torch; print(torch.__version__.split("+")[0])')"
    pip install torch_scatter -f "https://data.pyg.org/whl/torch-\${TV}.html"
  }
fi

python run.py \\
  -a ${agent} \\
  -w sumo \\
  -n ${NETWORK} \\
  --seed ${SEED} \\
  --ngpu -1 \\
  --interface libsumo \\
  --prefix ${prefix}
EOF
  chmod +x "${out}"
  echo "${out}"
}

submit_list() {
  local prefix="$1"
  shift
  local agent
  for agent in "$@"; do
    script="$(write_job "${agent}" "${prefix}")"
    sbatch --mcs-label="${MCS_LABEL}" "${script}"
  done
}

case "$MODE" in
  l1)
    echo "Submitting L1 (axes off, new OD): ${L1_AGENTS[*]}"
    submit_list "${PREFIX_L1}" "${L1_AGENTS[@]}"
    ;;
  l2)
    echo "Submitting L2 (realism_full + new OD): ${L2_AGENTS[*]}"
    submit_list "${PREFIX_L2}" "${L2_AGENTS[@]}"
    ;;
  all)
    echo "Submitting all 10 OD-hub early-stop jobs"
    submit_list "${PREFIX_L1}" "${L1_AGENTS[@]}"
    submit_list "${PREFIX_L2}" "${L2_AGENTS[@]}"
    ;;
  *)
    echo "Usage: $0 {all|l1|l2}"
    exit 1
    ;;
esac

echo "Done. Monitor: squeue -u \$USER"
echo "Primary metric: HELDOUT_MEAN in logger BRF/DTL under prefix ${PREFIX_L1} / ${PREFIX_L2}"
