#!/usr/bin/env bash
# Submit Ingolstadt (sumo1x21) experiment matrix on deepnet / gpujobs.
#
# Job groups (5 methods: fixedtime, maxpressure, dqn, presslight, colight):
#   baseline   — default homo demand, axes off, early-stop (prefix: homo_1x21_es)
#   axes       — one realism axis on at a time × 5 axes (25 jobs)
#   l1         — OD-hub L1 (new OD, axes off; prefix: odh_l1_1x21_es)
#   l2         — OD-hub L2 (new OD + realism_full; prefix: odh_l2_1x21_es)
#   all        — baseline + axes + l1 + l2  (40 jobs)
#
# Usage:
#   export MCS_LABEL=crs-XXXX
#   ./extras/submit_ingolstadt_1x21_matrix.sh all
#   ./extras/submit_ingolstadt_1x21_matrix.sh baseline
#   ./extras/submit_ingolstadt_1x21_matrix.sh axes
#   ./extras/submit_ingolstadt_1x21_matrix.sh l1
#   ./extras/submit_ingolstadt_1x21_matrix.sh l2
#   DRY_RUN=1 ./extras/submit_ingolstadt_1x21_matrix.sh all   # list only
#
# Do NOT use sbatch --export=ALL on this cluster.
# Requires: feat/early-stop-episodes + Ingolstadt realism/OD assets on the server.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs extras/_slurm_generated

if [[ -z "${MCS_LABEL:-}" && "${DRY_RUN:-0}" != "1" ]]; then
  echo "Set MCS_LABEL first, e.g. export MCS_LABEL=crs-XXXX"
  exit 1
fi

MODE="${1:-all}"
SEED="${SEED:-42}"
NETWORK="${NETWORK:-sumo1x21}"
PREFIX_BASE="${PREFIX_BASE:-homo_1x21_es}"
PREFIX_L1="${PREFIX_L1:-odh_l1_1x21_es}"
PREFIX_L2="${PREFIX_L2:-odh_l2_1x21_es}"

METHODS=(fixedtime maxpressure dqn presslight colight)
AXES=(hetero slow_start crossing_proxy obs noise)

write_job() {
  local agent="$1"
  local prefix="$2"
  local tag="$3"
  local hours="${4:-48}"
  local out="extras/_slurm_generated/${tag}.sh"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN would submit agent=${agent} prefix=${prefix} tag=${tag}"
    return 0
  fi
  cat > "${out}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${tag}
#SBATCH --output=logs/${tag}_%j.out
#SBATCH --error=logs/${tag}_%j.err
#SBATCH --time=${hours}:00:00
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

if [[ "${agent}" == colight* ]]; then
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

submit_one() {
  local agent="$1"
  local prefix="$2"
  local tag="$3"
  local hours="${4:-48}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    write_job "${agent}" "${prefix}" "${tag}" "${hours}"
    return 0
  fi
  local script
  script="$(write_job "${agent}" "${prefix}" "${tag}" "${hours}")"
  sbatch --mcs-label="${MCS_LABEL}" "${script}"
}

submit_baseline() {
  local m
  echo "Submitting baseline (axes off): ${METHODS[*]}"
  for m in "${METHODS[@]}"; do
    submit_one "${m}" "${PREFIX_BASE}" "i21_base_${m}" 48
  done
}

submit_axes() {
  local m axis agent prefix
  echo "Submitting single-axis ablations: ${AXES[*]} × ${METHODS[*]}"
  for axis in "${AXES[@]}"; do
    for m in "${METHODS[@]}"; do
      agent="${m}_${axis}"
      prefix="axis_${axis}_1x21_es"
      submit_one "${agent}" "${prefix}" "i21_${axis}_${m}" 48
    done
  done
}

submit_l1() {
  local m
  echo "Submitting OD-hub L1 (OD only): ${METHODS[*]}"
  for m in "${METHODS[@]}"; do
    submit_one "${m}_odh_l1_1x21" "${PREFIX_L1}" "i21_odh_l1_${m}" 48
  done
}

submit_l2() {
  local m
  echo "Submitting OD-hub L2 (OD + realism_full): ${METHODS[*]}"
  for m in "${METHODS[@]}"; do
    submit_one "${m}_odh_l2_1x21" "${PREFIX_L2}" "i21_odh_l2_${m}" 48
  done
}

case "$MODE" in
  baseline) submit_baseline ;;
  axes) submit_axes ;;
  l1) submit_l1 ;;
  l2) submit_l2 ;;
  all)
    submit_baseline
    submit_axes
    submit_l1
    submit_l2
    ;;
  *)
    echo "Usage: $0 {all|baseline|axes|l1|l2}"
    exit 1
    ;;
esac

echo "Done. Monitor: squeue -u \$USER"
echo "Outputs under data/output_data/tsc/sumo_*_${NETWORK}/..."
