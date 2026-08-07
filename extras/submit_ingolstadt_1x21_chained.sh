#!/usr/bin/env bash
# Submit Ingolstadt (sumo1x21) as 4 independent SLURM jobs (fits QOS 4-job limit).
#
# Jobs run in parallel when GPUs are free. Within each job, methods run sequentially:
#   1) baseline  — 5 methods, homo, FIXED 200 episodes (*_e200)
#   2) axes      — 25 runs (5 axes × 5 methods), FIXED 200 episodes
#   3) l1        — 5 methods, OD-hub only, ADAPTIVE held-out early-stop
#   4) l2        — 5 methods, OD + realism_full, ADAPTIVE held-out early-stop
#
# Usage:
#   ./extras/submit_ingolstadt_1x21_chained.sh
#   GROUPS=axes,l1,l2 ./extras/submit_ingolstadt_1x21_chained.sh   # skip baseline
#   DRY_RUN=1 ./extras/submit_ingolstadt_1x21_chained.sh
#
# Do NOT use sbatch --export=ALL on this cluster.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs extras/_slurm_generated

SEED="${SEED:-42}"
NETWORK="${NETWORK:-sumo1x21}"
MCS_LABEL="${MCS_LABEL:-15288}"
PREFIX_BASE="${PREFIX_BASE:-homo_1x21_e200}"
PREFIX_L1="${PREFIX_L1:-odh_l1_1x21_es}"
PREFIX_L2="${PREFIX_L2:-odh_l2_1x21_es}"

# gpu2 MaxTime is typically 48h
HOURS_BASELINE="${HOURS_BASELINE:-48}"
HOURS_AXES="${HOURS_AXES:-48}"
HOURS_L1="${HOURS_L1:-48}"
HOURS_L2="${HOURS_L2:-48}"

METHODS_CLASSICAL=(fixedtime maxpressure)
METHODS_RL=(dqn presslight colight)
METHODS=(fixedtime maxpressure dqn presslight colight)
AXES=(hetero slow_start crossing_proxy obs noise)

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  echo "Using MCS_LABEL=${MCS_LABEL}"
fi

run_one_py() {
  cat <<'EOS'
run_one() {
  local agent="$1"
  local prefix="$2"
  echo "===== START agent=${agent} prefix=${prefix} $(date -Is) ====="
  if [[ "${agent}" == colight* ]]; then
    python -c "import torch_scatter" 2>/dev/null || {
      TV="$(python -c 'import torch; print(torch.__version__.split("+")[0])')"
      pip install torch_scatter -f "https://data.pyg.org/whl/torch-${TV}.html"
    }
  fi
  python run.py \
    -a "${agent}" \
    -w sumo \
    -n "${NETWORK}" \
    --seed "${SEED}" \
    --ngpu -1 \
    --interface libsumo \
    --prefix "${prefix}"
  echo "===== DONE  agent=${agent} prefix=${prefix} $(date -Is) ====="
}
EOS
}

write_group_job() {
  local group="$1"
  local hours="$2"
  local out="extras/_slurm_generated/i21_group_${group}.sh"
  local body=""

  case "${group}" in
    baseline)
      local m
      # Classical: 1-ep configs. RL: fixed 200-ep.
      for m in "${METHODS_CLASSICAL[@]}"; do
        body+="run_one ${m} ${PREFIX_BASE}"$'\n'
      done
      for m in "${METHODS_RL[@]}"; do
        body+="run_one ${m}_e200 ${PREFIX_BASE}"$'\n'
      done
      ;;
    axes)
      local axis m
      for axis in "${AXES[@]}"; do
        for m in "${METHODS_CLASSICAL[@]}"; do
          body+="run_one ${m}_${axis} axis_${axis}_1x21_e200"$'\n'
        done
        for m in "${METHODS_RL[@]}"; do
          body+="run_one ${m}_${axis}_e200 axis_${axis}_1x21_e200"$'\n'
        done
      done
      ;;
    l1)
      local m
      for m in "${METHODS[@]}"; do
        body+="run_one ${m}_odh_l1_1x21 ${PREFIX_L1}"$'\n'
      done
      ;;
    l2)
      local m
      for m in "${METHODS[@]}"; do
        body+="run_one ${m}_odh_l2_1x21 ${PREFIX_L2}"$'\n'
      done
      ;;
    *)
      echo "unknown group: ${group}" >&2
      exit 1
      ;;
  esac

  cat > "${out}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=i21_${group}
#SBATCH --output=logs/i21_group_${group}_%j.out
#SBATCH --error=logs/i21_group_${group}_%j.err
#SBATCH --time=${hours}:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
cd ~/LibSignalFork

NETWORK="${NETWORK}"
SEED="${SEED}"

CONDA_PREFIX="/data1/mmirzata/.conda/envs/libsignal"
export SUMO_HOME="\${CONDA_PREFIX}/share/sumo"
export PATH="\${CONDA_PREFIX}/bin:\${SUMO_HOME}/bin:\${PATH}"

echo "Host: \$(hostname)"
echo "Group: ${group}  Network: \${NETWORK}  Seed: \${SEED}"
echo "Start: \$(date -Is)"

$(run_one_py)

${body}
echo "Group ${group} finished: \$(date -Is)"
EOF
  chmod +x "${out}"
  echo "${out}"
}

hours_for() {
  case "$1" in
    baseline) echo "${HOURS_BASELINE}" ;;
    axes) echo "${HOURS_AXES}" ;;
    l1) echo "${HOURS_L1}" ;;
    l2) echo "${HOURS_L2}" ;;
    *) echo "unknown group: $1" >&2; exit 1 ;;
  esac
}

submit_groups() {
  local groups_csv="${GROUPS:-baseline,axes,l1,l2}"
  local group hours script jid
  IFS=',' read -r -a groups <<< "${groups_csv}"

  for group in "${groups[@]}"; do
    group="$(echo "${group}" | tr -d '[:space:]')"
    [[ -n "${group}" ]] || continue
    hours="$(hours_for "${group}")"
    script="$(write_group_job "${group}" "${hours}")"

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      echo "DRY_RUN: sbatch --mcs-label=${MCS_LABEL} ${script}  # group=${group} hours=${hours}"
      continue
    fi

    jid="$(sbatch --mcs-label="${MCS_LABEL}" "${script}" | awk '{print $NF}')"
    echo "Submitted group=${group} job_id=${jid} hours=${hours} (independent)"
  done
}

echo "Ingolstadt submit: 4 independent jobs (methods sequential inside each)"
echo "network=${NETWORK} seed=${SEED} groups=${GROUPS:-baseline,axes,l1,l2}"
echo "prefixes: base=${PREFIX_BASE} l1=${PREFIX_L1} l2=${PREFIX_L2}"
submit_groups
echo "Monitor: squeue -u \$USER"
echo "Logs: logs/i21_group_{baseline,axes,l1,l2}_<jid>.out"
