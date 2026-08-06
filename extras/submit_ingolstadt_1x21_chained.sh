#!/usr/bin/env bash
# Submit Ingolstadt (sumo1x21) as 4 chained SLURM jobs on deepnet.
#
# Order (each waits for the previous with --dependency=afterok):
#   1) baseline  — 5 methods, homo demand, axes off
#   2) axes      — 25 runs (5 axes × 5 methods), one at a time
#   3) l1        — 5 methods, OD-hub only
#   4) l2        — 5 methods, OD-hub + realism_full
#
# Inside each job, runs are sequential (set -e: fail stops that group).
#
# Usage (on deepnet login node, after git pull + assets present):
#   ./extras/submit_ingolstadt_1x21_chained.sh
#   DRY_RUN=1 ./extras/submit_ingolstadt_1x21_chained.sh
#   # optional: export MCS_LABEL=... if your queue needs it
#
# Do NOT use sbatch --export=ALL on this cluster.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs extras/_slurm_generated

if [[ -z "${MCS_LABEL:-}" && "${DRY_RUN:-0}" != "1" ]]; then
  echo "Note: MCS_LABEL unset; submitting without --mcs-label"
fi

SEED="${SEED:-42}"
NETWORK="${NETWORK:-sumo1x21}"
PREFIX_BASE="${PREFIX_BASE:-homo_1x21_es}"
PREFIX_L1="${PREFIX_L1:-odh_l1_1x21_es}"
PREFIX_L2="${PREFIX_L2:-odh_l2_1x21_es}"

# Wall-clock per group (sequential runs inside).
# gpu2 rejects requests above the partition MaxTime (PartitionTimeLimit) —
# other deepnet submits here use 24–48h, so default all groups to 48h.
HOURS_BASELINE="${HOURS_BASELINE:-48}"
HOURS_AXES="${HOURS_AXES:-48}"
HOURS_L1="${HOURS_L1:-48}"
HOURS_L2="${HOURS_L2:-48}"

METHODS=(fixedtime maxpressure dqn presslight colight)
AXES=(hetero slow_start crossing_proxy obs noise)

run_one_py() {
  # Emitted into job scripts: run one agent; install torch_scatter for colight*.
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
      for m in "${METHODS[@]}"; do
        body+="run_one ${m} ${PREFIX_BASE}"$'\n'
      done
      ;;
    axes)
      local axis m
      for axis in "${AXES[@]}"; do
        for m in "${METHODS[@]}"; do
          body+="run_one ${m}_${axis} axis_${axis}_1x21_es"$'\n'
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

submit_chained() {
  local groups=(baseline axes l1 l2)
  local hours=("${HOURS_BASELINE}" "${HOURS_AXES}" "${HOURS_L1}" "${HOURS_L2}")
  local prev_jid=""
  local i group script jid dep_args

  for i in "${!groups[@]}"; do
    group="${groups[$i]}"
    script="$(write_group_job "${group}" "${hours[$i]}")"

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      if [[ -z "${prev_jid}" ]]; then
        echo "DRY_RUN: sbatch ${script}  # group=${group} hours=${hours[$i]}"
      else
        echo "DRY_RUN: sbatch --dependency=afterok:${prev_jid} ${script}  # group=${group} hours=${hours[$i]}"
      fi
      prev_jid="JOBID_${group}"
      continue
    fi

    sbatch_args=()
    if [[ -n "${MCS_LABEL:-}" ]]; then
      sbatch_args+=(--mcs-label="${MCS_LABEL}")
    fi
    if [[ -n "${prev_jid}" ]]; then
      sbatch_args+=(--dependency="afterok:${prev_jid}")
    fi

    # Capture job id from sbatch ("Submitted batch job 12345")
    jid="$(sbatch "${sbatch_args[@]}" "${script}" | awk '{print $NF}')"
    echo "Submitted group=${group} job_id=${jid} hours=${hours[$i]} dep=${prev_jid:-none}"
    prev_jid="${jid}"
  done
}

echo "Ingolstadt chained submit: baseline -> axes -> l1 -> l2"
echo "network=${NETWORK} seed=${SEED}"
submit_chained
echo "Monitor: squeue -u \$USER"
echo "Logs: logs/i21_group_{baseline,axes,l1,l2}_<jid>.out"
