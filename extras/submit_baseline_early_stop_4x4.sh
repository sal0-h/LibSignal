#!/usr/bin/env bash
# Submit baseline early-stop 4x4 jobs on gpujobs (login node).
#
# Usage:
#   export MCS_LABEL=crs-XXXX
#   ./extras/submit_baseline_early_stop_4x4.sh              # all 5
#   ./extras/submit_baseline_early_stop_4x4.sh baselines    # FT + MP
#   ./extras/submit_baseline_early_stop_4x4.sh rl           # DQN + PressLight + CoLight
#   ./extras/submit_baseline_early_stop_4x4.sh dqn          # single agent
#
# Note: do NOT use sbatch --export=ALL on this cluster (causes
# "user env retrieval failed requeued held"). Agent is baked into a
# per-job script instead, matching extras/submit_crossing_proxy_4x4.sh.

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs extras/_slurm_generated

if [[ -z "${MCS_LABEL:-}" ]]; then
  echo "Set MCS_LABEL first, e.g. export MCS_LABEL=crs-XXXX"
  exit 1
fi

MODE="${1:-all}"
NETWORK="${NETWORK:-sumo4x4}"
SEED="${SEED:-42}"
PREFIX="${PREFIX:-baseline_early_stop}"
NGPU="${NGPU:--1}"
INTERFACE="${INTERFACE:-libsumo}"

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

write_job_script() {
  local agent="$1"
  local out="extras/_slurm_generated/bes_${agent}_4x4.sh"
  cat > "${out}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=bes_${agent}_4x4
#SBATCH --output=logs/baseline_early_stop_4x4_${agent}_%j.out
#SBATCH --error=logs/baseline_early_stop_4x4_${agent}_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

cd ~/LibSignalFork

AGENT="${agent}"
NETWORK="${NETWORK}"
SEED="${SEED}"
PREFIX="${PREFIX}"
NGPU="${NGPU}"
INTERFACE="${INTERFACE}"

CONDA_PREFIX="/data1/mmirzata/.conda/envs/libsignal"
export SUMO_HOME="\${CONDA_PREFIX}/share/sumo"
export PATH="\${CONDA_PREFIX}/bin:\${SUMO_HOME}/bin:\${PATH}"

echo "Host: \$(hostname)"
echo "Agent: \${AGENT}  Network: \${NETWORK}  Seed: \${SEED}  Prefix: \${PREFIX}"
echo "Budget: min=20 max=200 patience=20 (from configs/tsc/base.yml)"

if [[ "\${AGENT}" == "colight" ]]; then
  python -c "import torch_scatter" 2>/dev/null || {
    echo "torch_scatter not found, attempting install..."
    TV="\$(python -c 'import torch; print(torch.__version__.split("+")[0])')"
    pip install torch_scatter -f "https://data.pyg.org/whl/torch-\${TV}.html" || {
      echo "ERROR: torch_scatter install failed; CoLight cannot run."
      exit 1
    }
  }
fi

python run.py \\
  -a "\${AGENT}" \\
  -w sumo \\
  -n "\${NETWORK}" \\
  --seed "\${SEED}" \\
  --ngpu "\${NGPU}" \\
  --interface "\${INTERFACE}" \\
  --prefix "\${PREFIX}"
EOF
  chmod +x "${out}"
  echo "${out}"
}

AGENTS=($(pick_agents))
echo "Submitting agents: ${AGENTS[*]}"
echo "network=${NETWORK} seed=${SEED} prefix=${PREFIX}"

for agent in "${AGENTS[@]}"; do
  script="$(write_job_script "${agent}")"
  sbatch --mcs-label="${MCS_LABEL}" "${script}"
done

echo "Submitted ${#AGENTS[@]} job(s). Monitor: squeue -u \$USER"
echo "Outputs: data/output_data/tsc/sumo_<agent>_${PREFIX}/${NETWORK}/${PREFIX}/logger/"
