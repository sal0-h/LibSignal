#!/usr/bin/env bash
#SBATCH --job-name=baseline_es_4x4
#SBATCH --output=logs/baseline_early_stop_4x4_%x_%j.out
#SBATCH --error=logs/baseline_early_stop_4x4_%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#
# One agent per job. Submit via extras/submit_baseline_early_stop_4x4.sh
# Required env: AGENT (maxpressure|fixedtime|dqn|presslight|colight)

set -euo pipefail

cd ~/LibSignalFork

AGENT="${AGENT:?Set AGENT=maxpressure|fixedtime|dqn|presslight|colight}"
NETWORK="${NETWORK:-sumo4x4}"
SEED="${SEED:-42}"
PREFIX="${PREFIX:-baseline_early_stop}"
NGPU="${NGPU:--1}"
INTERFACE="${INTERFACE:-libsumo}"

CONDA_PREFIX="${CONDA_PREFIX:-/data1/mmirzata/.conda/envs/libsignal}"
export SUMO_HOME="${SUMO_HOME:-${CONDA_PREFIX}/share/sumo}"
export PATH="${CONDA_PREFIX}/bin:${SUMO_HOME}/bin:${PATH}"

echo "Host: $(hostname)"
echo "Agent: ${AGENT}  Network: ${NETWORK}  Seed: ${SEED}  Prefix: ${PREFIX}"
echo "Budget: min=20 max=2000 patience=20 (from configs/tsc/base.yml)"

if [[ "${AGENT}" == "colight" ]]; then
  python -c "import torch_scatter" 2>/dev/null || {
    echo "torch_scatter not found, attempting install..."
    TV="$(python -c 'import torch; print(torch.__version__.split("+")[0])')"
    pip install torch_scatter -f "https://data.pyg.org/whl/torch-${TV}.html" || {
      echo "ERROR: torch_scatter install failed; CoLight cannot run."
      exit 1
    }
  }
fi

python run.py \
  -a "${AGENT}" \
  -w sumo \
  -n "${NETWORK}" \
  --seed "${SEED}" \
  --ngpu "${NGPU}" \
  --interface "${INTERFACE}" \
  --prefix "${PREFIX}"
