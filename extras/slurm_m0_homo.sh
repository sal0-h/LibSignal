#!/usr/bin/env bash
# Single M0 (movie · axes OFF) job for LibSignal TSC.
#
# Submit via extras/submit_m0_homo.sh (preferred), or manually:
#   export MCS_LABEL=crs-XXXX   # or QSIURP_Salman
#   export AGENT=dqn NETWORK=sumo4x4 PREFIX=m0_homo
#   sbatch --export=ALL --mcs-label="${MCS_LABEL}" extras/slurm_m0_homo.sh
#
# Cluster note: LibSignal TSC CPU jobs on the lab are submitted from the
# gpujobs login node with --mcs-label (there is no "deepnet" partition for
# these). Deepnet2 --partition=gpu2 is for GPU/LLM jobs only.

#SBATCH --job-name=m0_homo
#SBATCH --output=logs/m0_homo_%j.out
#SBATCH --error=logs/m0_homo_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

REPO_DIR="${REPO_DIR:-${HOME}/LibSignalFork}"
CONDA_ENV="${CONDA_ENV:-libsignal}"
AGENT="${AGENT:?Set AGENT (fixedtime|maxpressure|dqn|presslight|colight)}"
NETWORK="${NETWORK:?Set NETWORK (sumo4x4|sumo1x21)}"
SEED="${SEED:-42}"
PREFIX="${PREFIX:-m0_homo}"
INTERFACE="${INTERFACE:-libsumo}"
NGPU="${NGPU:--1}"

cd "${REPO_DIR}"
mkdir -p logs

# Prefer an already-active conda env; otherwise activate CONDA_ENV.
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  if [[ -d "/data1/mmirzata/.conda/envs/${CONDA_ENV}" ]]; then
    CONDA_PREFIX="/data1/mmirzata/.conda/envs/${CONDA_ENV}"
    export PATH="${CONDA_PREFIX}/bin:${PATH}"
  elif command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
  else
    echo "No conda env found; expected ${CONDA_ENV}" >&2
    exit 1
  fi
fi

export SUMO_HOME="${SUMO_HOME:-${CONDA_PREFIX}/share/sumo}"
if [[ ! -d "${SUMO_HOME}" ]]; then
  SUMO_HOME="$(python -c 'import os,sumo; print(os.path.dirname(sumo.__file__))')"
  export SUMO_HOME
fi
export PATH="${CONDA_PREFIX}/bin:${SUMO_HOME}/bin:${PATH}"

echo "Host:      $(hostname)"
echo "Job:       ${SLURM_JOB_ID:-local}"
echo "Repo:      ${REPO_DIR}"
echo "Conda:     ${CONDA_PREFIX}"
echo "SUMO_HOME: ${SUMO_HOME}"
echo "Agent:     ${AGENT}"
echo "Network:   ${NETWORK}"
echo "Seed:      ${SEED}"
echo "Prefix:    ${PREFIX}"
echo "Start:     $(date -Is)"

if [[ "${AGENT}" == colight* ]]; then
  python -c "import torch_scatter" 2>/dev/null || {
    TV="$(python -c 'import torch; print(torch.__version__.split("+")[0])')"
    echo "Installing torch_scatter for torch ${TV}..."
    pip install torch_scatter -f "https://data.pyg.org/whl/torch-${TV}.html"
  }
fi

python -c "import libsumo; print('libsumo: OK')" 2>/dev/null || {
  echo "WARNING: libsumo missing — run.py may fall back to traci (slower)."
}

python run.py \
  --agent "${AGENT}" \
  --world sumo \
  --network "${NETWORK}" \
  --seed "${SEED}" \
  --ngpu "${NGPU}" \
  --interface "${INTERFACE}" \
  --prefix "${PREFIX}"

echo "Done: $(date -Is)"
echo "Expect metrics under:"
echo "  data/output_data/tsc/sumo_${AGENT}/${NETWORK}/${PREFIX}/logger/new_metrics*.csv"
