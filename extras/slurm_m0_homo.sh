#!/usr/bin/env bash
# Single M0 (movie · axes OFF) job for LibSignal TSC.
#
# Submit via extras/submit_m0_homo.sh (preferred).
#
# Important: do NOT inherit login-node CONDA_PREFIX=/opt/anaconda3 via
# --export=ALL — compute nodes often lack that path (exit 127: python not found).
# Always use the shared libsignal env under /data1/mmirzata/.conda/envs/.

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

# Force shared env (same as extras/slurm_realism_full_*.sh). Ignore login CONDA_PREFIX.
CONDA_PREFIX="${CONDA_PREFIX_OVERRIDE:-/data1/mmirzata/.conda/envs/${CONDA_ENV}}"
if [[ ! -x "${CONDA_PREFIX}/bin/python" ]]; then
  echo "ERROR: python not found at ${CONDA_PREFIX}/bin/python" >&2
  ls -la "${CONDA_PREFIX}/bin" 2>&1 | head -20 >&2 || true
  exit 127
fi

export CONDA_PREFIX
export SUMO_HOME="${CONDA_PREFIX}/share/sumo"
export PATH="${CONDA_PREFIX}/bin:${SUMO_HOME}/bin:${PATH}"
PYTHON="${CONDA_PREFIX}/bin/python"

echo "Host:      $(hostname)"
echo "Job:       ${SLURM_JOB_ID:-local}"
echo "Repo:      ${REPO_DIR}"
echo "Python:    ${PYTHON} ($("${PYTHON}" --version 2>&1))"
echo "Conda:     ${CONDA_PREFIX}"
echo "SUMO_HOME: ${SUMO_HOME}"
echo "Agent:     ${AGENT}"
echo "Network:   ${NETWORK}"
echo "Seed:      ${SEED}"
echo "Prefix:    ${PREFIX}"
echo "Start:     $(date -Is)"

if [[ "${AGENT}" == colight* ]]; then
  "${PYTHON}" -c "import torch_scatter" 2>/dev/null || {
    TV="$("${PYTHON}" -c 'import torch; print(torch.__version__.split("+")[0])')"
    echo "Installing torch_scatter for torch ${TV}..."
    "${PYTHON}" -m pip install torch_scatter -f "https://data.pyg.org/whl/torch-${TV}.html"
  }
fi

"${PYTHON}" -c "import libsumo; print('libsumo: OK')" 2>/dev/null || {
  echo "WARNING: libsumo missing — run.py may fall back to traci (slower)."
}

"${PYTHON}" run.py \
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
