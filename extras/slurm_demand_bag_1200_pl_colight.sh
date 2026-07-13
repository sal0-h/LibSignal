#!/usr/bin/env bash
#SBATCH --job-name=bag1200_pl_cl
#SBATCH --output=logs/bag1200_pl_cl_%A_%a.out
#SBATCH --error=logs/bag1200_pl_cl_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-3

# PressLight + CoLight on the same 1200s fixed-vs-bag protocol as DQN.
# One seed (42) is enough for this follow-up (DQN already showed seed stability).
# Array layout (4 jobs):
#   0 presslight_fixed1200 seed42
#   1 presslight_bag1200   seed42
#   2 colight_fixed1200    seed42
#   3 colight_bag1200      seed42
#
# Submit (repo root on gpujobs):
#   mkdir -p logs
#   export MCS_LABEL=crs-XXXX
#   sbatch --mcs-label="${MCS_LABEL}" extras/slurm_demand_bag_1200_pl_colight.sh

set -euo pipefail

cd "${HOME}/LibSignalFork"

CONDA_PREFIX="${CONDA_PREFIX:-/data1/mmirzata/.conda/envs/libsignal}"
if [[ -x "${CONDA_PREFIX}/bin/python" ]]; then
  export SUMO_HOME="${CONDA_PREFIX}/share/sumo"
  export PATH="${CONDA_PREFIX}/bin:${SUMO_HOME}/bin:${PATH}"
  PYTHON="${CONDA_PREFIX}/bin/python"
else
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-traffic}"
  PYTHON=python
  export SUMO_HOME="${SUMO_HOME:-$(python -c 'import os,sumo; print(os.path.dirname(sumo.__file__))')}"
fi

IDX="${SLURM_ARRAY_TASK_ID:-0}"

AGENTS=(presslight_fixed1200 presslight_bag1200 colight_fixed1200 colight_bag1200)
SEEDS=(42 42 42 42)
PREFIXES=(pl_fixed1200_s42 pl_bag1200_s42 cl_fixed1200_s42 cl_bag1200_s42)

AGENT="${AGENTS[$IDX]}"
SEED="${SEEDS[$IDX]}"
PREFIX="${PREFIXES[$IDX]}"

echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID:-local}  Array: ${IDX}"
echo "Agent: ${AGENT}  Network: sumo4x4  Seed: ${SEED}  Prefix: ${PREFIX}"
echo "SUMO_HOME=${SUMO_HOME}"
date

if [[ "${AGENT}" == colight_* ]]; then
  if ! "${PYTHON}" -c "import torch_scatter" 2>/dev/null; then
    echo "ERROR: torch_scatter missing — required for CoLight. Install matching torch wheel, then resubmit array tasks 2-3."
    exit 1
  fi
fi

"${PYTHON}" run.py \
  -a "${AGENT}" \
  -w sumo \
  -n sumo4x4 \
  --seed "${SEED}" \
  --interface libsumo \
  --ngpu -1 \
  --prefix "${PREFIX}"

date
echo "Done."
