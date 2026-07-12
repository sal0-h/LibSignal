#!/usr/bin/env bash
#SBATCH --job-name=demand_bag_1200
#SBATCH --output=logs/demand_bag_1200_%x_%A_%a.out
#SBATCH --error=logs/demand_bag_1200_%x_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-5

# Grid4x4 1200s fixed-vs-bag pilot.
# Array layout (6 jobs):
#   0 dqn_fixed1200 seed42
#   1 dqn_fixed1200 seed43
#   2 dqn_bag1200   seed42
#   3 dqn_bag1200   seed43
#   4 maxpressure_1200 seed42   (flat ref)
#   5 fixedtime_1200   seed42   (flat ref)
#
# Optional: set PRESSLIGHT=1 to swap dqn_* for presslight_* (same indices 0-3).
#
# Submit (from repo root on gpujobs login node):
#   mkdir -p logs
#   export MCS_LABEL=crs-XXXX
#   sbatch --mcs-label="${MCS_LABEL}" extras/slurm_demand_bag_1200.sh

set -euo pipefail

cd "${HOME}/LibSignalFork"

CONDA_PREFIX="${CONDA_PREFIX:-/data1/mmirzata/.conda/envs/libsignal}"
if [[ -x "${CONDA_PREFIX}/bin/python" ]]; then
  export SUMO_HOME="${CONDA_PREFIX}/share/sumo"
  export PATH="${CONDA_PREFIX}/bin:${SUMO_HOME}/bin:${PATH}"
  PYTHON="${CONDA_PREFIX}/bin/python"
else
  # Fallback: activate a named conda env if present
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-traffic}"
  PYTHON=python
  export SUMO_HOME="${SUMO_HOME:-$(python -c 'import os,sumo; print(os.path.dirname(sumo.__file__))')}"
fi

IDX="${SLURM_ARRAY_TASK_ID:-0}"
USE_PRESSLIGHT="${PRESSLIGHT:-0}"

SEEDS=(42 43 42 43 42 42)
if [[ "${USE_PRESSLIGHT}" == "1" ]]; then
  AGENTS=(presslight_fixed1200 presslight_fixed1200 presslight_bag1200 presslight_bag1200 maxpressure_1200 fixedtime_1200)
  PREFIXES=(pl_fixed1200_s42 pl_fixed1200_s43 pl_bag1200_s42 pl_bag1200_s43 mp_heldout_s42 ft_heldout_s42)
else
  AGENTS=(dqn_fixed1200 dqn_fixed1200 dqn_bag1200 dqn_bag1200 maxpressure_1200 fixedtime_1200)
  PREFIXES=(dqn_fixed1200_s42 dqn_fixed1200_s43 dqn_bag1200_s42 dqn_bag1200_s43 mp_heldout_s42 ft_heldout_s42)
fi

AGENT="${AGENTS[$IDX]}"
SEED="${SEEDS[$IDX]}"
PREFIX="${PREFIXES[$IDX]}"

echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID:-local}  Array: ${IDX}"
echo "Agent: ${AGENT}  Network: sumo4x4  Seed: ${SEED}  Prefix: ${PREFIX}"
echo "SUMO_HOME=${SUMO_HOME}"
date

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
