#!/usr/bin/env bash
#SBATCH --job-name=mp_7x28_slowstart
#SBATCH --output=logs/mp_7x28_slowstart_%j.out
#SBATCH --error=logs/mp_7x28_slowstart_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

cd ~/LibSignalFork

CONDA_PREFIX="/data1/mmirzata/.conda/envs/libsignal"
export SUMO_HOME="${CONDA_PREFIX}/share/sumo"
export PATH="${CONDA_PREFIX}/bin:${SUMO_HOME}/bin:${PATH}"

echo "Host: $(hostname)"
echo "Agent: maxpressure_slow_start  Network: sumo7x28  Seed: 42  (slow start)"

python run.py -a maxpressure_slow_start -w sumo -n sumo7x28 --seed 42 --interface libsumo --prefix slow_start_test
