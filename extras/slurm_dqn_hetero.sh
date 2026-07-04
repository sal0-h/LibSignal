#!/usr/bin/env bash
#SBATCH --job-name=dqn_hetero_4x4
#SBATCH --output=logs/dqn_hetero_4x4_%j.out
#SBATCH --error=logs/dqn_hetero_4x4_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

cd ~/LibSignalFork
eval "$(conda shell.bash hook)"
conda activate libsignal
export SUMO_HOME=${CONDA_PREFIX}/share/sumo
export PATH=${SUMO_HOME}/bin:${PATH}

echo "Host: $(hostname)"
echo "Agent: dqn_hetero  Network: sumo4x4  Seed: 42"

python run.py -a dqn_hetero -w sumo -n sumo4x4 --seed 42 --interface libsumo --prefix hetero_test
