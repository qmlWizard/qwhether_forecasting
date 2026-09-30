#!/bin/bash
#SBATCH --job-name=pennylane30
#SBATCH --partition=gpu-small
#SBATCH --nodes=20  
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --output=/home/cdacB/santhoshj/digvijay/output/%j.out
#SBATCH --error=/home/cdacB/santhoshj/digvijay/error/%j.err

source ~/.bashrc

. /home/apps/spack/share/spack/setup-env.sh

spack load /oxmullt
spack load /gaily2jw
spack load /2wdhn7r

export OPENBLAS_ROOT=$(spack location -i /geypa2)
export LD_LIBRARY_PATH=$OPENBLAS_ROOT/lib:$LD_LIBRARY_PATH

conda activate pennylane-mpi

python qwhether_forecasting/benchmark.py

echo "Execution Complete!!!"