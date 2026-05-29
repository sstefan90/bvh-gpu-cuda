#!/bin/bash
#SBATCH --job-name=lbvh_mpi
#SBATCH --partition=gpu-turing
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=sbatch/logs/lbvh_mpi_%j.out
#SBATCH --error=sbatch/logs/lbvh_mpi_%j.err

ml course/cme213/nvhpc/24.1

TRIANGLES=${1:-data/bunny.tri.bin}
PREFIX=${2:-data/out_mpi}

mkdir -p sbatch/logs
echo "=== lbvh_mpi P=${SLURM_NTASKS} ==="
mpirun -np ${SLURM_NTASKS} ./lbvh_mpi "$TRIANGLES" "$PREFIX"
