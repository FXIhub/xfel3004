#!/bin/bash

#SBATCH --array=82-119
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --partition=upex
#SBATCH --reservation=upex_002316
#SBATCH --export=ALL

module load anaconda/3
mpirun -n 40 python vds_to_emc.py $SLURM_ARRAY_TASK_ID -s 1 -o /gpfs/exfel/exp/SPB/201901/p002316/scratch/sparse/chunks/ -c 10

