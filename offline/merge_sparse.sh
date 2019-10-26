#!/bin/bash

#SBATCH --array=82-119
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --partition=upex
#SBATCH --export=ALL

module load anaconda/3
python merge_sparse.py $SLURM_ARRAY_TASK_ID

