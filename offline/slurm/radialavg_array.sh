#!/bin/bash

#SBATCH --array=5-13
#SBATCH --time=04:00:00
#SBATCH --partition=upex-beamtime
#SBATCH --reservation=upex_003004
#SBATCH --export=ALL
#SBATCH -J radialavg
#SBATCH -o .%j.out
#SBATCH -e .%j.out

# Change the runs to process using the --array option on line 3

PREFIX=/gpfs/exfel/exp/SQS/202302/p003004

source /etc/profile.d/modules.sh
source ${PREFIX}/usr/Shared/xfel3004/source_this_at_euxfel

DARK_RUN=61

python ../radialavg.py ${SLURM_ARRAY_TASK_ID} ${DARK_RUN}

