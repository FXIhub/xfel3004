#!/bin/sh

if [[ $USER == "sellberj" ]];
then
    DIRLOGS="/gpfs/exfel/exp/SPB/201901/p002316/scratch/sellberj/logs/"
else
    DIRLOGS="./"
fi
DIROUT="/gpfs/exfel/exp/SPB/201901/p002316/scratch/vds/"

if [ $# -eq 0 ];
then
    echo "USAGE: ./submit_vds.sh 115 [, 117]"
else
    eval STARTNUM=$1
    eval ENDNUM=$1
    if [ $# -gt 1 ];
    then
       eval ENDNUM=$2
       if [ $# -gt 2 ];
       then
	   echo "WARNING: following arguments are ignored:"
	   for i in $(seq 3 $#);
	   do
	       eval RUNARG=\${$i}
	       echo $RUNARG
	   done
       fi
    fi
    for RUNNUM in $(seq $STARTNUM $ENDNUM);
    do
	echo "#!/bin/sh" > vds.sh
	echo "#SBATCH --array="$RUNNUM >> vds.sh
	echo "#SBATCH --job-name=vds"$RUNNUM >> vds.sh
	echo "#SBATCH -p upex --time=12:00:00" >> vds.sh
	echo "#SBATCH --export=ALL" >> vds.sh
	echo "#SBATCH -o "$DIRLOGS"vds"$RUNNUM"_%j.out" >> vds.sh
	echo "#SBATCH -e "$DIRLOGS"vds"$RUNNUM"_%j.err" >> vds.sh
	echo "" >> vds.sh
	echo 'HOST=`hostname`' >> vds.sh
	echo 'echo "Node: $HOST"' >> vds.sh
	echo 'echo "Submitted by: $USER"' >> vds.sh
	echo 'TIME=`date`' >> vds.sh
	echo 'echo "Submitted at: $TIME"' >> vds.sh
	echo "" >> vds.sh
	echo "source /etc/profile.d/modules.sh" >> vds.sh
	echo "module purge" >> vds.sh
	echo "module load anaconda/3" >> vds.sh
	echo "" >> vds.sh
	echo 'echo "Running vds.py.."' >> vds.sh
	echo "python vds.py $""SLURM_ARRAY_TASK_ID" >> vds.sh
	#echo "python vds.py $RUNNUM" >> vds.sh
	echo 'echo "vds.py done!"' >> vds.sh
	echo "" >> vds.sh
	echo 'echo "Changing permissions.."' >> vds.sh
	echo "chmod +r "$DIROUT"*h5" >> vds.sh
	echo "chmod a+w "$DIROUT"*h5" >> vds.sh
	echo 'echo "Done!"' >> vds.sh
	chmod +xr vds.sh
	sbatch vds.sh
	echo "submitted vds $RUNNUM"
    done
fi
