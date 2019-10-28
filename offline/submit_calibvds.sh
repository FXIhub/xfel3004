#!/bin/sh

if [[ $USER == "sellberj" ]];
then
    DIRLOGS="/gpfs/exfel/exp/SPB/201901/p002316/scratch/sellberj/logs/"
else
    DIRLOGS="./"
fi
DIROUT="/gpfs/exfel/exp/SPB/201901/p002316/scratch/hits/"

if [ $# -eq 0 ];
then
    echo "USAGE: ./submit_calibvds.sh 115 [, 117]"
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
	echo "#!/bin/sh" > calibvds.sh
	echo "#SBATCH --job-name=run"$RUNNUM"_hits" >> calibvds.sh
	echo "#SBATCH -p upex --time=10:00:00" >> calibvds.sh
	echo "#SBATCH -o "$DIRLOGS"run"$RUNNUM"_%j.out" >> calibvds.sh
	echo "#SBATCH -e "$DIRLOGS"run"$RUNNUM"_%j.err" >> calibvds.sh
	echo "" >> calibvds.sh
	echo 'HOST=`hostname`' >> calibvds.sh
	echo 'echo "Node: $HOST"' >> calibvds.sh
	echo 'echo "Submitted by: $USER"' >> calibvds.sh
	echo 'TIME=`date`' >> calibvds.sh
	echo 'echo "Submitted at: $TIME"' >> calibvds.sh
	echo "" >> calibvds.sh  
	echo 'echo "Running calib_vds.py.."' >> calibvds.sh
	#echo "python calib_vds.py $RUNNUM -v -t 5000" >> calibvds.sh # TODO: implement adaptive threshold
	echo "python calib_vds.py $RUNNUM -v" >> calibvds.sh
	echo 'echo "calib_vds.py done!"' >> calibvds.sh
	echo "" >> calibvds.sh
	echo 'echo "Changing permissions.."' >> calibvds.sh
	echo "chmod +r "$DIROUT"*h5" >> calibvds.sh
	echo 'echo "Done!"' >> calibvds.sh
	chmod +xr calibvds.sh
	sbatch calibvds.sh
	echo "submitted run $RUNNUM"
    done
fi
