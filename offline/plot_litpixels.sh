#!/bin/sh

if [[ $USER == "sellberj" ]];
then
    DIROUT="/home/sellberj/2019-SPI_ribosomes/figures/"
else
    DIRLOGS="./"
fi

if [ $# -eq 0 ];
then
    echo "USAGE: ./plot_litpixels.sh 115 [, 117]"
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
	python calib_vds.py $RUNNUM -P 1 -v
	mv run$RUNNUM*png $DIROUT/.
    done
fi
