#!/usr/bin/env zsh

if [ $# -lt 1 ]
then
	echo Need folder name to move
	exit
fi

orig=`pwd`

dest=/gpfs/exfel/exp/SPB/201901/p002316/usr/Shared/calib/`echo $1|cut -d- -f1`
echo Moving to $dest
[ ! -d $dest ] && mkdir $dest; cp -v $1/*.h5 $dest/
chmod -v a+r $dest/*

cd $dest
cd ..
echo Current directory: $PWD
bname=`basename $dest`
rm latest
ln -v -s $bname latest

cd $orig
echo Base directory: $PWD

#dest=/gpfs/exfel/exp/SPB/201901/p002316/scratch/cheetah/calib/agipd/$1
dest=/gpfs/exfel/exp/SPB/201901/p002316/scratch/calib/$1
echo Moving to $dest
[ ! -d $dest ] && mkdir $dest; chmod -v a+r $dest
cp -v $1/*.h5 $dest/
chmod -v a+r $dest/*

cd $dest
cd ..
echo Current directory: $PWD
bname=`basename $dest`
rm latest
ln -v -s $bname latest

cd -
