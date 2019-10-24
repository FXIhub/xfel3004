3# Offline data access

To access the offline computer cluster, login to the display gateway of the Maxwell cluster:

```
ssh -X <upex account>@max-display.desy.de
```

You will need X11 forwarding (using e.g. XQuartz) to be able to display graphics from the login node on your screen.

Don't run big jobs on the login nodes. The Maxwell cluster uses Slurm to submit analysis jobs to the cluster, use the `upex` partition for XFEL analysis. To submit a job and view its status:

```
sbatch -p upex <job command>
squeue -u <upex account>
```

To allocate an analysis node that you can use interactively, type:

```
srun -p upex -t 10:00:00 --pty $SHELL -i
```

You can find more useful Slurm commands here:

http://xray.bmc.uu.se/~filipe/admin/davinci_user.html

You can find more info about offline data analysis at EuXFEL here:

http://www.desy.de/~barty/cheetah/Cheetah/EuXFEL_data_analysis.html

## Clone this repository

To get access to the analysis scripts in this repository, create a folder and clone the repository:

```
git clone https://github.com/FXIhub/xfel2316.git

```

This clones the repository to the current folder using HTTPS, which is fine for read access. To get write access to the repository, setup your SSH keys properly and do instead:

```
git clone git@github.com:FXIhub/xfel2316.git
```

Before you start performing your analysis, don't forget to load the proper modules:

```
source xfel2316/source_this_at_euxfel
```

## Experiment folder

Once you login to the cluster you will start in your home directory. To go to the experiment directory, write:

```
cd /gpfs/exfel/exp/SPB/201901/p002316
```

This contains the `proc` folder for processed runs, `raw` folder for raw data runs, `usr` for smaller user files like software and calibration files and `scratch` for larger data files. Please make a folder in `scratch` with your UPEX user name and place your analysis output for the experiment there:

```
cd scratch
mkdir <upex account>
```

For sharing files with other users, please make sure they have the correct permissions, which if you're being lazy means:

```
chmod 755 *
```

## Module combiner

AGIPD data is written in separate files for each module. The `combine_modules`
python script combines the data from different modules and applies detector
calibrations resulting in a detector image for a given run number and frame number.

### Usage

Here is some basic usage inside an `ipython` console.
```python
import combine_modules
c = combine_modules.AGIPD_Combiner(18) # For run 18 with AgBe data
frame = c.get_frame(8730, calibrate='true') # For frame number 8730

# For XFEL-calibrated data (from the /proc folder)
c = combine_modules.AGIPD_Combiner(18, raw=False)
frame = c.get_frame(8730)
```

Note that the frame numbers are of just the cells which contain data. Thus, in 
this experiment, there are currenty 176 frames per train, resulting in 1760 indices per second

## Cheetah

See online documentation for Cheetah at EuXFEL:

http://www.desy.de/~barty/cheetah/Cheetah/At_EuXFEL.html

## VDS files
An alternate way to look at the data is to use the virtual data set (VDS) feature of HDF5. These files can be generated with the `vds.py` script in this folder. Some runs should already be converted in the `/scratch/vds/` folder. These files have all the modules for a given train in the same dataset slice. Thus, one can use the simple HDF5/h5py API to access the data for a given frame.

```
$ h5ls -r r0018_vds_raw.h5
/                        Group
/INSTRUMENT              Group
/INSTRUMENT/SPB_DET_AGIPD1M-1 Group
/INSTRUMENT/SPB_DET_AGIPD1M-1/DET Group
/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/image Group
/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/image/data Dataset {16, 173008, 2, 512, 128}
/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/image/trainId Dataset {173008}

$ h5ls -r r0018_vds_proc.h5
/                        Group
/INSTRUMENT              Group
/INSTRUMENT/SPB_DET_AGIPD1M-1 Group
/INSTRUMENT/SPB_DET_AGIPD1M-1/DET Group
/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/image Group
/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/image/data Dataset {16, 173008, 512, 128}
/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/image/trainId Dataset {173008}
```
The extra dimension in the raw data has the gain (digital) data for the frame.

## Hitfinding
The virtual data sets can be used for hitinding with the `litpixels.py` and `calib_vds.py` scripts in this folder. The `litpixels.py` script produces an HDF5 file with lit pixel values for all shots, that should be placed in the `/scratch/hitlist/` folder with read permissions to all users `chmod +r /scratch/hitlist/*h5`. They can then be read by the `calib_vds.py` script to determine a hit threshold and calibrate and save the hits.  Some runs should already be converted in the `/scratch/hits/` folder. Example usage:

```
python litpixels.py /scratch/vds/r0072_vds_raw.h5 -n 40 -m 4 -t 25
python calib_vds.py 72 -P -v
```
