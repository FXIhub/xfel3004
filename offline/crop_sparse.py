#!/usr/bin/env python

import sys
import os
import argparse
import numpy as np
import glob
import h5py
# The following line works on Maxwell
sys.path.append('/home/ayyerkar/.local/dragonfly/utils/py_src')
import writeemc
import detector
import reademc

parser = argparse.ArgumentParser(description='Crop full sparse file into smaller size')
parser.add_argument('run', type=int, help='Run number')
parser.add_argument('-p', '--path', type=str, help='Path to allq emc files',
                    default='/gpfs/exfel/exp/SPB/201901/p002316/scratch/sparse/')
parser.add_argument('-r', '--res', help='Resolution to save to (0=lowq, 1=medq). Default=0', type=int, default=0)
args = parser.parse_args()

if 0 <= args.res < 2:
    post_tag = ['_lowq.h5', '_medq.h5'][args.res]
    # Module order matches geometry file
    modules = [[3,4,8,15], [2,3,4,5,9,8,15,14], np.arange(16)][args.res]
    subset = [128, 256, 0][args.res]
    num_pix = [4*128*128, 4*256*256, 4*512*512][args.res]
else:
    print('"res" parameter can only have values 0, 1')
    sys.exit(1)

allqdet = detector.Detector('/gpfs/exfel/exp/SPB/201901/p002316/scratch/benedikt/aux/det_allq.h5')
allqfile = os.path.join(args.path, 'r%04d.h5' %args.run)
print('Cropping the following files:', allqfile)
allq_emc = reademc.EMCReader(allqfile, allqdet)

det = detector.Detector('/gpfs/exfel/exp/SPB/201901/p002316/scratch/benedikt/aux/det'+post_tag)
emcfile = os.path.join(args.path, post_tag[1:-3] + "/r%04d.h5" %(args.run))
print("Saving to new emc file: ", emcfile)
emc = writeemc.EMCWriter(emcfile, num_pix)

for i in range(allq_emc.num_frames):
    frame = allq_emc.get_frame(i, raw=True).reshape((16,512,128))
    emc.write_frame(frame[modules,-subset:].ravel().astype('i4'))
    sys.stderr.write('\r%d/%d'%(i+1, allq_emc.num_frames))
sys.stderr.write('\n')
emc.finish_write()

print('copying ids/scores from: %s' % allqfile)
with h5py.File(allqfile, "r") as af:
    with h5py.File(emcfile, "a") as cf:
        if "id" in cf:
            del cf["id"]
        if "scores" in cf:
            del cf["scores"]
        af.copy(af["id"], cf)
        af.copy(af["scores"], cf)
os.system('chmod ao+rw %s' % (emcfile))
