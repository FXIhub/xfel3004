#!/usr/bin/env python

import sys
import os
import argparse
import numpy as np
import glob
# The following line works on Maxwell
sys.path.append('/home/ayyerkar/.local/dragonfly/utils/py_src')
import writeemc
import detector
import reademc

parser = argparse.ArgumentParser(description='Merge EMC file chunks into single file')
parser.add_argument('run', type=int, help='Run number')
parser.add_argument('-p', '--path', type=str, help='Path to chunked emc files',
                    default='/gpfs/exfel/exp/SPB/201901/p002316/scratch/sparse/chunks')
parser.add_argument('-o', '--out_folder', help='Path to output folder',
                    default='/gpfs/exfel/exp/SPB/201901/p002316/scratch/sparse/')
args = parser.parse_args()

post_tag = '_lowq.h5'
det = detector.Detector('../geometry/det'+post_tag)
chunked_flist = sorted(glob.glob(os.path.join(args.path, '*r%04d_*'%args.run)))
print('Merging the following files:', chunked_flist)
chunked_emc = reademc.EMCReader(chunked_flist, det)
emcfile = os.path.join(args.out_folder, "r%04d" %(args.run) +post_tag)
combined_emc = writeemc.EMCWriter(emcfile, det.raw_mask.shape)

for i in range(chunked_emc.num_frames):
    combined_emc.write_frame(chunked_emc.get_frame(i, raw=True))
    sys.stderr.write('\r%d/%d'%(i+1, chunked_emc.num_frames))
sys.stderr.write('\n')

combined_emc.finish_write()

for f in chunked_flist:
    os.system("rm %s" %f)
