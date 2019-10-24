#!/usr/bin/env python

import sys
import os
import argparse
import numpy as np
import glob
import h5py
import sparse
# The following line works on Maxwell
sys.path.append('/home/ayyerkar/.local/dragonfly/utils/py_src')
#sys.path.append('/mnt/cbis/home/benedikt/software/EMC/utils/py_src')
import writeemc
import detector
import reademc

parser = argparse.ArgumentParser(description='Select hits from sparse HDF5 file and save to a new file')
parser.add_argument('run', type=int, help='Run number')
parser.add_argument('-p', '--path', type=str, help='Path to emc files',
                    default='/gpfs/exfel/exp/SPB/201901/p002316/scratch/sparse/lowq')
parser.add_argument('-o', '--output', type=str, help='Output path',
                    default='/gpfs/exfel/exp/SPB/201901/p002316/scratch/sparse/lowq/hits')
parser.add_argument('-d', '--detector', type=str, help='Path to detector file',
                    default='/gpfs/exfel/exp/SPB/201901/p002316/scratch/benedikt/aux/det_lowq.h5')
parser.add_argument('-t', '--threshold', type=int, help='hit selection threshold',
                    default=105)
args = parser.parse_args()

# Read all events
det = detector.Detector(args.detector)
allfile = os.path.join(args.path, 'r%04d.h5' %args.run)
print('Reading from file:', allfile)
allemc = reademc.EMCReader(allfile, det)

# Load cells and scores
with sparse.Run(allfile) as r:
    cells = r.cellIds
    score = r.litpixel
goodcells = np.ones(len(cells), dtype=np.bool)
goodcells[cells == 0] = False
if args.run <= 577:
    goodcells[cells >= 142] = False

# Save hits only
hitfile = os.path.join(args.output, "r%04d.h5" %(args.run))
print("Saving to new emc file: ", hitfile)
hitemc = writeemc.EMCWriter(hitfile, det.raw_mask.shape)

for i in range(allemc.num_frames):
    hit = score[i] > args.threshold
    if hit and goodcells[i]:
        hitemc.write_frame(allemc.get_frame(i, raw=True))
    sys.stderr.write('\r%d/%d'%(i+1, allemc.num_frames))
sys.stderr.write('\n')
hitemc.finish_write()

print('copying ids/scores from: %s' % allfile)
selection = (score > args.threshold) & (goodcells)
with h5py.File(allfile, "r") as af:
    with h5py.File(hitfile, "a") as hf:
        if "id" in hf:
            del hf["id"]
        if "scores" in hf:
            del hf["scores"]
        gid = hf.create_group("id")
        gscore = hf.create_group("scores")
        for k,v in af["id"].items():
            gid[k] = v[selection]
        for k,v in af["scores"].items():
            gscore[k] = v[selection]
