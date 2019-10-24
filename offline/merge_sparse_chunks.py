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

parser = argparse.ArgumentParser(description='Merge sparse file chunks into single file')
parser.add_argument('run', type=int, help='Run number')
parser.add_argument('-p', '--path', type=str, help='Path to chunked sparse files',
                    default='/gpfs/exfel/exp/SPB/201901/p002316/scratch/sparse/chunks')
parser.add_argument('-o', '--out_folder', help='Path to output folder',
                    default='/gpfs/exfel/exp/SPB/201901/p002316/scratch/sparse/')
parser.add_argument('-d', '--delete', action='store_false', default=True, 
                    help='Use this option for not deleting the chunks afterwards')
args = parser.parse_args()

# Merge sparse data
det = detector.Detector('/gpfs/exfel/exp/SPB/201802/p002160/scratch/ayyerkar/det/det_2160_allq2.h5')
chunked_flist = sorted(glob.glob(os.path.join(args.path, 'r%04d' %args.run, '*r%04d_*'%args.run)))
print('Merging %d chunked files' %len(chunked_flist))
chunked_emc = reademc.EMCReader(chunked_flist, det)
emcfile = os.path.join(args.out_folder, "r%04d.h5" %(args.run))
combined_emc = writeemc.EMCWriter(emcfile, det.raw_mask.shape)
for i in range(chunked_emc.num_frames):
    combined_emc.write_frame(chunked_emc.get_frame(i, raw=True))
    sys.stderr.write('\r%d/%d'%(i+1, chunked_emc.num_frames))
sys.stderr.write('\n')
nframes = chunked_emc.num_frames
combined_emc.finish_write()
print("Merged %d frames into %s" %(nframes, emcfile))

# Merge train, cell, pulse id
train_ids, cell_ids, pulse_ids = [], [], []
for filename in sorted(chunked_flist):
    with h5py.File(filename, 'r') as f:
        train_ids = np.hstack([train_ids, f['id/trains'][:]])
        cell_ids  = np.hstack([cell_ids,  f['id/cells'][:]])
        pulse_ids = np.hstack([pulse_ids, f['id/pulses'][:]])
with h5py.File(emcfile, 'a') as f:
    f['id/trains'] = train_ids
    f['id/cells']  = cell_ids
    f['id/pulses'] = pulse_ids

assert cell_ids.shape[0] == nframes, "Mismatch between shape of data and ids"
os.system('chmod ao+rw %s' % (emcfile))
if args.delete:
    for f in chunked_flist:
        os.system("rm %s" %f)
else:
    print("chunks are not deleted")
