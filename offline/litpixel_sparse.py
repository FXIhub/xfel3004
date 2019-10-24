#!/usr/bin/env python
import sys
import os
import argparse
import numpy as np
import h5py
import sparse

parser = argparse.ArgumentParser(description='Count lit-pixels on sparse HDF5 files')
parser.add_argument('run', type=int, help='Run number')
parser.add_argument('-p', '--path', type=str, help='Path to sparse file',
                    default='/gpfs/exfel/exp/SPB/201901/p002316/scratch/sparse/')
parser.add_argument('-m', '--mask', type=str, default=None, help='Good pixel mask')
args = parser.parse_args()

# Good pixel mask
if args.mask is None:
    goodpixels = np.ones((16,128,512), dtype=np.bool)
else:
    with h5py.File(args.mask,'r') as f:
        goodpixels = f['data/data'][:]
assert goodpixels.shape == (16,128,512), "Good pixel mask should have shape (16,128,512)"

# Counting lit-pixel on modules 3,4,8,15
sparsefile = os.path.join(args.path, "r%04d.h5" %args.run)
print("Running lit-pixel counting on file ", sparsefile)
with sparse.Litpixel(sparsefile, goodmask=goodpixels) as lp:
    score = lp.count()

# Save score into input file
with h5py.File(sparsefile, 'a') as f:
    f['scores/litpixel'] = score
