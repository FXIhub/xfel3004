import sys
import os.path as op
import time
import argparse
import multiprocessing as mp
import ctypes
import itertools
import glob

import numpy as np
import h5py
from mpi4py import MPI

PREFIX = '/gpfs/exfel/exp/SQS/202102/p002601/'
DET_NAME = 'SQS_DET_DSSC1M-1'
CHUNK_SIZE = 32
NBINS = 25

class Histogrammer():
    def __init__(self, run, dark_run, testing=False):
        self.run = run
        self.dark_run = dark_run
        self.testing = testing

        if testing:
            self.out_fname = 'r%.4d'%self.run
        else:
            self.out_fname = PREFIX + 'scratch/hist/r%.4d'%self.run
        self.out_fname += '_hist.h5'

    def _get_cellids(self):
        fname = PREFIX+'/raw/r%.4d/RAW-R%.4d-DSSC00-S00000.h5'%(self.run, self.run)
        with h5py.File(fname, 'r') as fptr:
            cids = fptr['INSTRUMENT/'+DET_NAME+'/DET/0CH0:xtdf/image/cellId'][:1600,0]
        self.cellids = np.unique(cids)
        self.cell_mask = np.ones(self.cellids.max()+1, dtype='i8')*-1
        self.cell_mask[self.cellids] = np.arange(len(self.cellids))

    def run_mpi(self):
        comm = MPI.COMM_WORLD
        rank = comm.rank
        nproc = comm.size
        if nproc % 16 != 0:
            raise ValueError('Need number of processes to be multiple of 16')

        self._get_cellids()

        file_ind = rank % (nproc//16)
        my_module = rank // (nproc//16)
        num_files = len(glob.glob(PREFIX + '/raw/r%.4d/*DSSC00*.h5'%self.run))
        abort_job = 0
        if file_ind >= num_files:
            abort_job = 1
            #raise ValueError('Too many MPI ranks (max %d)'%(16*num_files))
        comm.allreduce(abort_job, op=MPI.SUM)
        if abort_job > 0:
            return
        fname = sorted(glob.glob(PREFIX + '/raw/r%.4d/*DSSC%.2d*.h5'%(self.run, my_module)))[file_ind]

        # Communicator group for other ranks with same module
        my_group = comm.group.Incl(np.arange(my_module*(nproc//16), (my_module+1)*(nproc//16)))
        my_comm = comm.Create_group(my_group)

        if rank == 0:
            print('%d files per module in run %d' % (num_files, self.run))
            print('%d ranks per module' % (my_comm.size))
            print('%d cells from %d to %d' % (len(self.cellids), self.cellids.min(), self.cellids.max()))
            print('Calculating histogram for %d cells' % len(self.cellids))
            print('Will write output to', self.out_fname)
            sys.stdout.flush()

        # Get bin minimum 
        with h5py.File(PREFIX + 'scratch/dark/r%.4d_dark.h5'%self.dark_run, 'r') as f:
            bin_min = np.round(f['data/mean'][my_module]).astype('i4').reshape(len(self.cellids),128*512) - 5

        # Allocate histogram
        my_hist = np.zeros((len(self.cellids), 128*512, NBINS), dtype='u8')
        stime = time.time()

        with h5py.File(fname, 'r') as f:
            dset = f['INSTRUMENT/'+DET_NAME+'/DET/%dCH0:xtdf/image/data'%my_module]
            cid = f['INSTRUMENT/'+DET_NAME+'/DET/%dCH0:xtdf/image/cellId'%my_module][:].ravel()
            
            num_chunks = int(np.ceil(dset.shape[0] / CHUNK_SIZE))
            if self.testing:
                num_chunks = 2
            for chunk in range(num_chunks):
                st, en = chunk*CHUNK_SIZE, (chunk+1)*CHUNK_SIZE

                cells = self.cell_mask[cid[st:en]]
                frames = dset[st:en,0,:,:].astype('i4').reshape(CHUNK_SIZE, 128*512)

                frames -= bin_min[cells]
                frames[frames < 0] = 0
                frames[frames > NBINS-1] = NBINS-1

                for c in range(CHUNK_SIZE):
                    my_hist[cells[c],np.arange(128*512),frames[c]] += 1

                if rank == 0:
                    sys.stderr.write('\r%d/%d chunks in %s (%f frames/s)' % (
                        chunk+1, num_chunks, 
                        op.basename(fname), 
                        (nproc//16)*(chunk+1)*CHUNK_SIZE/(time.time()-stime)))
                    sys.stderr.flush()
        if rank == 0:
            sys.stderr.write('\n')
            sys.stderr.flush()

        #with h5py.File(op.splitext(self.out_fname)[0]+'_%.2d.h5'%my_module, 'w') as f:
        #    f['data/data'] = my_hist.reshape(len(self.cellids), 128, 512, NBINS)
        #    f['data/bin_min'] = bin_min.reshape(len(self.cellids), 128, 512)
        #    f['data/cellId'] = self.cellids

        if my_comm.rank == 0:
            mod_hist = np.zeros_like(my_hist)
            my_comm.Reduce(my_hist, mod_hist, op=MPI.SUM, root=0)
            with h5py.File(op.splitext(self.out_fname)[0]+'_%.2d.h5'%my_module, 'w') as f:
                f['data/data'] = mod_hist.reshape(len(self.cellids), 128, 512, NBINS)
                f['data/bin_min'] = bin_min.reshape(len(self.cellids), 128, 512)
                f['data/cellId'] = self.cellids
            print('Written file for module %d' % my_module)
        else:
            my_comm.Reduce(my_hist, None, op=MPI.SUM, root=0)

    @staticmethod
    def _iterating_median(v, tol=3):
        if len(v) == 0:
            return 0
        vmin, vmax = v.min(), v.max()
        #vmin, vmax = -2*tol, 2*tol
        vmed = np.median(v[(vmin < v) & (v < vmax)])
        vmed0 = vmed
        i = 0
        while True:
            vmin, vmax = vmed-tol, vmed+tol
            vmed = np.median(v[(vmin < v) & (v < vmax)])
            if vmed == vmed0:
                break
            else:
                vmed0 = vmed
            i += 1
            if i > 20:
                break
        return vmed

    def _common_mode(self, img):
        """img should be subtracted by the dark.
        img.shape == (X, Y) 
        There is no mask
        The correction is applied IN-PLACE
        """
        ig = img.astype('f8').copy()
        L = 64
        for i, j in itertools.product(range(ig.shape[0] // L),
                                      range(ig.shape[1] // L)):
            img = ig[i*64:(i+1)*64, j*64:(j+1)*64]
            med = self._iterating_median(img.flatten())
            img -= med
        return ig

def main():
    parser = argparse.ArgumentParser(description='Calculate cell-wise histogram')
    parser.add_argument('run', help='Run number', type=int)
    parser.add_argument('dark_run', help='Dark run to get hist bins', type=int, default=-1)
    parser.add_argument('-t', '--testing', help='Testing mode (only 10 chunks)', action='store_true')
    args = parser.parse_args()

    hister = Histogrammer(args.run, args.dark_run, testing=args.testing)
    hister.run_mpi()

if __name__ == '__main__':
    main()
