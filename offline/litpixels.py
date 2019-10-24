#!/usr/bin/env python

'''Calculate lit pixels/frame for a run using the VDS files'''

import os
import sys
import time
import glob
import multiprocessing as mp
import subprocess
import ctypes
import h5py
import numpy as np

class LitPixels():
    def __init__(self, vds_file, nproc=0, calib_glob=None, good_cells=None, thresh=25, chunk_size=4096):
        self.vds_file = vds_file
        self.calib_glob = calib_glob
        self.thresh = thresh
        self.chunk_size = chunk_size # Needs to be multiple of 32 for raw data
        if self.chunk_size % 32 != 0:
            print('WARNING: Performance is best with a multiple of 32 chunk_size')
        if nproc == 0:
            self.nproc = int(subprocess.check_output('nproc').decode().strip())
        else:
            self.nproc = nproc
        print('Using %d processes' % self.nproc)
        if good_cells is None:
            self.good_cells = np.arange(176)
        else:
            self.good_cells = np.array(good_cells)

        with h5py.File(vds_file, 'r') as f:
            self.dset_name = 'INSTRUMENT/'+list(f['INSTRUMENT'])[0]+'/DET/image/data'
            self.dshape = f[self.dset_name].shape
        if len(self.dshape) == 4:
            print('Processed data')
            self.is_raw = False
        elif len(self.dshape) == 5:
            print('Raw data. Need to calibrate')
            self.is_raw = True
            if self.calib_glob is None:
                self.calib_glob='/gpfs/exfel/exp/SPB/201901/p002316/usr/Shared/calib/latest/Cheetah*.h5'
        else:
            raise TypeError('Unparseable data shape: %s'%self.dshape)

        # TODO Filter specific warnings for NaN data in VDS files
        np.warnings.filterwarnings('ignore', category=RuntimeWarning)

    def run(self):
        sys.stderr.write('Calculating number of lit pixels for %d frames\n'%self.dshape[1])
        # Litpixels for each module and each frame
        litpix = mp.Array(ctypes.c_ulong, self.dshape[0]*self.dshape[1])
        jobs = []
        for m in range(16):
            p = mp.Process(target=self._module_worker, args=(m, litpix))
            jobs.append(p)
            p.start()
        for j in jobs:
            j.join()
        self.litpix = np.frombuffer(litpix.get_obj(), dtype='u8').reshape(16,-1).sum(0)

        out_fname = os.path.basename(self.vds_file).split('_')[0] + '_hits.h5'
        with h5py.File(out_fname, 'a') as f:
            if 'litpixels' in f:
                del f['litpixels']
            f['litpixels'] = self.litpix
            self._copy_ids(f)

    def run_module(self, module):
        sys.stderr.write('Calculating number of lit pixels for %d frames\n'%self.dshape[1])
        # Litpixels for each module and each frame
        litpix = mp.Array(ctypes.c_ulong, self.dshape[1])
        if self.is_raw:
            offsets, mask = self._get_constants(module)
        else:
            offsets = None
            mask = None
        jobs = []
        for c in range(self.nproc):
            p = mp.Process(target=self._part_worker, args=(c, module, offsets, mask, litpix))
            jobs.append(p)
            p.start()
        for j in jobs:
            j.join()
        self.litpix = np.frombuffer(litpix.get_obj(), dtype='u8')

        out_fname = os.path.basename(self.vds_file).split('_')[0] + '_hits.h5'
        with h5py.File(out_fname, 'a') as f:
            if 'litpixels_%.2d'%module in f:
                del f['litpixels_%.2d'%module]
            f['litpixels_%.2d'%module] = self.litpix
            self._copy_ids(f)

    def _copy_ids(self, fptr):
        f_vds = h5py.File(self.vds_file, 'r')
        dset_prefix = 'INSTRUMENT/'+list(f_vds['INSTRUMENT'])[0]+'/DET/image/'
        if 'ID/trainId' in fptr:
            del fptr['ID/trainId']
        fptr['ID/trainId'] = f_vds[dset_prefix+'trainId'][:]
        if 'ID/cellId' in fptr:
            del fptr['ID/cellId']
        fptr['ID/cellId'] = f_vds[dset_prefix+'cellId'][:]
        if 'ID/pulseId' in fptr:
            del fptr['ID/pulseId']
        fptr['ID/pulseId'] = f_vds[dset_prefix+'pulseId'][:]
        f_vds.close()

    def _get_constants(self, m):
        calib_fname = sorted(glob.glob(self.calib_glob))[m]
        with h5py.File(calib_fname, 'r') as f:
            offset = f['AnalogOffset'][:][0,self.good_cells]
            mask = f['Badpixel'][:][0,self.good_cells]
        return offset, 1 - mask

    def _module_worker(self, m, litpix):
        np_litpix = np.frombuffer(litpix.get_obj(), dtype='u8').reshape(16,-1)
        offsets, mask = self._get_constants(m)
        if m == 0:
            print('offsets shape:', offsets.shape)
        with h5py.File(self.vds_file, 'r') as f:
            num_chunks = int(np.ceil(self.dshape[1]/self.chunk_size))
            for c in range(num_chunks):
                pmin = c*self.chunk_size
                pmax = (c+1) * self.chunk_size
                sel_range = np.arange(pmin, pmax)
                sel_cells = sel_range % 176
                
                stime = time.time()
                #vals = f[self.dset_name][m, sel_range]
                vals = f[self.dset_name][m, pmin:pmax]
                etime = time.time()
                if m == 0:
                    sys.stderr.write('%d frames in %.4f s (%.2f Hz)\n' % (vals.shape[0], etime - stime, vals.shape[0]/(etime-stime)))

                if len(vals.shape) > 3:
                    vals = vals[:,0]
                np_litpix[m, pmin:pmax] = (vals>self.thresh).sum((1,2))
                if c > 2:
                    break

    @classmethod
    def cmode(self, inp):
        assert inp.shape == (512,128)
        mod = np.copy(inp)
        mod = mod.reshape(8,64,2,64).transpose(1,3,0,2).reshape(64,64,16)
        mod -= np.median(mod, axis=(0,1))
        mod = mod.reshape(64,64,8,2).transpose(2,0,3,1).reshape(512,128)
        return mod

    def _part_worker(self, p, m, offsets, mask, litpix):
        np_litpix = np.frombuffer(litpix.get_obj(), dtype='u8')

        nframes = self.dshape[1]
        my_start = (nframes // self.nproc) * p
        my_end = min((nframes // self.nproc) * (p+1), nframes)
        num_chunks = int(np.ceil((my_end-my_start)/self.chunk_size))
        if p == 0:
            print('Doing %d chunks of %d frames each' % (num_chunks, self.chunk_size))

        with h5py.File(self.vds_file, 'r') as f:
            for c in range(num_chunks):
                pmin = my_start + c*self.chunk_size
                pmax = min(my_start + (c+1) * self.chunk_size, my_end)
                sel_cells = np.arange(pmin, pmax) % 176
                
                stime = time.time()
                vals = f[self.dset_name][m, pmin:pmax]
                etime = time.time()
                if p == 0:
                    sys.stderr.write('(%.4d/%.4d) %d frames in %.4f s (%.2f Hz)\n' % (c, num_chunks, vals.shape[0], etime - stime, vals.shape[0]*self.nproc/(etime-stime)))

                if self.is_raw:
                    cvals = (vals[:,0] - offsets[sel_cells]) * mask[sel_cells]
                    # Common mode over whole module
                    #cvals = (cvals.transpose(1,2,0) - np.median(cvals, axis=(1,2))).transpose(2,0,1)
                    # Common mode over ASICS
                    cvals = np.array([self.cmode(cv) for cv in cvals])
                else:
                    cvals = vals
                np_litpix[pmin:pmax] = (cvals>self.thresh).sum((1,2))

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Lit pixel calculator')
    parser.add_argument('vds_file', help='Path to VDS HDF5 file for run')
    parser.add_argument('-n', '--nproc', help='Number of processes to use', type=int, default=0)
    parser.add_argument('-m', '--module', help='Run on only this module (much faster)', type=int, default=-1)
    parser.add_argument('-t', '--thresholdADU', help='ADU threshold for lit pixel', type=float, default=25.)
    parser.add_argument('-c', '--calibString', help='Glob string to Cheetah*h5 calib files (default:latest)', default='')
    args = parser.parse_args()

    if args.calibString != '':
        l = LitPixels(args.vds_file, nproc=args.nproc, thresh=args.thresholdADU, calib_glob=args.calibString)
    else:
        l = LitPixels(args.vds_file, nproc=args.nproc, thresh=args.thresholdADU)
    if args.module == -1:
        l.run()
    else:
        l.run_module(args.module)

if __name__ == '__main__':
    main()
