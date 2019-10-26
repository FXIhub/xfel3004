#!/usr/bin/env python

'''
Create calibrated shots and powder sums from AGIPD VDS files
Author: Jonas Sellberg, Kartik Ayyer
'''

import sys
import h5py
import numpy as np
import glob
import multiprocessing as mp
import ctypes
import geom
import argparse

class AGIPD_VDS_Calibrator():
    '''
    Interface to get frames interactively
    Initially specify path to folder with raw/proc data
    Then use get_frame(num) to get specific frame
    '''
    def __init__(self, fname, raw=True, calib_run=None, good_cells=range(176), chunk_size=None, verbose=0,
            geom_fname='/gpfs/exfel/exp/SPB/201901/p002316/scratch/geom/b1.geom'):
        self.num_h5cells = 176
        self.chunk_size = chunk_size
        self.verbose = verbose
        self.good_cells = np.array(good_cells)
        self.geom_fname = geom_fname
        self.is_raw_data = raw
        if self.geom_fname is not None:
            self.x, self.y = geom.pixel_maps_from_geometry_file(geom_fname)
        calib_glob = None
        if self.is_raw_data and calib_run is None:
            calib_glob='/gpfs/exfel/exp/SPB/201901/p002316/usr/Shared/calib/latest/Cheetah*.h5'
        elif self.is_raw_data:
            calib_glob='/gpfs/exfel/exp/SPB/201901/p002316/scratch/calib/r%.4d/Cheetah*.h5' % calib_run
        self.dset_name = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/image/data'
        self.train_name = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/image/trainId'
        self.pulse_name = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/image/pulseId'
        self.cell_name = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/image/cellId'
        self._load_vds(fname, calib_glob)
        self.frame = np.empty((16,512,128))
        self.powder = None
        self.train_ids = None
        self.cell_ids = None
        self.pulse_ids = None
        
    def __enter__(self):
        return self
    
    def _load_vds(self, fname, calib_glob):
        try:
            self.vds = h5py.File(fname, 'r')
        except IOError:
            print('ERROR: file does not exist: %s' % fname)
            sys.exit(-1)
        self.nframes = self.vds[self.dset_name].shape[1]
        if self.verbose > 0:
            print('VDS file contains %d shots' % self.nframes)
        
        if calib_glob is not None:
            self.calib = [h5py.File(f, 'r') for f in sorted(glob.glob(calib_glob))]
        if self.verbose > 0:
            print('%d calibration files found'%len(self.calib))
    
    def __exit__(self, exc_type, exc_value, traceback):
        self._close_vds()
    
    def _close_vds(self):
        try:
            self.vds.close()
        except ValueError:
            print('WARNING: VDS file has already been closed')

    def _calibrate_powder(self, data_sum, module, cell, gain_mode=0, nframes=1):
        offset = self.calib[module]['AnalogOffset'][gain_mode,cell]
        badpix = self.calib[module]['Badpixel'][gain_mode,cell]
        gain = self.calib[module]['RelativeGain'][gain_mode,cell]
        data = (np.float32(data_sum) - nframes*offset)*gain
        data[badpix != 0] = 0
        return data
    
    def _calibrate_module(self, data, gain, module, cell, cmode=False, photonThresh=0.75):
        if np.all(data == 65535):
            data[:] = np.nan
            return data
        gain_mode = self._threshold(gain, module, cell)
        offset = np.empty(gain_mode.shape)
        gain = np.empty(gain_mode.shape)
        badpix = np.empty(gain_mode.shape)
        for i in range(3):
            offset[gain_mode==i] = self.calib[module]['AnalogOffset'][i,cell][gain_mode==i]
            gain[gain_mode==i] = self.calib[module]['RelativeGain'][i,cell][gain_mode==i]
            badpix[gain_mode==i] = self.calib[module]['Badpixel'][i,cell][gain_mode==i]
        data = (np.float32(data) - offset)*gain
        data[badpix != 0] = 0
        if self.verbose > 1:
            print('Found %d bad pixels for module %d' % ((badpix != 0).sum(), module))
        if cmode:
            # Median subtraction by 64x64 asics
            data = data.reshape(8,64,2,64).transpose(1,3,0,2).reshape(64,64,16)
            if self.verbose > 1:
                print('Common-mode correction for module %d: %.1f ADU' % (module, np.sum(np.median(data, axis=(0,1)))))
            data -= np.median(data, axis=(0,1))
            data = data.reshape(64,64,8,2).transpose(2,0,3,1).reshape(512,128)
        # Threshold below 0.5-0.7 photon (1 photon = 45 ADU)
        if self.verbose > 1:
            print('Setting %d pixels below photon threshold to zero for module %d' % (((data < photonThresh*45*gain) & (data > 0)).sum(), module))
        data[data < photonThresh*42*gain] = 0
        #data[data > 10000] = 10000

        return data

    def _threshold(self, gain, module, cell):        
        threshold = self.calib[module]['DigitalGainLevel'][:,cell]
        high_gain = gain < threshold[1]
        low_gain = gain > threshold[2]
        medium_gain = (~high_gain) * (~low_gain)
        return low_gain*2 + medium_gain

    def _get_frames(self, num, type='frame', calibrate=False, threshold=False, assemble=True):
        assert len(num.shape) == 1, "Must contain a 1D array of integers"
        for n in num:
            if n > self.nframes or n < 0:
                print('Out of range: %d, skipping event..' % n)
                num = np.delete(num, np.where(num == n))
        # frame or frames?
        if num.shape[0] > 1:
            self.frame = np.empty((num.shape[0],16,512,128))
        # these will not be the same as the IDs in the VDS file, depends on the selection of good_cells
        #cell_ind = num % len(self.good_cells)
        #train_ind = num // len(self.good_cells)
        #ind = self.good_cells[cell_ind] + train_ind * self.num_h5cells
        # stick to VDS file for IDs
        self.cell_ids = self.vds[self.cell_name][:][num]
        self.pulse_ids = self.vds[self.pulse_name][:][num]
        self.train_ids = self.vds[self.train_name][:][num]
        if type == 'frame':
            type_ind = 0
            threshold = False
        elif type == 'gain':
            type_ind = 1
            calibrate = False
        else:
            print('Unknown type string: %s' % type)
            return
        data = self.vds[self.dset_name][:,num]
        for n in range(num.shape[0]):
            if self.verbose:
                print('Getting frame with cellId=%d, pulseId=%d and trainId=%d' % (self.cell_ids[n], self.pulse_ids[n], self.train_ids[n]))
            if calibrate:
                if self.verbose:
                    print('Calibrating frame %d' % num[n])
                for i in range(16):
                    if num.shape[0] > 1:
                        self.frame[n,i] = self._calibrate_module(data[i,n,0,:,:], data[i,n,1,:,:], i, self.cell_ids[n])
                    else:
                        self.frame[i] = self._calibrate_module(data[i,n,0,:,:], data[i,n,1,:,:], i, self.cell_ids[n])
            else:
                if self.is_raw_data:
                    if num.shape[0] > 1:
                        self.frame[n] = data[:,n,0,:,:]
                    else:
                        self.frame = data[:,n,0,:,:]
                else:
                    if num.shape[0] > 1:
                        self.frame[n] = data[:]
                    else:
                        self.frame = data[:]
        if not assemble or self.geom_fname is None:
            return np.copy(self.frame)
        else:
            if num.shape[0] > 1:
                output = []
                for n in range(num.shape[0]):
                    if self.verbose:
                        print('Assembling frame %d' % num[n])
                    output.append(geom.apply_geom_ij_yx((self.y, self.x), self.frame[n]))
                return np.array(output)
            else:
                return geom.apply_geom_ij_yx((self.y, self.x), self.frame)
    
    def get_frame(self, num, calibrate=True, assemble=True):
        if type(num) is int:
            num = np.array([num])
        elif type(num) is float:
            num = np.array([int(num)])
        elif type(num) is list:
            num = np.array(num, dtype=np.int)
        return self._get_frames(num, type='frame', calibrate=calibrate, assemble=assemble)

    def get_gain(self, num, threshold=False, assemble=True):
        if type(num) is int:
            num = np.array([num])
        elif type(num) is float:
            num = np.array([int(num)])
        elif type(num) is list:
            num = np.array(num, dtype=np.int)
        return self._get_frames(num, type='gain', calibrate=False, threshold=threshold, assemble=assemble)

    def get_powder(self):
        if self.powder is not None:
            print('Powder sum already calculated')
            return self.powder
        
        powder_shape = (len(self.good_cells),) + self.frame.shape
        powder = mp.Array(ctypes.c_double, len(self.good_cells)*self.frame.size)
        jobs = []
        for i in range(16):
            p = mp.Process(target=self._powder_worker, args=(i, powder, powder_shape))
            jobs.append(p)
            p.start()
        for j in jobs:
            j.join()
        #sys.stderr.write('\n')
        self.powder = np.frombuffer(powder.get_obj()).reshape(powder_shape)
        self.npowder = np.zeros_like(self.good_cells)
        # calibrate powder
        for k,cell in enumerate(self.good_cells):
            ind = np.zeros((self.nframes,), dtype=np.bool)
            ind[cell::self.num_h5cells] = True
            self.npowder[k] = ind.sum()
            if self.is_raw_data:
                for i in range(16):
                    self.powder[k,i,:,:] = self._calibrate_powder(self.powder[k,i,:,:], i, cell, nframes=self.npowder[k])
        return (self.powder, self.npowder)
    
    def _powder_worker(self, i, powder, shape):
        np_powder = np.frombuffer(powder.get_obj()).reshape(shape)
        
        # For each cell
        for k,cell in enumerate(self.good_cells):
            if self.is_raw_data:
                np_powder[k,i] += self.vds[self.dset_name][i][cell::self.num_h5cells,0,:,:].sum(0)
                #np_powder[k,i] += self._calibrate_powder(self.vds[self.dset_name][i][cell::self.num_h5cells,0,:,:].sum(0), i, cell, nframes=self.npowder[k])
                #np_powder[k,i] += self.vds[self.dset_name][i][cell::self.num_h5cells,0,:,:].sum(0)
            else:
                np_powder[k,i] += self.vds[self.dset_name][i][cell::self.num_h5cells,:,:].sum(0)
        


def write_chunk(args, chunk, frames, vds_fname):

    import sys
    import glob
    # The following line works on Maxwell
    sys.path.append('/home/ayyerkar/.local/dragonfly/utils/py_src')
    import writeemc
    import os


    if 0 <= args.res < 3:
        post_tag = ['_lowq.h5', '_medq.h5', '_allq.h5'][args.res]
        # Module order matches geometry file
        modules = [[3,4,8,15], [2,3,4,5,9,8,15,14], np.arange(16)][args.res]
        subset = [128, 256, 0][args.res]
        num_pix = [4*128*128, 4*256*256, 4*512*512][args.res]
    else:
        print('"res" parameter can only have values 0, 1 or 2')
        sys.exit(1)

    shift = args.threshold - 0.5
    emcfile = os.path.join(args.out_folder, "r%04d_%04d" %(args.run, chunk) + post_tag)
    emc = writeemc.EMCWriter(emcfile, num_pix)
    photon_ADU = 45.

    print('Calibrating virtual data set for run %d' % args.run)
    with AGIPD_VDS_Calibrator(vds_fname, good_cells=np.where(good_cells)[0], calib_run=args.calib_run, verbose=int(args.verbose)) as c:
        print('Calibrating %s frames from run %d' % (len(frames), args.run))
        frame = c.get_frame(frames, calibrate=True, assemble=False)
        print('Converting %d frames from run %d' % (len(frames), args.run))
        nframes = frame.shape[0]
        for i in range(nframes):
            emc.write_frame(np.round(frame[i][modules,-subset:]/photon_ADU - shift).ravel().astype('i4'))
            sys.stderr.write('\r%d/%d'%(i+1, nframes))
        sys.stderr.write('\n')
    emc.finish_write()
    os.system('chmod ao+rw %s' % (emcfile))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create calibrated AGIPD VDS files')
    parser.add_argument('run', help='Run number', type=int)
    parser.add_argument('-c', '--chunk', help='Chunk size (default=2000)', type=int, default=2000)
    parser.add_argument('-C', '--calib_run', help='Calibration run number (default:latest)', default=None)
    parser.add_argument('-v', '--verbose', help='Output additional information (default=False)', default=False, action='store_true')
    parser.add_argument('-s', '--skip', help='Skip pulses', default=1, type=int)
    parser.add_argument('-m', '--max_pulses', help='Maximum nr. of pulses', default=176, type=int)
    parser.add_argument('-r', '--res', help='Resolution to save to (0=lowq, 1=medq, 2=allq). Default=0', type=int, default=0)
    parser.add_argument('-o', '--out_folder', help='Path to output folder')
    parser.add_argument('-t', '--threshold', help='Photon conversion threshold (fraction of photon). Default=0.7', type=float, default=0.7)
    args = parser.parse_args()

    npulses = 128
    pulseskip = args.skip
    maxpulse = args.max_pulses
    good_cells = np.zeros(npulses ,dtype=np.bool)
    good_cells[1:pulseskip*maxpulse+1:pulseskip] = True
    good_cells[18::32] = False

    import os, sys
    hlname = '/gpfs/exfel/exp/SPB/201901/p002316/scratch/litpixels/r%04d_hits.h5' % args.run
    if not os.path.exists(hlname):
        print('No hitlist available, run: python litpixels.py r%04d_vds_raw.h5' % args.run)
        sys.exit(0)
    else:
        with h5py.File(hlname, 'r') as hl:
            lp = hl['litpixels_15'][:]
            print(list(hl))
            bin_values, bin_edges = np.histogram(lp, bins=(lp.max()-lp.min())); 
        bin_centers = np.array([(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_values))])
            
        good_data = lp.reshape(-1,128)
        good_data = good_data[:,good_cells].ravel()
        sel = ((good_data > 0) & (good_data < 25000))
        med = np.median(good_data[sel])
        for _ in range(10):
            sigma = good_data[sel].std()
            sel = (np.abs(good_data-med) < 3*sigma)
        threshold = med + 3*sigma
        print('threshold =', threshold)            
        frames = np.where((lp > threshold) & (lp < 25000))[0]
        print('Found %d hits above lit-pixel threshold' % len(frames))    
    vds_fname = '/gpfs/exfel/exp/SPB/201901/p002316/scratch/vds/r%04d_vds_raw.h5' % args.run

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    nchunks = len(frames)//args.chunk+1
    for i in range(nchunks)[rank::size]:
        print("rank %d is running on %d hits and saving into chunk %d" %(rank, frames[i*args.chunk:(i+1)*args.chunk].shape[0], i))
        write_chunk(args, i, frames[i*args.chunk:(i+1)*args.chunk], vds_fname)
