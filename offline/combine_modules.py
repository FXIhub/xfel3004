#!/usr/bin/env python

'''
Offline event getter
Author: Kartik Ayyer, Filipe Maia
'''

import sys
import h5py
import numpy as np
import glob
import multiprocessing as mp
import ctypes
import geom

class AGIPD_Combiner():
    '''
    Interface to get frames interactively
    Initially specify path to folder with raw/proc data
    Then use get_frame(num) to get specific frame
    '''
    def __init__(self, run, raw=True, calib_run=None, good_cells=range(176), verbose=0,
            geom_fname='/gpfs/exfel/exp/SPB/201901/p002316/scratch/ayyerkar/ana/geometry/b1.geom'):
        self.num_h5cells = 176
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
            calib_glob='/gpfs/exfel/exp/SPB/201901/p002316/usr/Shared/calib/r%.4d/Cheetah*.h5'%calib_run
        self._make_flist(run, calib_glob)
        self._get_nframes_list()
        self.frame = np.empty((16,512,128))
        self.powder = None
        self.train_ids = None
        
    def _make_flist(self, run, calib_glob):
        if self.is_raw_data:
            folder_path = '/gpfs/exfel/exp/SPB/201901/p002316/raw/r%.4d/'%run
        else:
            folder_path = '/gpfs/exfel/exp/SPB/201901/p002316/proc/r%.4d/'%run
        self.flist = np.array([np.sort(glob.glob('%s/*-AGIPD%.2d*.h5'%(folder_path, r))) for r in range(16)])
        
        try:
            assert len(self.flist.shape) == 2
        except AssertionError:
            print('WARNING: Each module does not have the same number of files')
            print([len(f) for f in self.flist])
        if self.verbose > 0:
            print('%d files per module' % len(self.flist[0]))

        if calib_glob is not None:
            self.calib = [h5py.File(f, 'r') for f in sorted(glob.glob(calib_glob))]
        if self.verbose > 0:
            print('%d calibration files found'%len(self.calib))

    def _get_nframes_list(self):
        module_nframes = np.zeros((16,), dtype='i4')
        self.nframes_list = []
        self.first_module = -1
        for i in range(16):
            if len(self.flist[i]) > 0 and self.first_module == -1:
                self.first_module = i
            for fname in self.flist[i]:
                with h5py.File(fname, 'r') as f:
                    try:
                        dset_name = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/%dCH0:xtdf/image/data'%i
                        module_nframes[i] += f[dset_name].shape[0] / self.num_h5cells * len(self.good_cells)
                        if i == self.first_module:
                            self.nframes_list.append(f[dset_name].shape[0])
                    except KeyError:
                        print(fname)
                        raise
        try:
            assert np.all(module_nframes == module_nframes[0])
        except AssertionError:
            print('Not all modules have the same frames')
            print(module_nframes)
        if self.verbose > -1:
            print('%d good frames in run' % module_nframes.max())
        self.nframes = module_nframes.max()
        self.nframes_list = np.cumsum(self.nframes_list)

    def _calibrate(self, data, gain, module, cell, cmode=True):        
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
        #data[data > 10000] = 10000

        if cmode:
            # Median subtraction by 64x64 asics
            data = data.reshape(8,64,2,64).transpose(1,3,0,2).reshape(64,64,16)
            data -= np.median(data, axis=(0,1))
            data = data.reshape(64,64,8,2).transpose(2,0,3,1).reshape(512,128)
        
        return data

    def _threshold(self, gain, module, cell):        
        threshold = self.calib[module]['DigitalGainLevel'][:,cell]
        high_gain = gain < threshold[1]
        low_gain = gain > threshold[2]
        medium_gain = (~high_gain) * (~low_gain)
        return low_gain*2 + medium_gain

    def _get_frame(self, num, type='frame', calibrate=False, threshold=False, sync=True, assemble=True):
        if num > self.nframes or num < 0:
            print('Out of range')
            return
        
        if not sync:
            shift = 0
        cell_ind = num % len(self.good_cells)
        train_ind = num // len(self.good_cells)
        
        ind = self.good_cells[cell_ind] + train_ind * self.num_h5cells
        if type == 'frame':
            type_ind = 0
            threshold = False
        elif type == 'gain':
            type_ind = 1
            calibrate = False
        else:
            print('Unknown type string: %s' % type)
            return
        
        file_num = np.where(ind < self.nframes_list)[0][0]
        if file_num == 0:
            frame_num = ind 
        else:
            frame_num = ind - self.nframes_list[file_num-1]
        for i in range(16):
            if len(self.flist[i]) == 0:
                self.frame[i] = np.zeros_like(self.frame[0])
                continue
            with h5py.File(self.flist[i][file_num], 'r') as f:
                dset_name = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/%dCH0:xtdf/image/data'%i
                cell_name = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/%dCH0:xtdf/image/cellId'%i
                train_name = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/%dCH0:xtdf/image/trainId'%i
                if self.is_raw_data:
                    if sync:
                        if i == self.first_module:
                            trainid = f[train_name][frame_num].astype('i8')[0]
                            cellid = f[cell_name][frame_num].astype('i8')[0]
                            shift = 0
                        else:
                            shift = (trainid - f[train_name][frame_num].astype('i8')[0]) * self.num_h5cells
                            shift += (cellid - f[cell_name][frame_num].astype('i8')[0])
                        if frame_num+shift > f[dset_name].shape[0]:
                            print('WARNING: Not syncing in last train')
                            shift = 0
                    data = f[dset_name][frame_num+shift, type_ind]
                    if calibrate:
                        data = self._calibrate(data,
                                               f[dset_name][frame_num+shift,1],
                                               i, cellid)
                    if threshold:
                        data = self._threshold(data, i, cellid)
                else:
                    if sync:
                        if i == self.first_module:
                            trainid = f[train_name][frame_num].astype('i8')
                            cellid = f[cell_name][frame_num].astype('i8')
                            shift = 0
                        else:
                            shift = (trainid - f[train_name][frame_num].astype('i8')) * self.num_h5cells
                            shift += (cellid - f[cell_name][frame_num].astype('i8'))
                        if frame_num+shift > f[dset_name].shape[0]:
                            print('Not syncing in last train')
                            shift = 0
                    data = f[dset_name][frame_num + shift]
                    data[data>1.e9] = 0
                    data[data<-1.e6] = 0
                self.frame[i] = data
        if not assemble or self.geom_fname is None:
            return np.copy(self.frame)
        else:
            return geom.apply_geom_ij_yx((self.x, self.y), self.frame)

    def get_ids(self):
        if self.train_ids is not None:
            return
        self.train_ids = np.empty((0,), dtype='u8')
        self.pulse_ids = np.empty((0,), dtype='u8')
        self.cell_ids = np.empty((0,), dtype='u8')
        for fname in self.flist[self.first_module]:
            with h5py.File(fname, 'r') as f:
                cell_name = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/%dCH0:xtdf/image/cellId'%self.first_module
                pulse_name = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/%dCH0:xtdf/image/cellId'%self.first_module
                train_name = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/%dCH0:xtdf/image/trainId'%self.first_module
                self.train_ids = np.append(self.train_ids, f[train_name][:].reshape(-1,self.num_h5cells)[:,self.good_cells].flatten())
                self.pulse_ids = np.append(self.pulse_ids, f[pulse_name][:].reshape(-1,self.num_h5cells)[:,self.good_cells].flatten())
                self.cell_ids = np.append(self.cell_ids, f[cell_name][:].reshape(-1,self.num_h5cells)[:,self.good_cells].flatten())

    def get_frame_id(self, num):
        cell_ind = num % len(self.good_cells)
        train_ind = num // len(self.good_cells)
        ind = self.good_cells[cell_ind] + train_ind * self.num_h5cells
        file_num = np.where(ind < self.nframes_list)[0][0]
        if file_num == 0:
            frame_num = ind 
        else:
            frame_num = ind - self.nframes_list[file_num-1]
        with h5py.File(self.flist[self.first_module][file_num], 'r') as f:
            cell_name = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/%dCH0:xtdf/image/cellId'%self.first_module
            pulse_name = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/%dCH0:xtdf/image/pulseId'%self.first_module
            train_name = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/%dCH0:xtdf/image/trainId'%self.first_module
            train_id = f[train_name][frame_num]
            pulse_id = f[pulse_name][frame_num]
            cell_id = f[cell_name][frame_num]

        shifts = np.zeros((16,), dtype='i8')
        for i in range(16):
            with h5py.File(self.flist[i][file_num], 'r') as f:
                cell_name = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/%dCH0:xtdf/image/cellId'%i
                train_name = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/%dCH0:xtdf/image/trainId'%i
                shifts[i] = (train_id - f[train_name][frame_num].astype('i8')) * self.num_h5cells
                shifts[i] += (cell_id - f[cell_name][frame_num].astype('i8'))
        if self.is_raw_data:
            train_id = train_id[0]
            pulse_id = pulse_id[0]
            cell_id = cell_id[0]
            
        return {'train_id': train_id, 
                'cell_id': cell_id, 
                'pulse_id': pulse_id,
                'shift': shifts}

    def get_frame(self, num, calibrate=False, sync=True, assemble=True):
        return self._get_frame(num, type='frame', calibrate=calibrate, sync=sync, assemble=assemble)

    def get_gain(self, num, threshold=False, sync=True, assemble=True):
        return self._get_frame(num, type='gain', calibrate=False, threshold=threshold, sync=sync, assemble=assemble)

    def get_powder(self, minADU=0):
        if self.powder is not None:
            print('Powder sum already calculated')
            return self.powder
        
        powder_shape = (len(self.good_cells),) + self.frame.shape
        print('powder_shape', powder_shape)
        powder = mp.Array(ctypes.c_double, len(self.good_cells)*self.frame.size)
        jobs = []
        for i in range(16):
            p = mp.Process(target=self._powder_worker, args=(i, powder, powder_shape, minADU))
            jobs.append(p)
            p.start()
        for j in jobs:
            j.join()
        sys.stderr.write('\n')
        self.powder = np.frombuffer(powder.get_obj()).reshape(powder_shape)
        
        return self.powder

    def _powder_worker(self, i, powder, shape, minADU):
        dset_name = '/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/%dCH0:xtdf/image/data'%i
        np_powder = np.frombuffer(powder.get_obj()).reshape(shape)
        
        # For each file with module i
        for j in range(len(self.flist[i])):
            with h5py.File(self.flist[i][j] , 'r') as f:
                # For each cell
                for k,cell in enumerate(self.good_cells):
                    ind = np.zeros((f[dset_name].shape[0],), dtype=np.bool)
                    ind[cell::self.num_h5cells] = True
                    analog = f[dset_name][cell::self.num_h5cells,0,:,:]
                    digital = f[dset_name][cell::self.num_h5cells,1,:,:]
                    for p in range(analog.shape[0]):
                        data = self._calibrate(analog[p], digital[p], i, cell)
                        data[data<minADU] = 0
                        np_powder[k,i] += data
                    if i == self.first_module:
                        sys.stderr.write('\rModule %d: (%d, %d)'%(i,j,k))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Format: %s <run_number>'%sys.argv[0])
        sys.exit(1)
    run = int(sys.argv[1])
    print('Calculating powder sum for run %d'%run)
    
    #c = AGIPD_Combiner(int(sys.argv[1]), good_cells=list(range(2,62,2)))
    c = AGIPD_Combiner(int(sys.argv[1]))
    c.get_powder(minADU=30)
    
    import os
    if not os.path.isdir('data'):
        os.mkdir('data')
    f = h5py.File('data/powder_r%.4d.h5'%int(sys.argv[1]), 'w')
    f['data'] = c.powder
    f.close()

