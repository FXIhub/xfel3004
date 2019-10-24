#!/usr/bin/env python
'''
Convert calibrated AGIPD data to sparse data format.
Author: Jonas Sellberg, Kartik Ayyer, Benedikt Daurer
'''

import sys
import h5py
import numpy as np
import glob
import multiprocessing as mp
import ctypes
import geom
import agipd_vds
import argparse

def write_chunk(args, chunk, frames, vds_fname, out):

    import sys
    import glob
    # The following line works on Maxwell
    sys.path.append('/home/ayyerkar/.local/dragonfly/utils/py_src')
    import writeemc
    import os

    post_tag = ['_lowq.h5', '_medq.h5', '_allq.h5'][2]
    # Module order matches geometry file
    modules = [[3,4,8,15], [2,3,4,5,9,8,15,14], np.arange(16)][2]
    subset = [128, 256, 0][2]
    num_pix = [4*128*128, 4*256*256, 4*512*512][2]
    
    shift = args.threshold - 0.5
    emcfile = os.path.join(out, "r%04d_%06d" %(args.run, chunk) + post_tag)
    emc = writeemc.EMCWriter(emcfile, num_pix)
    photon_ADU = 45.

    print('Calibrating virtual data set for run %d' % args.run)
    with agipd_vds.AGIPD_VDS_Calibrator(vds_fname, calib_run=args.calib_run, verbose=int(args.verbose)) as c:
        print('Calibrating %s frames from run %d' % (len(frames), args.run))
        frame = c.get_frame(frames, calibrate=True, assemble=False)
        if (frame is None):
            print("Chunk %04d is empty" %chunk)
            os.system('rm %s' % (emcfile))
            return
        print('Converting %d frames from run %d' % (len(frames), args.run))
        nframes = frame.shape[0]
        for i in range(nframes):
            emc.write_frame(np.round(frame[i][modules,-subset:]/photon_ADU - shift).ravel().astype('i4'))
            sys.stderr.write('\r%d/%d'%(i+1, nframes))
        sys.stderr.write('\n')
        emc.finish_write()
        with h5py.File(emcfile, 'a') as f:
            f['id/cells'] = c.cell_ids
            f['id/pulses'] = c.pulse_ids
            f['id/trains'] = c.train_ids
    print("Chunk %04d is finished" %chunk)
    os.system('chmod ao+rw %s' % (emcfile))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert calibrated AGIPD frames to sparse format')
    parser.add_argument('run', help='Run number', type=int)
    parser.add_argument('-c', '--chunk', help='Chunk size (default=2000)', type=int, default=2000)
    parser.add_argument('-s', '--start', help='Start from this chunk index (default=0)', type=int, default=0)
    parser.add_argument('-C', '--calib_run', help='Calibration run number (default:latest)', default=None)
    parser.add_argument('-v', '--verbose', help='Output additional information (default=False)', default=False, action='store_true')
    parser.add_argument('-o', '--out_folder', help='Path to output folder')
    parser.add_argument('-t', '--threshold', help='Photon conversion threshold (fraction of photon). Default=0.7', type=float, default=0.7)
    parser.add_argument('-d', '--dry-run', help='Dry run with printouts (default=False)', default=False, action='store_true')
    args = parser.parse_args()

    vds_fname = '/gpfs/exfel/exp/SPB/201901/p002316/scratch/vds/r%04d_vds_raw.h5' % args.run
    with h5py.File(vds_fname, 'r') as f:
        ndata = f['INSTRUMENT/SPB_DET_AGIPD1M-1/DET/image/data'].shape[1]
    good_cells = np.zeros(ndata, dtype=np.bool).reshape(-1,176)
    good_cells[:,:150] = True
    good_cells[:,18::32] = False
    idxlist = np.arange(ndata)[good_cells.ravel()]
    ntrains = good_cells.shape[0]
    print("Run %04d, converting %d events from %d trains" %(args.run, len(idxlist), ntrains))
    if args.dry_run:
        sys.exit(0)
    out = args.out_folder + 'r%04d' %args.run
    import os
    try:
        os.mkdir(out)
    except FileExistsError:
        pass
        
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    nchunks = len(idxlist)//args.chunk+1
    for i in range(args.start,nchunks)[rank::size]:
        if (i*args.chunk < len(idxlist)):
            print("rank %d is running on %d events and saving into chunk %d" %(rank, idxlist[i*args.chunk:(i+1)*args.chunk].shape[0], i))
            write_chunk(args, i, idxlist[i*args.chunk:(i+1)*args.chunk], vds_fname, out)
