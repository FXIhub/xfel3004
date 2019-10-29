#!/usr/bin/env python

'''Creates VDS file with synchronized AGIPD data

There is one giant VDS data set with the dimensions:
    (module, pulse_number, <A/D>, fs, ss)
where <A/D> is an extra dimension in the raw data for the analog and digital data.
Additionally it has the corresponding train, pulse and cell IDs for all the frames.
The fill value for the detector data is NaN 
(i.e. if one or more modules does not have that train)
The fill value for the cell and pulse IDs is 65536

Run `./vds.py -h` for command line options.
'''

import sys
import os.path as op
import glob
import argparse
import numpy as np
import h5py

MAX_TRAINS_IN_FILE = 260

def main():
    parser = argparse.ArgumentParser(description='Create synchronized AGIPD VDS files')
    parser.add_argument('run', help='Run number', type=int)
    parser.add_argument('-p', '--proc', help='If proc data (default=False)', action='store_true', default=False)
    parser.add_argument('-o', '--out_folder', help='Path of output folder (default=/gpfs/exfel/u/scratch/SPB/201901/p002316/vds/)', default='/gpfs/exfel/u/scratch/SPB/201901/p002316/vds/')
    args = parser.parse_args()

    npulses = 128
    if not args.proc:
        folder = '/gpfs/exfel/exp/SPB/201901/p002316/raw/r%.4d/'%args.run
    else:
        folder = '/gpfs/exfel/exp/SPB/201901/p002316/proc/r%.4d/'%args.run

    ntrains = -1
    ftrain = sys.maxsize
    for m in range(16):
        tmin = sys.maxsize
        tmax = 0
        flist = sorted(glob.glob(folder+'/*AGIPD%.2d*.h5'%m))
        for fname in flist:
            with h5py.File(fname, 'r') as f:
                tid = f['INDEX/trainId'][:]
                if tid.max() - tid.min() > MAX_TRAINS_IN_FILE:
                    print('WARNING: Too large trainId range in %s (%d)' % (op.basename(fname), tid.max()-tid.min()))
                    if tid.min() > 0:
                        tmin = min(tmin, tid.min())
                    if fname == flist[-1]:
                        tmax = max(tmax, tid[-1])
                    continue
                tmin = min(tmin, tid.min())
                tmax = max(tmax, tid.max())
                ftrain = min(ftrain, tmin)
            print(fname, ftrain, tmax-tmin)
            sys.stdout.flush()
        ntrains = max(ntrains, tmax-tmin)
    ntrains = int(ntrains) + 4
    ltrain = ftrain + ntrains
    print(ntrains, 'trains in run starting from', ftrain)
    all_trains = np.repeat(np.arange(ftrain, ftrain+ntrains, dtype='u8'), npulses)
    all_cells = np.tile(np.arange(npulses, dtype='u8'), ntrains)
    all_flatid = (all_trains - ftrain)*npulses + all_cells

    fname = glob.glob(folder+'/*AGIPD00*.h5')[0]
    with h5py.File(fname, 'r') as f:
        det_name = list(f['INSTRUMENT'])[0]
        dshape = f['INSTRUMENT/'+det_name +'/DET/0CH0:xtdf/image/data'].shape[1:]
    print('Shape of data in', det_name, 'is', dshape)

    if not args.proc:
        out_fname = op.join(args.out_folder, 'r%.4d_vds_raw.h5'%args.run)
    else:
        out_fname = op.join(args.out_folder, 'r%.4d_vds_proc.h5'%args.run)
    outf = h5py.File(out_fname, 'w', libver='latest')
    outf['INSTRUMENT/'+det_name+'/DET/image/trainId'] = all_trains

    layout_data = h5py.VirtualLayout(shape=(16, ntrains*npulses) + dshape, dtype=np.uint16)
    outdset_cid = outf.create_dataset('INSTRUMENT/'+det_name+'/DET/image/cellId',
                                      shape=(ntrains*npulses,), dtype='u2',
                                      data=65535*np.ones(ntrains*npulses, dtype='u2'))
    outdset_pid = outf.create_dataset('INSTRUMENT/'+det_name+'/DET/image/pulseId',
                                      shape=(ntrains*npulses,), dtype='u8',
                                      data=65535*np.ones(ntrains*npulses, dtype='u8'))
    for m in range(16):
        flist = sorted(glob.glob(folder+'/*AGIPD%.2d*.h5'%m))
        for fname in flist:
            dset_prefix = 'INSTRUMENT/'+det_name+'/DET/%dCH0:xtdf/image/'%m
            with h5py.File(fname, 'r') as f:
                # Annoyingly, raw data has an extra dimension for the IDs
                #   (which is why we need the ravel)
                tid = f[dset_prefix+'trainId'][:].ravel()
                cid = f[dset_prefix+'cellId'][:].ravel()
                flatid = (tid-ftrain)*npulses + cid
                # Remove the following bad data:
                #   Train ID = 0, suggesting no input from AGIPD
                #   Train ID out of range, for bit flips from the trainID server
                #   Repeated train IDs: Keep only first train with that ID
                sel = (tid>0) & (tid<ltrain)
                uniq, nuniq = np.unique(tid, return_counts=True, return_index=True)[1:]
                for i in uniq[nuniq>npulses]:
                    print('WARNING: Repeated train IDs in %s from ind %d' % (op.basename(fname), i))
                    sel[np.where(tid==tid[i])[0][npulses:]] = False
                if sel.sum() == 0:
                    continue
                tid = tid[sel]
                indices = np.where(np.in1d(all_trains, tid))[0]

                dset = f[dset_prefix+'data']
                chunk_size = 32
                num_chunks = int(np.ceil(dset.shape[0] / chunk_size))
                for chunk in range(num_chunks):
                    st, en = chunk*chunk_size, min(dset.shape[0], (chunk+1)*chunk_size)
                    chunk_sel = np.zeros_like(sel)
                    chunk_sel[st:en] = sel[st:en]
                    chunk_ind = np.where(np.in1d(all_flatid, flatid[chunk_sel]))[0]
                    if chunk_sel.sum() != len(chunk_ind):
                        print('Mismatch: %s %d/%d (%d vs %d)'%(fname, chunk, num_chunks, chunk_sel.sum(), len(chunk_ind)))
                        return
                    if chunk_sel.sum() == 0:
                        continue
                    if args.proc:
                        vsource_data = h5py.VirtualSource(dset)[chunk_sel,:,:]
                    else:
                        vsource_data = h5py.VirtualSource(dset)[chunk_sel,:,:,:]
                    layout_data[m, chunk_ind] = vsource_data

                cid = f[dset_prefix+'cellId'][:].ravel()[sel]
                pid = f[dset_prefix+'pulseId'][:].ravel()[sel]
                sel_indices = np.zeros(len(all_trains), dtype=np.bool)
                sel_indices[indices] = True
                outdset_cid[sel_indices] = cid
                outdset_pid[sel_indices] = pid
                print(fname, len(indices), '/', dset.shape)
                sys.stdout.flush()

    outf.create_virtual_dataset('INSTRUMENT/'+det_name+'/DET/image/data', layout_data, fillvalue=np.iinfo(np.uint16).max)
    outf.close()

if __name__ == '__main__':
    main()
