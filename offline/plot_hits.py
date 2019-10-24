#!/usr/bin/env python

'''
Plot calibrated hits from VDS files
Author: Jonas Sellberg
'''

import h5py
import numpy as np
import argparse
import sys
import os
from matplotlib import pyplot as plt
from matplotlib import colors
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create calibrated AGIPD VDS files')
    parser.add_argument('run', help='Get specific run, can also be a set of runs, for example: 1,3,5-10,20,22')
    parser.add_argument('-P', '--plot', help='Plot the first X number of hits (default=all)', type=int, default=0)
    parser.add_argument('-i', '--in_folder', help='Path of output folder (default=/gpfs/exfel/exp/SPB/201901/p002316/scratch/hits/)', default='/gpfs/exfel/exp/SPB/201901/p002316/scratch/hits/')
    parser.add_argument('-o', '--out_folder', help='Path of output folder (default=./)', default='./')
    parser.add_argument('-v', '--verbose', help='Output additional information (default=False)', default=False, action='store_true')
    args = parser.parse_args()

    if args.run is not None:
        runs = []
        tmp = args.run
        for s in tmp.split(','):
            if "-" in s:
                rmin, rmax = s.split('-')
                runs += list(range(int(rmin), int(rmax)+1))
            else:
                runs += [int(s)]
    if not os.path.isdir(args.out_folder):
        os.mkdir(args.out_folder)
        if args.verbose:
            print('Created folder: %s' % args.out_folder)
    for r in runs:
    	fname = '%s/r%04d_hits.h5' % (args.in_folder, r)
    	if not os.path.exists(fname):
    	        print('WARNING: hits file does not exist, skipping %s' % fname)
    	        continue
    	with h5py.File(fname, 'r') as f:
    	    lp = f['litpixels_04'][:]
    	    threshold = f['hits/litpixelThreshold'][()]
    	    frames = np.where(lp > threshold)[0]
    	    if len(frames) > 0:
    	        try:
    	            hits = f['hits/assembled']
    	            nhits = hits.shape[0]
    	        except KeyError:
    	            nhits = 0
    	    else: nhits = 0
    	    if args.verbose:
    	        print('Found %d hits from run %d in: %s' % (nhits, r, fname))
    	    if args.plot < 1 or args.plot > nhits:
    	        nhits_to_plot = nhits
    	    else:
    	        nhits_to_plot = args.plot
            print('Plotting %d hits from run %d in: %s' % (nhits_to_plot, r, fname))
    	    # plot shots in photons [gain = 45 ADU]
    	    for n in range(nhits_to_plot):
    	        fig = plt.figure(n)
    	        #plt.imshow(hits[n]/45., norm=colors.LogNorm(vmin=0.1, vmax=100), interpolation='nearest', cmap='jet')
    	        #plt.imshow(hits[n], norm=colors.LogNorm(vmin=0.1, vmax=1000))
    	        plt.imshow(hits[n]/45., norm=colors.LogNorm(vmin=0.1, vmax=100))
    	        plt.xlim(420, 620)
    	        plt.ylim(560, 720)
    	        ax = fig.gca()
    	        circ = plt.Circle(((hits[n].shape[1]-1)/2., (hits[n].shape[0]-1)/2.), radius=1, linewidth=1, color='k')
    	        circ.set_fill(True)
    	        ax.add_patch(circ)
    	        plt.title('run %d - shot %d - %d lit pixels' % (r, frames[n], lp[frames][n]))
    	        plt.colorbar()
    	        plt.savefig('%s/run%d_shot%d_zoom.png' % (args.out_folder, r, frames[n]))
    	        if args.verbose:
    	            print('\tSaved figure: run%d_shot%d_zoom.png' % (r, frames[n]))
    	        plt.close()
