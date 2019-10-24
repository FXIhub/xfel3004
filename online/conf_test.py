"""
For testing the EuXFEL backend, start the karabo server:

./karabo-bridge-server-sim 1234
    OR
./karabo-bridge-server-sim -d AGIPDModule -r 1234

from the karabo-bridge (https://github.com/European-XFEL/karabo-bridge-py).
and then start the Hummingbird backend:

./hummingbird.py -b examples/euxfel/mock/conf.py
"""
import plotting.image
import plotting.line
import analysis.agipd
import analysis.event
import analysis.hitfinding
import ipc.mpi
from backend import add_record

import numpy as np
import sys, os; sys.path.append(os.path.split(__file__)[0])
from online_agipd_calib import AGIPD_Calibrator_Simple
from online_agipd_calib import common_mode_correction
import xfel_online as xo


state = {}
# state['Facility'] = 'EuXFELtrains'
state['Facility'] = 'EuXFEL'
state['EventIsTrain'] = True
state['EuXFEL/DataSource'] = 'tcp://10.253.0.51:45000' # Raw (and single module)
# state['EuXFEL/DataSource'] = 'tcp://10.253.0.51:45011' # Calibrated (and all modules)
# state['EuXFEL/DataSource'] = 'tcp://10.253.0.61:45000' # Test
# state['EuXFEL/DataSource'] = 'tcp://exflonc08.desy.de:1237' # Karabo bridge server
state['EuXFEL/DataSource_GMD'] = 'tcp://10.253.0.142:6666' # Hack for the GMD
state['EuXFEL/DataFormat'] = 'Raw'
state['EuXFEL/MaxTrainAge'] = 4
# state['EuXFEL/MaxPulses'] = 120
state['EuXFEL/MaxPulses'] = 8
state['EuXFEL/SelModule'] = 4


readout_length = state['EuXFEL/MaxPulses']

# Switches
do_calibrate = True
do_hitfinding = True
do_sizeing = True
do_alignment = True

# Binning
binning = 8

print("try read calibrator")
# AGIPD calibrator
path_to_calib = "/gpfs/exfel/exp/SPB/201901/p002316/usr/calib/"
calibrator = AGIPD_Calibrator_Simple(path_to_calib + "dark_r0004.h5")

print("read calibrator")

# Building the initial mask
base_initmask = np.ones((128, 512, readout_length-1)).astype(np.bool)
file_initmask = np.load("initial_mask.npy")
base_initmask *= file_initmask[:, :, np.newaxis]

base_pulse_filter = np.ones(readout_length, dtype="bool")
base_pulse_filter[state['EuXFEL/MaxPulses']:] = False
base_pulse_filter[0] = False
base_pulse_filter[18::32] = False
base_pulse_filter[29::32] = False

# maximum nr. of hits to be sent per train
show_max_hits = 2

# Hitfinding parameters
adu_threshold  = 160
hit_threshold  = 15000
dark_threshold = 50

adu_per_photon = 37.5

def onEvent(evt):
    global running_background

    # analysis.event.printProcessingRate(pulses_per_event=readout_length, label="Processing rate (pulses):" )
    analysis.event.printProcessingRate(pulses_per_event=1, label="Processing rate (trains):" )

    # Apply the pulse mask derived from the GMD
    pulse_filter = base_pulse_filter * xo.pulses_mask(evt, length=readout_length)

    # Shape of data: (module, ss, fs, cell)
    #print(evt['photonPixelDetectors']['AGIPD'].data.shape)
    agipd_data_all = evt['photonPixelDetectors']['AGIPD'].data[0,:,:,:]
    agipd_gain_all = evt['photonPixelDetectors']['AGIPD'].data[1,:,:,:]

    # print("agipd_data_all: ", agipd_data_all.shape)
    # agipd_data_all.dump("raw_detector.npy")
    
    data_len = agipd_data_all.shape[-1]

    # Dark calibration
    if do_calibrate:
        # calibrated, badpixmask = calibrator.calibrate_train(agipd_data_all, agipd_gain_all)
        calibrated = calibrator.calibrate_train(agipd_data_all, agipd_gain_all)
        calibrated = calibrated[:, :, pulse_filter[:data_len]]
        # badpixmask = badpixmask[:, :, pulse_filter[:data_len]]
        # badpixmask = np.bool8(badpixmask)
        agipd_module = add_record(evt['analysis'], 'analysis', 'AGIPD/Calib', calibrated)
    else:
        agipd_module.data[np.isnan(agipd_module.data)] = 0.
        # badpixmask = np.ones((128,512,npulses)).astype(np.bool)

    npulses = agipd_module.data.shape[-1]
    if not npulses:
        return

    initmask = base_initmask[:,:,pulse_filter[:data_len]]
    # mask = (badpixmask & initmask)
    mask = initmask

    
    if do_alignment:
        # Integrated image
        integrated_module = (agipd_module).data.mean(axis=2)
        integrated_record = add_record(evt['analysis'], 'analysis', 'All Integrated', integrated_module)
        # plotting.image.plotImage(integrated_record, group='Alignment', vmin=0., mask=mask[:,:,0])
        plotting.image.plotImage(integrated_record, group='Alignment', vmin=0.)

        module_photons = (agipd_module.data / adu_per_photon)
        module_photons[module_photons < 0.7] = 0.
        print(module_photons.sum())
        module_photons_integrated = module_photons.mean(axis=2)
        module_photons_integrated_record = add_record(evt['analysis'], 'analysis', 'All Integrated (photons)', module_photons_integrated)
        plotting.image.plotImage(module_photons_integrated_record, group='Alignment', vmin=0.)

    # # Filter all data to only active pulses
    # data_len = agipd_data_all.shape[-1]
    # agipd_data = agipd_data_all[:, :, pulse_filter[:data_len]]
    # agipd_gain = agipd_gain_all[:, :, pulse_filter[:data_len]]


    # print("eventId: ", evt["eventId"].keys())
    # T = evt["eventID"]["Timestamp"]
    # cellId = T.cellId[pulse_filter[:data_len]]
    # trainId = T.trainId[pulse_filter[:data_len]]
    # goodcells = T.cellId[pulse_filter[:data_len]]
    # badcells = T.badCells


    # calibrated_data, bad_pixel_mask = calibrator.calibrate_train_fast(agipd_data, agipd_gain, apply_gain_switch=True)

    # Do hit-finding on full trains
    if do_hitfinding:
        # Count lit pixel in each train using only badpixel mask
        analysis.hitfinding.countLitPixels(evt, agipd_module, aduThreshold=adu_threshold, hitscoreThreshold=hit_threshold,
                                           hitscoreDark=dark_threshold,
                                           mask=mask, stack=True)
        hitscore  = evt['analysis']['litpixel: hitscore'].data
        hittrain  = np.bool8(evt['analysis']['litpixel: isHit'].data)
        misstrain = np.bool8(evt['analysis']['litpixel: isMiss'].data)

        for i in range(npulses):
            # Update event ID for each pulse
            # evt.event_id = lambda: T.timestamp[i]
            hitscore_pulse = add_record(evt['analysis'], 'analysis', 'hitscore', hitscore[i])
            plotting.line.plotHistory(hitscore_pulse, group='Hitfinding', hline=hit_threshold, history=10000)


        # These are the hits
        nhits = hittrain.sum()
        if nhits:
            brightest = np.argsort(hitscore[hittrain])[-1]

            # Select pulses from the AGIPD train that are hits
            agipd_hits = agipd_module.data[:,:,hittrain]

            # Select (128,128) region and do binning
            binned_pulse = agipd_hits[:,:128].reshape((128//binning,binning,128//binning,binning,nhits)).sum(axis=(1,3))
            #print(binned_pulse.shape)


            # Plot some pretty pictures (but not more than show_max_hits)
            pick_indices = np.random.choice(nhits, min(show_max_hits, nhits), replace=False)
            for i in pick_indices:
                agipd_pulse = add_record(evt['analysis'], 'analysis', 'AGIPD - hits', agipd_hits[:,:,i]) # i -> brightest
                plotting.image.plotImage(agipd_pulse, group='Hitfinding', vmin=0., mask=mask[:,:,i])
                # plotting.image.plotImage(agipd_pulse, group='Hitfinding', vmin=0.)

                # Binned hit image
                agipd_pulse_binned = add_record(evt['analysis'], 'analysis', 'AGIPD - binned hits', binned_pulse[:,:,brightest])
                plotting.image.plotImage(agipd_pulse_binned, group='Hitfinding', vmin=0.)

            if do_sizeing:                
                size, intensity, size_score = xo.sizingAGIPD(agipd_hits, mask[:, :, hittrain], center=(-19.4, 9.4), r0=0.01, r1=0.7, num_div=100)
                for i in range(nhits):
                    size_record = add_record(evt['analysis'], 'analysis', 'sizing: size', size[i]*xo.REAL_UNIT)
                    plotting.line.plotHistory(size_record, history=10000, group='Sizing')
                    intensity_record = add_record(evt['analysis'], 'analysis', 'sizing: intensity', intensity[i])
                    plotting.line.plotHistory(intensity_record, history=10000, group='Sizing')
                    

                # print("size: ", size)
                # print("intensity: ", intensity)
                # print("integral: ", intensity / agipd_hits.sum(axis=(0,1)))
