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
import plotting.correlation
import analysis.agipd
import analysis.event
import analysis.hitfinding
import ipc.mpi
from backend import add_record

import numpy as np
import sys, os; sys.path.append(os.path.split(__file__)[0])
from online_agipd_calib import AGIPD_Calibrator
from online_agipd_calib import common_mode_correction
# from online_agipd_calib import common_mode_correction_twopass as common_mode_correction
import xfel_online as xo


state = {}
state['Facility'] = 'EuXFELtrains'
state['EuXFEL/DataSource'] = 'tcp://10.253.0.51:45000' # Raw
# state['EuXFEL/DataSource'] = 'tcp://10.253.0.51:45011' # Calibrated
state['EuXFEL/DataSource_GMD'] = 'tcp://10.253.0.142:6666' # Hack for the GMD
state['EuXFEL/DataFormat'] = 'Raw'
state['EuXFEL/MaxTrainAge'] = 4
# state['EuXFEL/MaxPulses'] = 120
state['EuXFEL/MaxPulses'] = 137

# Use SelModule = None or remove key to indicate a full detector
# [For simulator, comment if running with full detector, otherwise uncomment]
state['EuXFEL/SelModule'] = 4


# Roundness
roundness_calculator = xo.Roundness()

# Parameters and buffers
running_background = None

# Switches
alignment  = True
hitfinding = True
background = False
statistics = True
calibrate  = True
commonmode = True
usemask = True
sizing = True

# Hitfinding parameters
adu_threshold  = 25
#adu_threshold  = 37
hit_threshold  = 400
# hit_threshold  = 100 # Trigger hits even with no beam
hit_threshold_strong = hit_threshold * 1.5
maskhit_threshold = 90
ratio_threshold = 2
sumhit_threshold = 3000
dark_threshold = 50

# Pulse filter
# base_pulse_filter = np.zeros(176, dtype="bool")
# base_pulse_filter[1::1] = True
base_pulse_filter = np.ones(176, dtype="bool")
base_pulse_filter[state['EuXFEL/MaxPulses']:] = False
base_pulse_filter[0] = False
base_pulse_filter[18::32] = False
base_pulse_filter[29::32] = False

# AGIPD calibrator
path_to_calib = "/gpfs/exfel/exp/SPB/201802/p002160/usr/Shared/calib/latest/"
calibrator = AGIPD_Calibrator([path_to_calib + "Cheetah-AGIPD04-calib.h5"], max_pulses=state['EuXFEL/MaxPulses'])

# maximum nr. of hits to be sent per train
show_max_hits = 2

# Build a mask for hitfinding
hitmask = np.zeros((128,512)).astype(np.bool)
hitmask[:64,:64] = True
hitmask = hitmask[:,:,np.newaxis]

# Sperical mask for hitsum finding
hitsum_radius = 30
cx,cy = 15, -20
xx,yy = np.meshgrid(np.arange(128)-cx, np.arange(512)-cy, indexing='ij')
hitsummask = (xx**2 + yy**2) > (hitsum_radius**2)
hitsummask = hitsummask[:,:,np.newaxis]

# Manual mask from file
read_mask = False
manual_mask = np.ones((128,512,1)).astype(np.bool)
if read_mask:
    import h5py
    badpixel_file = "/gpfs/exfel/exp/SPB/201802/p002160/usr/Shared/calib/duane_dark_r0010.h5"
    with h5py.File(badpixel_file, "r") as file_handle:
        manual_mask = file_handle["mask"][...]
    manual_mask = np.bool8(np.transpose(manual_mask, axes=(2,1,0)))#[:, :, np.newaxis])

# Building the initial mask
base_initmask = np.ones((128,512,176)).astype(np.bool)
if usemask:
    base_initmask &= manual_mask

# Size estimation parameters
binning = 1

def onEvent(evt):
    global running_background

    # Calculate number of pulses in each train
    # npulses = len(T.timestamp) #Now get that from the length of the data
    analysis.event.printProcessingRate(pulses_per_event=176, label="Processing rate (pulses):" )
    analysis.event.printProcessingRate(pulses_per_event=1, label="Processing rate (trains):" )


    # Apply the pulse mask derived from the GMD
    pulse_filter = base_pulse_filter * xo.pulses_mask(evt)
    # pulse_filter = base_pulse_filter

    # Shape of data: (module, ss, fs, cell)
    #print(evt['photonPixelDetectors']['AGIPD'].data.shape)
    agipd_data_all = evt['photonPixelDetectors']['AGIPD'].data[0,:,:,:]
    agipd_gain_all = evt['photonPixelDetectors']['AGIPD'].data[1,:,:,:]

    # Filter all data to only active pulses
    data_len = agipd_data_all.shape[-1]
    agipd_data = agipd_data_all[:, :, pulse_filter[:data_len]]
    agipd_gain = agipd_gain_all[:, :, pulse_filter[:data_len]]

    T = evt["eventID"]["Timestamp"]
    cellId = T.cellId[pulse_filter[:data_len]]
    trainId = T.trainId[pulse_filter[:data_len]]
    goodcells = T.cellId[pulse_filter[:data_len]]
    badcells = T.badCells
    

    npulses = agipd_data.shape[-1]
    if not npulses:
        return
    
    # Print statistics
    if statistics and ipc.mpi.is_main_worker():
        print("Nr. of pulses per train: {:d}".format(npulses))
        # print("Bad cells (filtered out): ", badcells)

    agipd_module = add_record(evt['analysis'], 'analysis', 'AGIPD', agipd_data)

    # Remember to remove
    # agipd_module.data[np.isnan(agipd_module.data)] = 0.

    # Dark calibration
    if calibrate:
        calibrated, badpixmask = calibrator.calibrate_train_fast(agipd_data_all, agipd_gain_all,
                                                                 apply_gain_switch=False)
        calibrated = calibrated[:, :, pulse_filter[:data_len]]
        badpixmask = badpixmask[:, :, pulse_filter[:data_len]]
        badpixmask = np.bool8(badpixmask)
        agipd_module = add_record(evt['analysis'], 'analysis', 'AGIPD/Calib', calibrated)    
    else:
        agipd_module.data[np.isnan(agipd_module.data)] = 0.
        badpixmask = np.ones((128,512,npulses)).astype(np.bool)

    initmask = base_initmask[:,:,pulse_filter[:data_len]]
    mask = (badpixmask & initmask)

    # Common-mode  correction
    if commonmode:
        common_mode_corrected, asic_sum, asic_median = common_mode_correction(agipd_module.data, mask)
        agipd_module = add_record(evt['analysis'], 'analysis', 'AGIPD/Calib', common_mode_corrected)
        asic_sum = asic_sum.flatten()
        asic_median = asic_median.flatten()
        asic_sum = add_record(evt['analysis'], 'Common Mode', 'ASIC sum', asic_sum)
        asic_median = add_record(evt['analysis'], 'Common Mode', 'ASIC median', asic_median)
        plotting.line.plotTrace(asic_sum, paramX=asic_median, name='ASIC sum vs median', group='Common Mode')#, mask=masksum.min(axis=2))        

    # Flip the X-axis
    agipd_module = add_record(evt['analysis'], 'analysis', 'AGIPD/Calib', agipd_module.data[:,::-1])
    mask = mask[:, ::-1]

    # Update masks
    maskhit = (mask & hitmask)
    masksum = (mask & hitsummask & hitmask)

    # Save image and mask
    #import h5py
    #with h5py.File("image_and_mask_%d.h5" %(trainId[0]), "w") as file_handle:
    #     file_handle.create_dataset("image_train", data=np.float64(agipd_module.data))
    #     file_handle.create_dataset("mask_train", data=np.int64(mask))

    # For alignment purposes, calculate the average signal per train
    if alignment:
        # Mean image in a train
        agipd_module_thresh = agipd_module.data.copy() #Set low values to zero for mean img
        agipd_module_thresh[agipd_module_thresh < 35] = 0
        agipd_module_thresh
        meanimg = add_record(evt['analysis'], 'analysis', 'meanimg', agipd_module_thresh.mean(axis=-1))
        plotting.image.plotImage(meanimg, group='Alignment', vmin=-10, vmax=50)#, mask=masksum.min(axis=2))

        # Single pulse image
        random_index = np.random.choice(npulses, 1)[0]
        singleimg = add_record(evt['analysis'], 'analysis', 'singleimg', agipd_module.data[:,:,random_index])
        #plotting.image.plotImage(singleimg, group='Alignment', mask=mask[:,:,random_index], vmin=0.)
        plotting.image.plotImage(singleimg, group='Alignment', vmin=0.)

        mask_record = add_record(evt['analysis'], 'analysis', 'mask', mask[:, :, random_index])
        plotting.image.plotImage(mask_record, group='Alignment')


        # Mean data intensity over all pixels in all cells of detector
        mean = add_record(evt['analysis'], 'analysis', 'mean', agipd_module.data[:,:].mean())
        plotting.line.plotHistory(mean, group='Alignment')



    # Do hit-finding on full trains
    if hitfinding:

        # Subtract the running Background
        if background:
            if running_background is None:
                running_background = np.zeros(agipd_module.data.shape[:-1])
            agipd_module.data[:] -= running_background[:, :, np.newaxis]

        # Count lit pixel in each train using only badpixel mask
        analysis.hitfinding.countLitPixels(evt, agipd_module, aduThreshold=adu_threshold, hitscoreThreshold=hit_threshold,
                                           hitscoreDark=dark_threshold,
                                           mask=mask, stack=True)
        hitscore  = evt['analysis']['litpixel: hitscore'].data
        hittrain  = np.bool8(evt['analysis']['litpixel: isHit'].data)
        misstrain = np.bool8(evt['analysis']['litpixel: isMiss'].data)


        # hittrain_strong = hitscore > hit_threshold_strong
        analysis.hitfinding.countLitPixels(evt, agipd_module, aduThreshold=adu_threshold, hitscoreThreshold=hit_threshold_strong,
                                           hitscoreDark=dark_threshold, outkey="litpixel - strong: ",
                                           mask=mask, stack=True)
        hitscore_strong  = evt['analysis']['litpixel - strong: hitscore'].data
        hittrain_strong  = np.bool8(evt['analysis']['litpixel - strong: isHit'].data)
        misstrain_strong = np.bool8(evt['analysis']['litpixel - strong: isMiss'].data)


        # Calculate hitsore left-right ratio
        left_score = ((agipd_module.data*mask)[:, :256, :] > adu_threshold).sum(axis=(0, 1))
        right_score = ((agipd_module.data*mask)[:, 256:, :] > adu_threshold).sum(axis=(0, 1))
        litpixel_ratio = add_record(evt['analysis'], 'analysis', 'litpixel: ratio', (left_score+1) / (right_score+1))
        hittrain_ratio = (litpixel_ratio.data > ratio_threshold).astype(np.bool)
        misstrain_ratio = ~hittrain_ratio

        # Count lit pixel in each train using a special hit-finding mask
        analysis.hitfinding.countLitPixels(evt, agipd_module, aduThreshold=adu_threshold, hitscoreThreshold=maskhit_threshold,
                                           hitscoreDark=dark_threshold, outkey="litpixel - masked: ",
                                           mask=maskhit, stack=True)
        hitscore_masked  = evt['analysis']['litpixel - masked: hitscore'].data
        hittrain_masked  = np.bool8(evt['analysis']['litpixel - masked: isHit'].data)
        misstrain_masked = np.bool8(evt['analysis']['litpixel - masked: isMiss'].data)

        # Hitfinding based on integrated signal
        sum_litpixel = (agipd_module.data * masksum) > adu_threshold
        sum_hitscore = (agipd_module.data * sum_litpixel).sum(axis=(0,1))
        hittrain_sum = sum_hitscore > sumhit_threshold
        misstrain_sum = ~hittrain_sum

        # Plot the hitscores for each pulse

        first_hitscore_pulse = add_record(evt['analysis'], 'analysis', 'hitscore - first', hitscore[0])
        plotting.line.plotHistory(first_hitscore_pulse, group='First', hline=hit_threshold, history=10000)

        if hitscore[0] > hit_threshold:
            first_agipd_record = add_record(evt['analysis'], 'analysis', 'AGIPD - hits - first', agipd_module.data[:,:,0])
            plotting.image.plotImage(first_agipd_record, group='First', vmin=0., mask=mask[:,:,0])

        analysis.hitfinding.hitrate(evt, hitscore[0] > hit_threshold, outkey="hitrate - first")
        if ipc.mpi.is_main_worker():
            plotting.line.plotHistory(evt['analysis']['hitrate - first'], history=10000, group='First')


        for i in range(npulses):
            # Update event ID for each pulse
            evt.event_id = lambda: T.timestamp[i]
            hitscore_pulse = add_record(evt['analysis'], 'analysis', 'hitscore', hitscore[i])
            plotting.line.plotHistory(hitscore_pulse, group='Hitfinding', hline=hit_threshold, history=10000)
            ratio_pulse = add_record(evt['analysis'], 'analysis', 'litpixel ratio', litpixel_ratio.data[i])
            plotting.line.plotHistory(ratio_pulse, group='Hitfinding', hline=ratio_threshold, history=10000)
            hitscore_masked_pulse = add_record(evt['analysis'], 'analysis', 'hitscore - masked', hitscore_masked[i])
            plotting.line.plotHistory(hitscore_masked_pulse, group='Hitfinding', hline=maskhit_threshold, history=10000)
            hitscore_sum_pulse = add_record(evt['analysis'], 'analysis', 'hitscore - integrated', sum_hitscore[i])
            plotting.line.plotHistory(hitscore_sum_pulse, group='Hitfinding', hline=sumhit_threshold, history=10000)

        # Plot the hitscore against the cellID
        cellid_record = add_record(evt['analysis'], 'analysis', 'cellid', cellId)
        plotting.line.plotTrace(evt['analysis']['litpixel: hitscore'], paramX=cellid_record, history=10000, group='Hitfinding')

        # Plot the hitscore against the cellID
        hitsum_record = add_record(evt['analysis'], 'analysis', 'litpixel: sum', sum_hitscore)
        plotting.line.plotTrace(evt['analysis']['litpixel: sum'], paramX=cellid_record, history=10000, group='Hitfinding')

        # Decide which scores to use for defining hits
        hittrain |= hittrain_masked
        hittrain |= hittrain_sum

        # Update the hitrate
        analysis.hitfinding.hitrate(evt, hittrain)
        analysis.hitfinding.hitrate(evt, hittrain_strong, outkey="hitrate - strong")
        if ipc.mpi.is_main_worker():
            plotting.line.plotHistory(evt['analysis']['hitrate'], history=10000, group='Hitfinding')
            plotting.line.plotHistory(evt['analysis']['hitrate - strong'], history=10000, group='Hitfinding')

        # Update the running background
        if background and misstrain.sum():
            background_history_ratio = 0.5
            running_background[:] = (background_history_ratio*running_background +
                                     (1-background_history_ratio)*(agipd_module.data[:, :, misstrain].mean(axis=2)+running_background))

        # These are the hits
        Nhits = hittrain.sum()
        if Nhits:
            brightest = np.argsort(hitscore[hittrain])[-1]

            # Select pulses from the AGIPD train that are hits
            agipd_hits = agipd_module.data[:,:,hittrain]

            # Select (128,128) region and do binning
            binned_pulse = agipd_hits[:,:128].reshape((128//binning,binning,128//binning,binning,Nhits)).sum(axis=(1,3))
            #print(binned_pulse.shape)



            # Plot some pretty pictures (but not more than show_max_hits)
            pick_indices = np.random.choice(Nhits, min(show_max_hits, Nhits), replace=False)
            for i in pick_indices:
                agipd_pulse = add_record(evt['analysis'], 'analysis', 'AGIPD - hits', agipd_hits[:,:,i]) # i -> brightest
                plotting.image.plotImage(agipd_pulse, group='Hitfinding', vmin=0., mask=mask[:,:,i])

                # Binned hit image
                agipd_pulse_binned = add_record(evt['analysis'], 'analysis', 'AGIPD - binned hits', binned_pulse[:,:,brightest])
                plotting.image.plotImage(agipd_pulse_binned, group='Hitfinding', vmin=0.)

        Nhits_strong = hittrain_strong.sum()
        
        sphere_hits = np.zeros(len(hittrain), dtype="bool8")
        non_sphere_hits = np.zeros(len(hittrain), dtype="bool8")
        if Nhits_strong:
            brightest_strong = np.argsort(hitscore[hittrain_strong])[-1]
            agipd_hits_strong = agipd_module.data[:, :, hittrain_strong]
            pick_indices = np.random.choice(Nhits_strong, min(show_max_hits, Nhits_strong), replace=False)
            for i in pick_indices:
                agipd_pulse_strong = add_record(evt['analysis'], 'analysis', 'AGIPD - hits - strong', agipd_hits_strong[:,:,brightest_strong]) # i -> brightest
                plotting.image.plotImage(agipd_pulse_strong, group='Hitfinding', vmin=0., mask=mask[:,:,brightest_strong])


            # Sizing
            sizing_size, sizing_score, sizing_template_radii = xo.sizingAGIPD(agipd_hits_strong.data, mask, center=(-16.4, 8.5), r0=0.01, r1=0.7, num_div=1000)
            sizing_index = sizing_score.argmax(axis=1)
            for i in range(Nhits_strong):
                size_record = add_record(evt['analysis'], 'analysis', 'sizing: size', sizing_size[i]*xo.REAL_UNIT)
                plotting.line.plotHistory(size_record, history=10000, group='Sizing')

            
            # Roundness
            roundness = roundness_calculator.inv_roundness_stack(agipd_hits_strong)
            # print(roundness)
            is_sphere = roundness < 1.8
            sphere_hits[:is_sphere.sum()] = True
            non_sphere_hits[:(~is_sphere).sum()] = True
            # analysis.hitfinding.hitrate(evt, is_sphere, outkey="Sphere ratio", history=10)
            for a in is_sphere:
                is_hit_record = add_record(evt['analysis'], 'analysis', 'Round/Strong', int(a)) # i -> brightest
                plotting.line.plotHistory(is_hit_record, history=10000, group='Roundness')
                
            # if ipc.mpi.is_main_worker():
                # plotting.line.plotHistory(evt['analysis']['Sphere ratio'], history=10000, group='Roundness')

            
            round_particle = add_record(evt['analysis'], 'analysis', 'round', agipd_hits_strong[:, :, roundness.argmin()])
            not_round_particle = add_record(evt['analysis'], 'analysis', 'not round', agipd_hits_strong[:, :, roundness.argmax()])
            if roundness.min() <= 1.8: # 1.8
                plotting.image.plotImage(round_particle, group="Roundness", vmin=0)
            if roundness.max() > 1.8:
                plotting.image.plotImage(not_round_particle, group="Roundness", vmin=0)
        analysis.hitfinding.hitrate(evt, sphere_hits, outkey="Strong sphere hit rate")
        analysis.hitfinding.hitrate(evt, non_sphere_hits, outkey="Strong non-sphere hit rate")
        if ipc.mpi.is_main_worker():
            plotting.line.plotHistory(evt['analysis']['Strong sphere hit rate'], history=10000, group='Roundness')
            plotting.line.plotHistory(evt['analysis']['Strong non-sphere hit rate'], history=10000, group='Roundness')

        # These are the misses
        Nmiss = misstrain.sum()
        if Nmiss:
            agipd_misses = agipd_module.data[:,:,misstrain]
            pick_misses = np.random.choice(Nmiss, min(show_max_hits, Nmiss), replace=False)
            for i in pick_misses:
                agipd_pulse_miss = add_record(evt['analysis'], 'analysis', 'AGIPD - miss', agipd_misses[:,:,i])
                plotting.image.plotImage(agipd_pulse_miss, group='Hitfinding', vmin=0.)
