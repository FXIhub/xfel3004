'''
This is a conversion of an IDL script from Anton Barty. 
The original is part of the XFEL Cheetah template.

File format definitions:

Steffen agipd_offset_store
    $ h5ls -r data/agipd_offset_store_r0176_r0177_r0178.h5 | head -n 20
    /                        Group
    /Q1M1                    Group
    /Q1M1/Noise              Group
    /Q1M1/Noise/0            Group
    /Q1M1/Noise/0/data       Dataset {128, 512, 32, 3}
    /Q1M1/Offset             Group
    /Q1M1/Offset/0           Group
    /Q1M1/Offset/0/data      Dataset {128, 512, 32, 3}
    /Q1M1/Threshold          Group
    /Q1M1/Threshold/0        Group
    /Q1M1/Threshold/0/data   Dataset {128, 512, 32, 2}

Steffen agipd_base_store
    $ h5ls -r data/agipd_base_store_r0176_r0177_r0178.h5 | head -n 20
    /                        Group
    /Q1M1                    Group
    /Q1M1/BaseOffset         Group
    /Q1M1/BaseOffset/0       Group
    /Q1M1/BaseOffset/0/data  Dataset {128, 512, 32, 3}
    /Q1M1/BaseOffset/0/mask  Dataset {128, 512, 32, 2}
    /Q1M1/RelativeGain       Group
    /Q1M1/RelativeGain/0     Group
    /Q1M1/RelativeGain/0/data Dataset {128, 512, 32, 3}
    /Q1M1/RelativeGain/0/mask Dataset {128, 512, 32, 3}
    /Q1M1/Threshold          Group
    /Q1M1/Threshold/0        Group
    /Q1M1/Threshold/0/data   Dataset {128, 512, 32, 2}
    /Q1M1/Threshold/0/mask   Dataset {128, 512, 32, 3}

Manuela dark [THE ONLY ONE THIS SCRIPT CURRENTLY PROCESSES]
    $ h5ls -r dark_AGIPD00_xfel_2017-09-17.h5 
    /                        Group
    /gainlevel_mean          Dataset {3, 30, 512, 128}
    /offset                  Dataset {3, 30, 512, 128}
    /run_number              Dataset {3}
    /stddev                  Dataset {3, 30, 512, 128}
    /threshold               Dataset {2, 30, 512, 128}

Manuela pcdrs
    $ h5ls -r kuhnm-pcdrs/r0488-r0489-r0490-r0491-r0492-r0493-r0494-r0495/pcdrs_joined_constants_xfel.h5
    /                        Group
    /channel00               Group
    /channel00/collection    Group
    /channel00/collection/creation_date Dataset {SCALAR}
    /channel00/collection/run_number Dataset {1}
    /channel00/gainlevel_mean Dataset {2, 32, 512, 128}
    /channel00/offset        Dataset {2, 32, 512, 128}
    /channel00/slope         Dataset {2, 32, 512, 128}
    /channel00/threshold     Dataset {1, 32, 512, 128}
    ...
'''

import os
import sys
import configparser
import numpy as np
import h5py
from PyQt5 import QtGui

class GenConstants():
    def __init__(self, plotter=None, imview=None):
        self.plotter = plotter
        self.imview = imview
        self.hist_offsets = []
        self.hist_thresh = []
        
    def _process_offset(self, manuela_dark):
        '''
        process offsets
        Currently, just returning output from Manuela's processing.
        
        Original comment for posterity:
        From Steffen:
             offset from darks (O_d) and base offset (O_b) are rescaled using:
                offset = O_d
                do = np.median(O_b-O_d, axis=(0,1))
                offset[...,1:] -= do[...,1:]
            so for high gain the dark derived offset is directly used, for the other a module median of the difference between
            base offset and dark derived offset is created on a per-memory cell basis.
        '''
        print('Processing offsets:')
        return manuela_dark.astype('f4')

    def _adjust_offset(self, offset, gain, offset_correction):
        '''
        Adjust the offsets if needed
         This will change the zero intercept of each of the gain stages, improving the
        transition between gain stages when using calibration with only dark files

        '''
        print('Adusting offsets: ')
        print('offset_correction = ', offset_correction)
        
        try:
            if len(offset_correction) == 1:
                offset_correction = np.repeat(offset_correction, 3)
        except TypeError:
            offset_correction = np.repeat(offset_correction, 3)

        if len(offset_correction) != 3:
            print('Fault: offset correction array of a confusing size (need either 1 or 3). Skipping.')
            return offset
        offset_correction = np.array(offset_correction) # shape: (gain,)

        # gain shape: (gain, cell, ss, fs)
        # offset shape: (gain, cell, ss, fs)
        m = np.median(gain, axis=(2,3)) # shape: (gain, cell)
        offset_shift = (offset_correction / m.T).T # shape: (gain, cell)
        offset = (offset.transpose(2,3,0,1) - offset_shift).transpose(2,3,0,1)

        return offset

    def _set_nominal_gain(self, data):
        '''
        Nominal values:
            HG: Cf = 60fF        (1x)
            MG: Cf = 2 pF        (38x)
            LG: Cf = 10pF        (180x)
        '''
        print('Setting nominal gains:')
        out = np.ones(data.shape, dtype='f4')
        out[1] = 38
        out[2] = 180
        return out

    def _process_thresh(self, dark_thresh, g3_disable=False):
        '''
        Gain switch threshold 
        Either take native value and reject outliers (current implementation),
        or consider the width 'noise' and average it
        Need to take the low-side of the curve, not the median.
        Midpoint of two distributions probably the best for G1/G2 threshold
        '''
        print('Processing gain threshold:')
        if g3_disable:
            print("Disabling 3rd gain stage")
        
        s = dark_thresh.shape

        # We use a 3-level output array
        # Just in case we later want to do +/- some value for gain1 stage
        out = np.empty((3,s[1],s[2],s[3]), dtype='f4')
        out[0] = 0
        out[1] = dark_thresh[0]

        # Diable 3rd gain stage by setting the threshold value to a ridiculous value
        if g3_disable:
            out[2] = 32000
        else:
            out[2] = dark_thresh[1]
        
        return out

    def _process_badpix_offs(self, data):
        '''
        Find bad pixels based on offset outliers
        Currently set to 4 * sigma, where sigma itself is calculated
        after two rounds of outlier rejection.
        '''
        print('Processing bad pixels (offset):')
        s = data.shape

        badpix = np.zeros(s, dtype='u1')
        ngain = s[0]
        ncell = s[1]

        for g in range(ngain):
            for c in range(ncell):
                temp = data[g, c]
                m = np.median(temp)

                # First cycle of median and sigma calc.
                sel = np.abs(temp-m) <= 1000
                sigma = temp[sel].std()
                m = np.median(temp[sel])

                # Second cycle of median and sigma calc.
                sel = np.abs(temp-m) < 3*sigma
                sigma = temp[sel].std()
                m = np.median(temp[sel])

                badpix[g, c, np.abs(temp-m) > 4*sigma] = 1
                if g > 0:
                    badpix[g, c] = badpix[g, c] | badpix[g-1, c]

        return badpix

    def _process_badpix_gthresh(self, gmean, gthresh):
        '''
        Find bad pixels based on gain threshold outliers
        Bad pixel if gain threshold lower than mean gain value
        '''
        print('Processing bad pixels (gainthresh):')
        s = gmean.shape
        badpix = np.zeros(s, dtype='u1')

        badpix[1:][gthresh[:2] < gmean[:2]] = 1
        return badpix

    def _suppress_dodgy_asics(self, badpix):
        '''
        Kill entire ASICs that have many dodgy pixels 
        The rest are probably bad too
        '''
        print('Suppressing dodgy ASICS in module:')
        s = badpix.shape

        # Initial axes: (gain, cell, ss, fs)
        # 64 ASICs per module (8x8)
        # Reshape to: (gain, cell, 8, asicss, 8, asicfs)
        # Transpose to: (gain, cell, 8, 8, asicss, asicfs)
        badpix = badpix.reshape(s[0], s[1], 8, s[2]//8, 8, s[3]//8).transpose(0,1,2,4,3,5)

        # Step 1
        # Look for ASICs with a high number of bad pixels
        # If more than 50% of pixels in an ASIC are bad the others are probably dodgy too
        # Label the whole ASIC as dead
        fbad_asic1 = (badpix != 0).mean((4,5)) # Output: (gain, cell, 8, 8)
        badpix[fbad_asic1 > 0.5,:,:] = 1

        # Step #2
        # If the ASIC is bad for most of the memory cells, declare the whole thing bad
        # Call it bad if any of the gain stages are dodgy
        fbad_asic2 = badpix.max(axis=0).mean((0,3,4))
        badpix[:,:,fbad_asic2 > 0.5,:,:] = 1
        
        # Undo transpose and reshape
        badpix = badpix.transpose(0,1,2,4,3,5).reshape(s)
        return badpix

    def process_module(self, manuela_dark_file, file_format, module_name, outdir=None,
                       g3_disable=False, offset_correction=None):
        '''
        Code to generate constants for one module
        '''
        #
        #   File name and HDF5 field manipulation
        #
        
        # Output directory
        if outdir is None:
            outdir = 'calib'
        os.makedirs(outdir, exist_ok=True)

        # Cheetah output file name
        # AGIPD00 --> Cheetah-AGIPD00-calib.h5
        outfile = 'Cheetah-'+module_name+'-calib.h5'
        outfile = os.path.join(outdir, outfile)


        # Manuelas HDF5 field names
        # AGIPD00 --> /channel00
        manuela_h5_field = module_name.replace('AGIPD', '/channel')
        
        # Version specific stuff        
        if file_format == 'XFEL2012':
            manuela_dark_file = ['dark_AGIPD00_xfel', 'dark_AGIPD01_xfel', 'dark_AGIPD02_xfel', 'dark_AGIPD03_xfel',
                                 'dark_AGIPD04_xfel', 'dark_AGIPD05_xfel', 'dark_AGIPD06_xfel', 'dark_AGIPD07_xfel',
                                 'dark_AGIPD08_xfel', 'dark_AGIPD09_xfel', 'dark_AGIPD10_xfel', 'dark_AGIPD11_xfel',
                                 'dark_AGIPD12_xfel', 'dark_AGIPD13_xfel', 'dark_AGIPD14_xfel', 'dark_AGIPD15_xfel']

        # Sanity checking
        print('********************************  ', module_name, '  ********************************')
        print('Processing: ', module_name)
        print('    manuela_dark_file:    ', manuela_dark_file)

        print('    manuela_h5_field:     ', manuela_h5_field)
        print('    outfile:              ', outfile)
        print('---')

        #
        #   File reading section
        #

        # Read Manuela's dark calibration data
        print('Reading: ', manuela_dark_file, '    h5_block=', manuela_h5_field)
        with h5py.File(manuela_dark_file, 'r') as fptr:
            m_dark_thresh = fptr[manuela_h5_field+'/threshold'][:]
            m_dark_offset = fptr[manuela_h5_field+'/offset'][:]
            m_dark_stddev = fptr[manuela_h5_field+'/stddev'][:]
            m_dark_gainlevel_mean = fptr[manuela_h5_field+'/gainlevel_mean'][:]
        
        # Transpose from FSDS layout if we are reading a *_agipd.h5 input file
        #if 1 then begin
        if manuela_dark_file.endswith('agipd.h5'):
            m_dark_thresh = m_dark_thresh.transpose(1,0,2,3)
            m_dark_offset = m_dark_offset.transpose(1,0,2,3)
            m_dark_stddev = m_dark_stddev.transpose(1,0,2,3)
            m_dark_gainlevel_mean = m_dark_gainlevel_mean.transpose(1,0,2,3)
            
            m_dark_thresh = m_dark_thresh[:,::-1]
            m_dark_offset = m_dark_offset[:,::-1]
            m_dark_stddev = m_dark_stddev[:,::-1]
            m_dark_gainlevel_mean = m_dark_gainlevel_mean[:,::-1]
        
        #
        #   Do the actual processing
        # 
        
        # Offsets read by Cheetah are currently:  3 x <n_cells> x 512 x 128 of type int16_t
        offset_out = self._process_offset(m_dark_offset).astype('i4')
        if self.plotter is not None:
            self.plotter.plotItem.clear()
            self.plotter.plotItem.setTitle('Offset for %s' % module_name)
            for i in range(3):
                hy, hx = np.histogram(offset_out[i].ravel(), bins=np.arange(3000,14000,10))
                self.plotter.plotItem.plot(hx, hy, stepMode=True, fillLevel=None)
            QtGui.QApplication.processEvents()

        # Gains will be the median of module for that memory cell
        # (Currently just setting nominal gain)
        gain_out = self._set_nominal_gain(m_dark_offset).astype('f4')

        # Gain thresholds from Manuela
        thresh_out = self._process_thresh(m_dark_thresh, g3_disable=g3_disable).astype('i4')
        if self.plotter is not None:
            self.plotter.plotItem.clear()
            self.plotter.plotItem.setTitle('Thresholds for %s' % module_name)
            for i in range(1, 3):
                hy, hx = np.histogram(thresh_out[i].ravel(), bins=np.arange(3000,14000,10))
                self.plotter.plotItem.plot(hx, hy, stepMode=True, fillLevel=None)
            QtGui.QApplication.processEvents()

        # Manual adjustment to offsets 
        # (Note: must be done after gains are set)
        if offset_correction is not None:
            offset_out = self._adjust_offset(offset_out, gain_out, offset_correction).astype('i4')

        # Bad pixels map - very important !!
        badpix_out1 = self._process_badpix_offs(offset_out)
        badpix_out3 = self._process_badpix_gthresh(m_dark_gainlevel_mean, m_dark_thresh)
        badpix_out = badpix_out1 | badpix_out3
        badpix_out = self._suppress_dodgy_asics(badpix_out)
        badpix_out = badpix_out.astype('u1')
        if self.imview is not None:
            self.imview.setImage(np.sign(badpix_out.sum(1)))

        #
        # Save result in Cheetah format
        #
        print('Writing: ', outfile)
        with h5py.File(outfile, 'w') as fptr:
            fptr['AnalogOffset'] = offset_out
            fptr['RelativeGain'] = gain_out
            fptr['DigitalGainLevel'] = thresh_out
            fptr['Badpixel'] = badpix_out

    def quick_agipd_calib(self, manuela_dark_file, conf='exp.ini'):
        '''
        Process all modules
        Changes in file naming and HDF5 field conventions are all done one up from this layer
        From here on in changes should only occur with changes in file format
        '''
        # Names of modules to process
        modules_to_process = ['AGIPD00', 'AGIPD01', 'AGIPD02', 'AGIPD03',
                              'AGIPD04', 'AGIPD05', 'AGIPD06', 'AGIPD07',
                              'AGIPD08', 'AGIPD09', 'AGIPD10', 'AGIPD11',
                              'AGIPD12', 'AGIPD13', 'AGIPD14', 'AGIPD15']

        # EuXFEL HDF5 file labels things differently, these labels can not be easily derived
        steffen_h5_field = ['/Q1M1', '/Q1M2', '/Q1M3', '/Q1M4',
                            '/Q2M1', '/Q2M2', '/Q2M3', '/Q2M4',
                            '/Q3M1', '/Q3M2', '/Q3M3', '/Q3M4',
                            '/Q4M1', '/Q4M2', '/Q4M3', '/Q4M4']

        print('Modules to process:')
        print('    ', modules_to_process)

        print('Files to process:')
        print('    manuela_dark_file =    ', manuela_dark_file)


        # Allow for changing file formats
        file_format = 'XFEL2066'
        print('file_format = ', file_format)

        #
        #    Check input files exist
        #
        if not os.path.exists(manuela_dark_file):
            print('Error: can not find file', manuela_dark_file)
            return

        config = configparser.ConfigParser()
        config.read(conf)
        g3_disable = config.getboolean('constants', 'g3_disable', fallback=False)
        offset_correction = [int(o) for o in config.get('constants', 'offset_correction', fallback='0 0 0').split()]
        outdir = config.get('constants', 'output_dir', fallback=os.path.basename(os.path.dirname(manuela_dark_file)))

        if offset_correction is not None:
            print("Offset correction: ", offset_correction)
        
        if g3_disable:
            print('Disabling 3rd gain stage')

        # Loop through all modules
        for module in modules_to_process:
            self.process_module(manuela_dark_file, file_format, module, outdir=outdir,
                                g3_disable=g3_disable, offset_correction=offset_correction)

        print('Done (clean exit)')

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate Cheetah-style calibration constants')
    parser.add_argument('input_file', help='Input file from parsing XFEL data (currently using AGIPD toolbox)')
    parser.add_argument('-c', '--config_file', help='Path to config file. Default: exp.ini', default='exp.ini')
    args = parser.parse_args()

    gc = GenConstants()
    gc.quick_agipd_calib(args.input_file, conf=args.config_file)

if __name__ == '__main__':
    main()
