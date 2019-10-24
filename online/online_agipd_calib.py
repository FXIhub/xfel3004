import h5py
import numpy as np
import xfel_online as xo

class AGIPD_Calibrator_Simple:
    def __init__(self, filename):
        with h5py.File(filename, "r") as file_handle:
            self._mean = np.transpose(file_handle["data/mean"][4, ...], axes=(2, 1, 0))
            self._sigma = np.transpose(file_handle["data/sigma"][4, ...], axes=(2, 1, 0))

    def calibrate_train(self, adu_data, gain_data):
        npulses = adu_data.shape[-1]
        return (adu_data - self._mean[:, :, :npulses])



class AGIPD_Calibrator:
    def __init__(self, filenames, max_pulses=120, pulse_filter=None):
        self._nCells = 250
        self._badpixData       = []
        self._darkOffsetData   = []
        self._relativeGainData = []
        self._gainLevelData    = []
        # Hardcoded pulse filter
        # self._pulse_filter = np.ones(self._nCells, dtype='bool')

        if pulse_filter is None:
            self._pulse_filter = np.ones(self._nCells, dtype='bool')
        else:
            self._pulse_filter = pulse_filter

        # self._pulse_filter = np.zeros(self._nCells, dtype='bool')
        # self._pulse_filter[1::2] = True

        # self._pulse_filter[max_pulses:] = False
        # self._pulse_filter[0] = False
        # self._pulse_filter[18::32] = False
        # self._pulse_filter[29::32] = False

        for filename in sorted(filenames):
            self._read_and_append_calibration_data(filename=filename)
        self._badpixData       = np.asarray(self._badpixData)
        self._darkOffsetData   = np.asarray(self._darkOffsetData)
        self._relativeGainData = np.asarray(self._relativeGainData)
        self._gainLevelData    = np.asarray(self._gainLevelData)

    def _read_and_append_calibration_data(self, filename):
        # Meaning of indices: (gainID, cellID, pixcol, pixrow)
        # Anton's AGIPD calibration format
        #> h5ls calib/agipd/Cheetah-AGIPD00-calib.h5
        #AnalogOffset             Dataset {3, 176, 512, 128} H5T_STD_I16LE
        #Badpixel                 Dataset {3, 176, 512, 128} H5T_STD_U8LE
        #DigitalGainLevel         Dataset {3, 176, 512, 128} H5T_STD_U16LE
        #RelativeGain             Dataset {3, 176, 512, 128} H5T_IEEE_F32LE
        with h5py.File(filename) as f:
            self._badpixData.append(np.asarray(f["Badpixel"]))
            self._darkOffsetData.append(np.asarray(f["AnalogOffset"]))
            self._relativeGainData.append(np.asarray(f["RelativeGain"]))
            self._gainLevelData.append(np.asarray(f["DigitalGainLevel"]))
            # meaning of each dimensions: (null, gain level, y, x, cell id)
            self._badpixData = np.transpose(np.array(self._badpixData), axes=(0, 1, 4, 3, 2))
            self._darkOffsetData = np.transpose(np.array(self._darkOffsetData), axes=(0, 1, 4, 3, 2))
            self._relativeGainData = np.transpose(np.array(self._relativeGainData), axes=(0, 1, 4, 3, 2))
            self._gainLevelData = np.transpose(np.array(self._gainLevelData), axes=(0, 1, 4, 3, 2))
            # self._darkOffsetData = np.transpose(np.array(self._darkOffsetData), axes=(0, 1, 4, 3, 2))[...,self._pulse_filter]
            # self._relativeGainData = np.transpose(np.array(self._relativeGainData), axes=(0, 1, 4, 3, 2))[...,self._pulse_filter]
            # self._gainLevelData = np.transpose(np.array(self._gainLevelData), axes=(0, 1, 4, 3, 2))[...,self._pulse_filter]

    #def calibrate_train(self, evt, aduData, apply_gain_switch=True):
    def calibrate_train(self, aduData, gainData, apply_gain_switch=False):
        npulses = aduData.shape[-1]
        outData = aduData.copy()
        darkoffset = self._darkOffsetData[..., :npulses]
        relativegain = self._relativeGainData[...,:npulses]
        badpixdata = self._badpixData[...,:npulses]
        gainleveldata = self._gainLevelData[...,:npulses]
        badpixMask = np.ones(aduData.shape, dtype=np.bool)

        if apply_gain_switch:
            outData -= darkoffset[-1][0][:len(outData)]
            outData[gainData > gainleveldata[0,1]] = 1000
            badpixMask = (1 - badpixdata[0,0])
        else:
            pixGainLevel0 = gainData < gainleveldata[0,1]
            pixGainLevel2 = gainData > gainleveldata[0,2]
            pixGainLevel1 = ~(pixGainLevel0 | pixGainLevel2)
            #pixGainLevel1 = pixGainLevel0 == False
            #pixGainLevel2 = np.zeros(shape=gainData.shape, dtype='bool')
            pixGainLevels = [pixGainLevel0, pixGainLevel1, pixGainLevel2]

            for g, pixGain in enumerate(pixGainLevels):
                if not pixGain.any():
                    continue
                outData[pixGain] = (outData[pixGain] - darkoffset[0,g,pixGain]) * relativegain[0,g,pixGain]
                #outData[pixGain] *= (badpixdata[0,g,pixGain] == 0)
                badpixMask[pixGain] = (badpixdata[0,g,pixGain] == 0)
        return outData, badpixMask

    def calibrate_train_fast(self, aduData, gainData, apply_gain_switch=False):
        """
        gainData.ndim == 3 # y, x, cell id
        aduData.data.ndim == 3 # y, x, cell id
        """

        npulses = aduData.shape[-1]
        outData = aduData.copy()
        darkoffset = self._darkOffsetData[..., :npulses]
        relativegain = self._relativeGainData[...,:npulses]
        badpixdata = self._badpixData[...,:npulses] # dimensions: (null, gain level, y, x, cell id)
        gainleveldata = self._gainLevelData[...,:npulses] # dimensions: (null, gain level, y, x, cell id)
        badpixMask = np.ones(aduData.shape, dtype=np.bool)
        #print("before", outData.sum(), outData.shape)
        if apply_gain_switch:
            outData -= darkoffset[-1][0][:len(outData)]
            outData[gainData > gainleveldata[0,1]] = 1000
            badpixMask = (1 - badpixdata[0,0])
        else:
            xo.correctAGIPD(outData, badpixMask, gainData, gainleveldata[0], darkoffset[0], relativegain[0], badpixdata[0])
        return outData, badpixMask


def common_mode_correction(inarray, mask, d=64):
    """
    inarray: input array of shape (Y,X, cells)
    mask: a boolean mask
    d: size of sqaure asic

    returns per-asic common-mode corrected version of a
    """
    a = inarray.copy()
    nmask = ~mask
    a[nmask] = np.nan
    m, n, l = a.shape
    b = a.transpose(2, 0, 1).reshape(l, m, n//d, d).transpose(0, 3, 2, 1).reshape(l, d, n // d, m // d, d).transpose(0, 2, 3, 1, 4)
    asic_sum = np.nanmean(b,axis=(3,4))
    asic_median = np.nanmedian(b, axis=(3, 4))
    # Use 10 percentile
    #asic_median = np.nanpercentile(b, 10, axis=(3, 4))
    b -= asic_median[:, :, :, np.newaxis, np.newaxis]
    a[nmask] = inarray[nmask]
    return a, asic_sum, asic_median


def common_mode_correction_twopass(inarray, mask, d=64):
    a = inarray.copy()
    nmask = ~mask
    a[nmask] = np.nan
    m, n, l = a.shape
    b = a.transpose(2, 0, 1).reshape(l, m, n//d, d).transpose(0, 3, 2, 1).reshape(l, d, n // d, m // d, d).transpose(0, 2, 3, 1, 4)
    asic_sum = np.nanmean(b,axis=(3,4))
    asic_median = np.nanmedian(b, axis=(3, 4))

    b -= np.nanpercentile(b, 10, axis=(3, 4))[:, :, :, np.newaxis, np.newaxis]
    adu_threshold = 35
    lit_pixels = b > adu_threshold

    c = b.copy()
    c[lit_pixels] = np.nan
    b -= np.median(c, axis=(3, 4))[:, :, :, np.newaxis, np.newaxis]

    a[nmask] = inarray[nmask]
    return a, asic_sum, asic_median
