import numpy as np
from cpplib import radialAverage, correctAGIPD, radialM2, angularAverage
from scipy.special import jn
import scipy.ndimage

def ballRadialIntensity(fluence, size, pixel):
    p = pixel * size
    return fluence * p ** (-3) * jn(1.5, p) **2

class _BallTemplate:
    def __init__(self):
        self._last_input = (None, None)
        self._result = None

    def __call__(self, rs, r):
        if not (np.all(rs == self._last_input[0]) and
                np.all(r == self._last_input[1])):
            self._last_input = (rs, r)
            ball_fft = np.zeros(( len(rs), len(r)))
            for i, rad in enumerate(rs):
                ball_fft[i] = ballRadialIntensity(10, rad, r)
            self._result = ball_fft
        return self._result
ballTemplate = _BallTemplate()

def sizingAGIPD(hits, mask, center=(-19.4, 9.4), r0=0.01, r1=0.7, num_div=100):
    "dimension of hits and mask: (Y, X, cell Id)"
    s, c, r = radialAverage(*center, mask, hits, 1)
    s /= c
    rs = np.linspace(r0, r1, num_div)
    ball_inten = ballTemplate(rs, r)
    # ff = (s @ ball_inten.T) / np.linalg.norm(ball_inten, axis=1)
    #print(s.shape, ball_inten.shape)
    ff = (s @ ball_inten.T) / np.linalg.norm(ball_inten, axis=1)
    # ff = np.zeros((s.shape[0], ball_inten.shape[0]))
    # ball_inten = 1 / ball_inten
    # for i in range(s.shape[0]):
        # foo = s[i] * ball_inten
        # ff[i] = np.std(foo, axis=1) / np.mean(foo, axis=1)
        # ff[i] = -np.std(foo, axis=1)

    # ff = (s @ ball_inten.T) / np.linalg.norm(ball_inten, axis=1) / np.linalg.norm(s, axis=1)[:, np.newaxis]
    size_res = rs[np.argmax(ff, axis=1)]
    return size_res, ff, rs

# Semi-empirical high accuracy conversion from digital units to length.
REAL_UNIT = 2.23792136e-07*1e9


class Roundness:
    def __init__(self):
        center = (-16.6, 9.5)
        xx, yy = np.meshgrid(np.arange(128) - center[1],
                             np.arange(128) - center[0], indexing="ij")
        a = np.arctan2(xx, yy)
        self._nwedges = 36*2
        self._wedges = np.int_(((np.pi + a)/(2.*np.pi))*self._nwedges)
        self._wedges[(xx**2 + yy**2 < 50**2)] = -1
        self._wedges[(xx**2 + yy**2 > 100**2)] = -1
        # self._aavg_mean = np.load("/gpfs/exfel/exp/SPB/201802/p002160/usr/Shared/xfel2160/online/aavg_mean_72.npy")
        # self._aavg_mean_buffer = RingBuffer(1000)
        self._aavg_mean = None

    def get_wedges(self):
        return self._wedges

    def inv_roundness_stack(self, pattern_stack, mask=None):
        pattern_stack = pattern_stack[:128, :128, :].copy()
        pattern_stack[pattern_stack < 25] = 0.
        npatterns = pattern_stack.shape[2]
        aplot = np.zeros((npatterns, self._nwedges))
        acount = np.zeros((npatterns, self._nwedges))
        for iwedge in range(self._nwedges):
            aplot[:, iwedge] = pattern_stack[self._wedges == iwedge, :].sum(axis=0)
            acount[:, iwedge] = (self._wedges == iwedge).sum()

        aavg = aplot.copy()
        active_wedges = acount[0] > 0
        aavg[:, active_wedges] = aavg[:, active_wedges] / acount[:, active_wedges]
        #self._aavg_mean_buffer.append(aavg.mean())
        if self._aavg_mean is None:
            self._aavg_mean = aavg.mean(axis=0)
        else:
            self._aavg_mean = 0.95*self._aavg_mean + 0.05*aavg.mean(axis=0)
        aavg[:, active_wedges] /= (self._aavg_mean[active_wedges] + 1e-6)
        aavg[:, ~active_wedges] = 0.

        aavg_blur = scipy.ndimage.gaussian_filter1d(aavg, 1, axis=1)
        inv_roundness = aavg_blur.max(axis=1) / aavg_blur[:, active_wedges].mean(axis=1)
        return inv_roundness

nr_pulses = None

def pulses_mask(evt):
    global nr_pulses
    # Returns the current pulse range pattern by inspecting the GMD data from SQS
    # and making an educated guess from it
    npulses = (np.array(evt['SA3_XTD10_XGM/XGM/DOOCS:output']['SASE3 GMD[data.intensitySa1TD]'].data) != 1.0).sum()
    pulse_mask = np.zeros(176, dtype="bool")
    if npulses > 75:
        sl = slice(1, npulses+1, None)
    elif npulses > 50:
        sl = slice(1, npulses*2+1, 2)
    elif npulses > 38:
        sl = slice(1, npulses*3+1, 3)
    else:
        sl = slice(1, npulses*4+1, 4)
    pulse_mask[sl] = True
    
    if(nr_pulses is None or nr_pulses != pulse_mask.sum()):
        nr_pulses = pulse_mask.sum()
        print("Changed to %s pulses per train" % (nr_pulses))
    else:
        # Just to screw with people
        pulse_mask[-1] = True
    return pulse_mask
