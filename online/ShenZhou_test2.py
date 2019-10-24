import numpy as np
import matplotlib.pylab as plt
import cpplib
import time
import radial_avg
from pathlib import Path
import h5py
import functools
import xfel_online as xo

FOLDER = Path('/gpfs/exfel/u/scratch/SPB/201802/p002160/ekeberg/simulated')
with h5py.File(FOLDER / 'simulated_octa_var.h5') as fp:
    hits = fp['pattern'][:]
    sample_size = fp['sample_size'][:]

mask = np.ones_like(hits, dtype='bool')

for _ in range(4):
    t0 = time.time()
    size_res, ff, rs = xo.sizingAGIPD(hits, mask)
    t1 = time.time()
    print(t1-t0)
# plt.plot(size_res, sample_size, 'o')
# plt.title(np.polyfit(size_res, sample_size, 1))
# plt.show()

# plt.plot(ff[0])
# plt.plot(s[0])
# plt.semilogy(ball_fft[3])
# plt.plot(ff)
# plt.imshow(s / c)
# plt.show()

# p = np.linspace(0, 200, 100)
# plt.semilogy(p, radial_avg.ball_radial_intensity(10, 0.0005, p))
# plt.show()

# data = radial_avg.gen_data(cx=40, cy=-100, num_pulses=120)
# plt.imshow(data[:,:,3])
# plt.show()
# mask = np.ones_like(data, dtype='bool')
# t0 = time.time()
# b, m, r = cpplib.radialAverage(40, -100, mask, data, 1)
# dt = time.time() - t0
# plt.imshow(b / m)
# plt.title(dt)
# plt.show()
