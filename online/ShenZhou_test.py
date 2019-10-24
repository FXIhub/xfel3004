import numpy as np
import matplotlib.pylab as plt
import cpplib
import time
import radial_avg
from pathlib import Path
import h5py

#2.23792136e-07

# FOLDER = Path('/gpfs/exfel/u/scratch/SPB/201802/p002160/ekeberg/simulated')
# with h5py.File(FOLDER / 'simulated_octa_var.h5') as fp:
#     hits = fp['pattern'][:]
#     sample_size = fp['sample_size'][:]
FOLDER = Path('/gpfs/exfel/u/scratch/SPB/201802/p002160/ekeberg/')
with h5py.File(FOLDER / 'templates.h5') as fp:
    hits = fp['templates'][:]
    sample_size = fp['diameters'][:]


print(sample_size)
mask = np.ones_like(hits, dtype='bool')
s, c, r = cpplib.radialAverage(-8, 20, mask, hits, 1)
s /= c
rs = np.linspace(0.01, 1.7, 200)
ball_fft = np.zeros(( len(rs), len(r)))
for i, rad in enumerate(rs):
    ball_fft[i] = radial_avg.ball_radial_intensity(10, rad, r)
ff = (s @ ball_fft.T) / np.linalg.norm(ball_fft, axis=1)
# idx = np.where((rs[np.argmax(ff, axis=1)]<0.2)&(sample_size>1e-7))[0][0]
# plt.plot(ff[idx])

size_res = rs[np.argmax(ff, axis=1)]
plt.plot(rs[np.argmax(ff, axis=1)], sample_size, 'o')
plt.title(np.polyfit(size_res, sample_size, 1))
plt.show()

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
