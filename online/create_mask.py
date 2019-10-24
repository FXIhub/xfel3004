import numpy
import scipy.ndimage

data = numpy.load("raw_detector.npy")
data.mean(axis=2)


mask = scipy.ndimage.morphology.binary_dilation(data.mean(axis=2) < 4000, iterations=5)
mask = scipy.ndimage.morphology.binary_erosion(mask, iterations=3)

mask = ~mask

mask.dump("initial_mask.npy")
