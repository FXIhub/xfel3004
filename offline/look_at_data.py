import argparse
import h5py
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np
import os
import pickle
from tqdm import tqdm
import signal

import skimage.measure
import skimage.segmentation

ADU_PER_PHOTON = 3

GEOM      = '/gpfs/exfel/exp/SQS/202302/p003004/usr/Shared/amorgan/xfel3004/tools/maskmaker/dssc_xyz_map.pickle'
PREFIX    = '/gpfs/exfel/exp/SQS/202302/p003004/scratch/'
DATA_PATH = 'entry_1/instrument_1/detector_1/data'
DARK_PATH = 'data/mean'
MASK_PATH = 'entry_1/good_pixels'

depth = 1000

def parse_cmdline_args():
    parser = argparse.ArgumentParser(description='view shots from vds files')
    parser.add_argument('vds', type=int, help="run number of the vds file.")
    parser.add_argument('-d', '--dark', type=int, default = 29, help="dark run number.")
    parser.add_argument('-m', '--mask', type=int, default = 195, help="pixel mask run number.")
    parser.add_argument('-l', '--litpixels', action='store_true', help="use litpixels to sort events.")
    return parser.parse_args()

class Application:
    def __init__(self, data, cellID, dark_dict, sorted_indices, indices_image_space, background_mask, xy_map, im_shape, d, litpix):
        self.Z = data.shape[0]
        self.frame_index = -1
         
        self.xmin, self.ymin, self.dx = d
        
        self.im_shape = im_shape
        self.dark_dict = dark_dict
        self.sorted_indices = sorted_indices

        self.cellID = cellID
        self.z_data = data

        self.litpix = litpix
         
        self.data = np.empty(np.squeeze(data[0]).shape, dtype=np.float32)
        
        self.pixel_map       = indices_image_space
        self.background_mask = background_mask

        self.display = np.zeros(background_mask.shape, dtype=np.float32)
          
        self.in_replot = False
        
        self.initUI()
        
        self.replot_frame(True)

    def initUI(self):
        signal.signal(signal.SIGINT, signal.SIG_DFL) # allow Control-C
        
        # Always start by initializing Qt (only once per application)
        self.app = QtWidgets.QApplication([])
        
        # Define a top-level widget to hold everything
        w = QtWidgets.QWidget()
        
        # 2D plot for the cspad and mask
        self.plot = pg.ImageView()

        # add a + at the origin
        # x=0, i = -xmin / dx + 0.5
        i0 = -self.xmin / self.dx + 0.5
        j0 = -self.ymin / self.dx + 0.5
        scatter = pg.ScatterPlotItem([{'pos': (i0, j0), 'size': 5, 'pen': pg.mkPen('r'), 'brush': pg.mkBrush('r'), 'symbol': '+'}])
        self.plot.addItem(scatter)
        
        if self.Z > 1 :
            # add a z-slider for image selection
            z_sliderW = pg.PlotWidget()
            z_sliderW.plot(self.litpix, pen=(255, 150, 150))
            z_sliderW.setFixedHeight(100)
            
            # vline
            self.vline = z_sliderW.addLine(x = 0, movable=True, bounds = [0, self.Z-1])
             
            self.vline.sigPositionChanged.connect(self.replot_frame)
        
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.plot)
        vbox.addWidget(z_sliderW)
    
        ## Display the widget as a new window
        w.setLayout(vbox)
        w.resize(800, 480)
        w.show()
        
        ## Start the Qt event loop
        self.app.exec_()

    def replot_frame(self, auto=False):
        if self.in_replot:
            return
        try:
            self.in_replot = True
            i = int(self.vline.value())
            if self.frame_index != i :
                self.frame_index = i
                j = self.sorted_indices[self.frame_index]
                print(i, self.frame_index, j, self.data.dtype)
                self.data[:] = np.squeeze(self.z_data[j]) - self.dark_dict[self.cellID[j]]
                self.updateDisplayRGB(auto)
        finally:
            self.in_replot = False

    def updateDisplayRGB(self, auto = False):
        """
        Make an RGB image (N, M, 3) (pyqt will interprate this as RGB automatically)
        with masked pixels shown in blue at the maximum value of the cspad. 
        This ensures that the masked pixels are shown at full brightness.
        """
        self.display[~self.background_mask] = self.data.ravel()[self.pixel_map]
        if not auto :
            self.plot.setImage(self.display.reshape(self.im_shape))
        else :
            self.plot.setImage(self.display.reshape(self.im_shape), autoRange = False, autoLevels = False, autoHistogramRange = False)
    

def generate_pixel_lookup(xyz):
    # choose xy bounds
    xmin = xyz[0].min()
    xmax = xyz[0].max()
    
    ymin = xyz[1].min()
    ymax = xyz[1].max()
    
    # choose sampling
    dx = 177e-6 / 2
    
    shape = (int( (xmax-xmin)/dx ) + 2, int( (ymax-ymin)/dx ) + 2)

    # pixel coordinates in im
    ss = np.round((xyz[0] - xmin) / dx).astype(int)
    fs = np.round((xyz[1] - ymin) / dx).astype(int)
    
    i, j = np.indices(shape)
    xy_map = np.empty((2,) + shape, dtype=float)
    xy_map[0] = dx * i
    xy_map[1] = dx * j
    
    # now use pixel indices as labels
    i = np.arange(xyz[0].size)
     
    # create an image of the data raveled indices
    im = -np.ones(shape, dtype=int)
    im[ss.ravel(), fs.ravel()] = i
    
    # label image
    # problem, the labells dont equal i
    #l = skimage.measure.label(im, background=-1)
    
    # expand by oversampling rate (to fill gaps)
    l = skimage.segmentation.expand_labels(im+1, distance = 2)
        
    # set background mask
    background_mask = (l.ravel()==0).copy()
    
    # now subtract 1 from labels to turn them into pixel indices
    l -= 1
    
    indices_image_space = l.ravel()[~background_mask].copy()
    
    # now to map data to 2D image we have:
    # im[~background] = data.ravel()[indices_image_space]
    return indices_image_space, background_mask, shape, (xmin, ymin, dx), im, i, l

args = parse_cmdline_args()
args.run  = args.vds
args.vds  = PREFIX+'vds/r%.4d.cxi'%args.vds
args.dark = PREFIX+'dark/r%.4d_dark.h5'%args.dark
args.mask = PREFIX+'det/badpixel_mask_r%.4d.h5'%args.mask

if args.litpixels :
    args.litpixels = PREFIX+'events/r%.4d_events.h5'%args.run
    with h5py.File(args.litpixels) as f:
        litpixels = np.sum(f['/entry_1/litpixels'][()], axis=0)
        trainID   = f['/entry_1/trainId'][()]
        pulseID   = f['/entry_1/pulseId'][()]
        fiducial  = trainID * pulseID
        events    = np.argsort(litpixels)[::-1]
        litpix    = litpixels[events]
        fiducial_lit  = fiducial[events]
        sort = True
else :
    sort = False

#with h5py.File(args.vds) as f:
f = h5py.File(args.vds)
data = f[DATA_PATH]
cellID    = f['entry_1/cellId'][:, 0]
trainID   = f['/entry_1/trainId'][()]
pulseID   = f['/entry_1/pulseId'][()]
fiducial  = trainID * pulseID
"""
frame_shape = np.squeeze(f[DATA_PATH][0]).shape
cellID = f['entry_1/cellId'][:, 0]
trainID   = f['/entry_1/trainId'][()]
pulseID   = f['/entry_1/pulseId'][()]
fiducial  = trainID * pulseID
if sort:
    data = np.empty( (depth,) + frame_shape, dtype=np.float32)
    for i in tqdm(range(depth)):
        index = np.where(fiducial == fiducial_lit[i])[0]
        data[i] = np.squeeze(f[DATA_PATH][index].astype(np.float32))
else :
    data = f[DATA_PATH][fiducial].astype(np.float32)
"""


if sort:
    sorted_indices = np.empty((data.shape[0],), dtype=int)
    j = np.argsort(fiducial)
    k = np.arange(fiducial.shape[0])[j]
    fiducial_sorted = fiducial[j]
    for i in tqdm(range(data.shape[0])):
        #sorted_indices[i] = np.where(fiducial == fiducial_lit[i])[0]
        sorted_indices[i] = k[ np.searchsorted(fiducial_sorted, fiducial_lit[i]) ]
        #sorted_indices = sorted_indices[::-1]

else :
    sorted_indices = np.arange(data.shape[0])

with h5py.File(args.dark) as f:
    dark   = f[DARK_PATH][()]
    cellID_dark = f['data/cellId'][:]

# make a dictionary for easy look up
dark_dict = {}
for i in range(dark.shape[1]):
    dark_dict[cellID_dark[i]] = dark[:,i]

"""
with h5py.File(args.mask) as f:
    mask   = f[MASK_PATH][()]
"""

# make cellID a dictionary
xyz = pickle.load(open(GEOM, 'rb'))

# generate pixel lookup
indices_image_space, background_mask, im_shape, (xmin, ymin, dx), im, i, l = generate_pixel_lookup(xyz)

"""
# make display images
disp = np.zeros((data.shape[0],) + im_shape, dtype=data.dtype)

# dark correction
for d in range(data.shape[0]):
    data[d] -= dark_dict[cellID[d]]

# save 
with h5py.File('temp.h5', 'w') as f:
    f['data'] = data

frame = np.zeros(np.prod(im_shape), dtype=np.float32)
for d in range(data.shape[0]):
    frame[~background_mask] = data[d].ravel()[indices_image_space]
    disp[d] = frame.reshape(im_shape)
"""


#pg.show(disp)
Application(data, cellID, dark_dict, sorted_indices, indices_image_space, background_mask, xyz, im_shape, (xmin, ymin, dx), litpix)


