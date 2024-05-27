# read in photons from each h5 file
# sort by photon count
# display

# global index is seqential list
# index events by global_index -> fnam, file_index pairs
import h5py
import numpy as np
import pyqtgraph as pg
import extra_geom
from tqdm import tqdm
from PyQt5 import QtGui, QtCore, QtWidgets
from pyqtgraph.graphicsItems.InfiniteLine import InfiniteLine
import signal
import os
import sys


PREFIX = sys.argv[1]
#PREFIX = '/home/andyofmelbourne/Documents/2023/P3004-take-2/gold/'
fnams = [PREFIX + 'hits_r%.4d.cxi'%run for run in range(87, 96)]

dset   = '/entry_1/instrument_1/detector_1/photons'
geom_fnam = f'{PREFIX}/crystfel_geom_0087.geom'

geom = extra_geom.DSSC_1MGeometry.from_crystfel_geom(geom_fnam)
with h5py.File(fnams[0]) as f:
    frame, centre = geom.position_modules(f['entry_1/data_1/data'][0])
frame_shape = frame.shape
frame_dtype = frame.dtype

Nevents = 0
for fnam in fnams:
    with h5py.File(fnam) as f:
        data = f['entry_1/data_1/data']
        Nevents += data.shape[0]

global_index  = np.arange(Nevents)
photon_counts = np.empty((Nevents,), dtype = int)
file_index    = np.empty((Nevents,), dtype = int)
frame_index   = np.empty((Nevents,), dtype = int)
selection     = np.zeros((Nevents,), dtype = bool)

out = '/manual_selection/is_sample_hit'

index = 0
for i, fnam in enumerate(fnams):
    with h5py.File(fnam) as f:
        data = f['entry_1/data_1/data']
        photon_counts[index: index + data.shape[0]] = f[dset][()]
        file_index[index: index + data.shape[0]]  = i
        frame_index[index: index + data.shape[0]] = np.arange(data.shape[0])

        # load previous selection 
        if out in f :
            selection[index: index + data.shape[0]] = f[out][()]
        
        index += data.shape[0]

# sort file and frame indices by photon counts (most to least)
sorted_index = np.argsort(photon_counts)[::-1]
file_index   = file_index[sorted_index]
frame_index  = frame_index[sorted_index]
selection    = selection[sorted_index]

# get data
#N = 100
#sorted_index = np.argsort(photon_counts)[::-1]
#frames = np.zeros((N,) + frame.shape, dtype=np.uint16)
#for i, index in tqdm(enumerate(sorted_index[:N]), total = N) :
#    with h5py.File(fnams[file_index[index]]) as f:
#        t         = f['entry_1/data_1/data'][frame_index[index]]
#        geom.position_modules(t, out = frames[i])

# load the n'th most intense frame
def load_frame(i, frame):
    j = frame_index[i]
    fnam = fnams[file_index[i]]
    #print(f'loading frame {j} from file {fnam}')
    with h5py.File(fnam) as f:
        t         = f['entry_1/data_1/data'][j]
        geom.position_modules(t, out = frame)

def clip_scalar(val, vmin, vmax):
    """ convenience function to avoid using np.clip for scalar values """
    return vmin if val < vmin else vmax if val > vmax else val

class Application(QtWidgets.QMainWindow):
    def __init__(self, selection):
        super().__init__()
        
        print("type 'r' to clear all frame selections (including those loaded from file)")
        print("type 'x' to toggle good/bad state of frame")
        print("type 's' to save good frames list to h5 file under /manual_selection/is_sample_hit")
        
        self.im = np.zeros(frame_shape, dtype = frame_dtype)
        self.selection = selection
        
        self.initUI()
    
    def initUI(self):
        # 2D plot for the cspad and mask
        self.imageitem = pg.ImageItem()
        
        w = pg.GraphicsLayoutWidget()
        
        vb = w.addViewBox(lockAspect=True)
        vb.addItem(self.imageitem)
        
        cbar = pg.HistogramLUTItem(image=self.imageitem)
        cbar.gradient.loadPreset('thermal')
        w.addItem(cbar)
        
        self.bounds = [0, Nevents-1]
        
        self.timeline = InfiniteLine(0, movable=True)
        self.timeline.setPen((255, 255, 0, 200))
        self.timeline.setBounds(self.bounds)
        self.timeline.sigPositionChanged.connect(self.timeLineChanged)
        
        p2 = w.addPlot(row=1, col=0)
        p2.addItem(self.timeline)
        p2.hideAxis('left')
        p2.setXRange(self.bounds[0], self.bounds[1])
        w.ci.layout.setRowFixedHeight(1, 35)
        
        self.setCentralWidget(w)
        
        # display the image
        self.currentIndex = 0
        self.updateImage(init = True)
        
        ## Display the widget as a new window
        self.resize(800, 480)
        #self.show()
    
    def updateImage(self, init = False):
        if init :
            autoLevels = True
        else :
            autoLevels = False
        load_frame(self.currentIndex, self.im)
        self.imageitem.setImage(self.im, autoRange = True, autoLevels = autoLevels, autoHistogramRange = True)
        self.update_border()
        
    def timeLineChanged(self):
        ind = int(self.timeline.value())
        if ind != self.currentIndex:
            self.currentIndex = ind
            self.updateImage()
        
        #self.sigTimeChanged.emit(ind)

    def keyPressEvent(self, event):
        super(Application, self).keyPressEvent(event)
        key = event.key()
        
        if key == QtCore.Qt.Key_Left :
            ind = clip_scalar(self.currentIndex - 1, self.bounds[0], self.bounds[1]-1)
            self.timeline.setValue(ind)
        
        elif key == QtCore.Qt.Key_Right :
            ind = clip_scalar(self.currentIndex + 1, self.bounds[0], self.bounds[1]-1)
            self.timeline.setValue(ind)

        elif key == QtCore.Qt.Key_X :
            self.selection[self.currentIndex] = ~self.selection[self.currentIndex]
            print('total number of selected frames:', np.sum(self.selection))
            self.update_border()

        elif key == QtCore.Qt.Key_R :
            self.selection.fill(False)
            print('total number of selected frames:', np.sum(self.selection))
            self.update_border()

        elif key == QtCore.Qt.Key_S :
            self.save_selection()
            
    def update_border(self):
        if self.selection[self.currentIndex]:
            self.imageitem.setBorder('g')
        else :
            self.imageitem.setBorder('r')

    def save_selection(self):
        print('saving hit selection to h5 files...')
        selections = {}
        for fnam in fnams :
            selections[fnam] = []
            
        # save a list of selected indices for each file
        for i in np.where(self.selection)[0] :
            selections[fnams[file_index[i]]].append(frame_index[i])
        
        out = '/manual_selection/is_sample_hit'
        dset   = '/entry_1/instrument_1/detector_1/photons'
        for fnam in selections :
            with h5py.File(fnam, 'a') as f:
                s = np.zeros((f[dset].shape[0],), dtype = bool)
                if len(selections[fnam]) > 0 :
                    s[selections[fnam]] = True
                if out in f :
                    f[out][:] = s
                else :
                    f[out] = s
                print(f'finished writing {len(selections[fnam])} selected labels to {fnam}')
        print('done')
            
        



signal.signal(signal.SIGINT, signal.SIG_DFL) # allow Control-C

# Always start by initializing Qt (only once per application)
app = QtWidgets.QApplication([])
    
a = Application(selection)
a.show()

## Start the Qt event loop
app.exec_()
