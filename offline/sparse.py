"""
A library for the sparse (emc) HDF5 data format

Author: Benedikt Daurer
"""
import geom
import h5py
import sys
import numpy as np

class Run:
    """A wrapper for the sparse HDF5 file for a specific run."""
    def __init__(self, filename, mode="r"):
        """
        Parameters
        ----------
        filename : str
            Path to sparse HDF5 file
        """
        self.fname = filename
        self.mode = mode
    def __enter__(self):
        self._handle = h5py.File(self.fname, self.mode)
        self._numpix = self._handle['num_pix'][0]
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._handle.close()
    @property
    def nframes(self):
        """Nr. of events stored in the run file."""
        return self._handle['place_ones'].shape[0]
    @property
    def trainIds(self):
        """Train ID for all events."""
        return self._handle['id/trains'][:]
    @property
    def cellIds(self):
        """Cell ID for all events."""
        return self._handle['id/cells'][:]
    @property
    def pulseIds(self):
        """Pulse ID for all events."""
        return self._handle['id/pulses'][:]
    @property
    def litpixel(self):
        """
        Litpixel score for all events.
        returns None if score is not available.
        """
        try:
            return self._handle['scores/litpixel'][:]
        except KeyError:
            return None
    
class Frame(Run):
    """A wrapper for extracting frames from sparse HDF5 files."""
    def __init__(self, filename, geometry=None, goodmask=None):
        """
        Parameters
        ----------
        filename : str
            Path to sparse HDF5 file
        geometry : str
            Path to Cheetah-style geometry file
        goodmask : str 
            Path to HDF5 good-pixel mask file
            must have a dataset 'data/data' with shape (16,128,512)
        """
        Run.__init__(self, filename)
        self.geom = geometry
        if geometry is not None:
            self.x, self.y = geom.pixel_maps_from_geometry_file(self.geom)
        if goodmask is None:
            self.goodmask = np.ones((16,128,512), dtype=np.bool)
        else:
            self.goodmask = goodmask
        assert self.goodmask.shape == (16,128,512), "Mask should have shape (16,128,512)"

    def trainId(self, i):
        """Train ID for specific event."""
        return self._handle['id/trains'][i]
    def cellId(self, i):
        """Cell ID for specific event."""
        return self._handle['id/cells'][i]
    def pulseId(self, i):
        """Pulse ID for specific event."""
        return self._handle['id/pulses'][i]
    def modules(self, i):
        """
        Returns module array for given event index
        with shape (16,128,512)
        """
        frame = np.zeros(self._numpix, dtype=np.int64)
        frame[self._handle['place_ones'][i]] = 1
        frame[self._handle['place_multi'][i]] = self._handle['count_multi'][i]
        return np.transpose(frame.reshape((16,512,128)),axes=(0,2,1))
    def _modules_for_geom(self, i):
        frame = np.zeros(self._numpix, dtype=np.int64)
        frame[self._handle['place_ones'][i]] = 1
        frame[self._handle['place_multi'][i]] = self._handle['count_multi'][i]
        return frame.reshape((16,128,512))
    def assembled(self, i):
        """Returns assembled frame for given event."""
        if self.geom is None:
            print("Cannot provide assembled image, no geometry has been provided")
        return geom.apply_geom_ij_yx((self.y, self.x), self._modules_for_geom(i))[::-1,::-1]
    @property
    def activepixels(self):
        """assembled boolean image with active pixels = True."""
        if self.geom is None:
            print("Cannot provide active-pixel mask, no geometry has been provided")
        return geom.apply_geom_ij_yx((self.y, self.x), np.ones((16,128,512), dtype=np.bool))[::-1,::-1]
    @property
    def goodpixels(self):
        """assembled boolean image with good pixels = True."""
        if self.geom is None:
            print("Cannot provide good-pixel mask, no geometry has been provided")
        return geom.apply_geom_ij_yx((self.y, self.x), np.transpose(self.goodmask, axes=(0,2,1)))[::-1,::-1]
        
class Litpixel(Frame):
    """A wrapper for counting lit-pixels in a sparse HDF5 file."""
    def __init__(self, filename, goodmask=None):
        """
        Parameters
        ----------
        filename : str
            Path to sparse HDF5 file
        goodmask : str 
            Path to HDF5 good-pixel mask file
            must have a dataset 'data/data' with shape (16,128,512)
        """        
        Frame.__init__(self, filename, goodmask=goodmask)
        self.hitmask = np.zeros((16,128,512), dtype=np.bool)
        for m in [3,4,8,15]:
            self.hitmask[m,:,-128:] = True
        self.mask = (self.hitmask & self.goodmask).ravel()
        
    def count(self):
        """
        counting lit-pixels in inner 128x128 part of modules 3,4,8,15

        Returns
        -------
        1D array of length `nframes`.
        """
        scores = np.empty(self.nframes)
        for i in range(self.nframes):
            pones = self._handle['place_ones'][i]
            pmult = self._handle['place_multi'][i]
            cones = self.mask[pones].sum()
            cmult = self.mask[pmult].sum()
            scores[i] = cones + cmult
            sys.stderr.write('\rcounting lit pixels: %d/%d'%(i+1, self.nframes))
        sys.stderr.write('\n')
        return scores

class Powder(Frame):
    """A wrapper for producing powder sums from sparse HDF5 file."""
    def __init__(self, filename, geometry=None, goodmask=None, selection=None):
        """
        Parameters
        ----------
        filename : str
            Path to sparse HDF5 file
        geometry : str
            Path to Cheetah-style geometry file
        goodmask : str 
            Path to HDF5 good-pixel mask file
            must have a dataset 'data/data' with shape (16,128,512)
        selection : 1D boolean array
            must have length `nframes`
        """
        Frame.__init__(self, filename, geometry, goodmask)
        self.selection = selection

    def powder(self):
        """
        Returns module array or assembled image (if `geom` is available)
        after summing frames based on given selection. 
        """
        flist = np.arange(self.nframes)
        if self.selection is not None:
            flist = flist[self.selection]
        p = 0
        for f in flist:
            if self.geom is not None:
                p += self.assembled(f)
            else:
                p += self.modules(f)
        return p


class SmallFrame(Run):
    """A wrapper for extracting frames from sparse HDF5 files."""
    def __init__(self, filename, geometry=None, goodmask=None, mode="r"):
        """
        Parameters
        ----------
        filename : str
            Path to sparse HDF5 file
        geometry : str
            Path to Cheetah-style geometry file
        goodmask : str 
            Path to HDF5 good-pixel mask file
            must have a dataset 'data/data' with shape (4,128,128)
        """
        Run.__init__(self, filename, mode=mode)
        self.geom = geometry
        if geometry is not None:
            self.x, self.y = geom.pixel_maps_from_geometry_file(self.geom)
        if goodmask is None:
            self.goodmask = np.ones((4,128,128), dtype=np.bool)
        else:
            self.goodmask = goodmask
        assert self.goodmask.shape == (4,128,128), "Mask should have shape (4,128,128)"

    def trainId(self, i):
        """Train ID for specific event."""
        return self._handle['id/trains'][i]
    def cellId(self, i):
        """Cell ID for specific event."""
        return self._handle['id/cells'][i]
    def pulseId(self, i):
        """Pulse ID for specific event."""
        return self._handle['id/pulses'][i]
    def modules(self, i):
        """
        Returns module array for given event index
        with shape (4,128,128)
        """
        frame = np.zeros(self._numpix, dtype=np.int64)
        frame[self._handle['place_ones'][i]] = 1
        frame[self._handle['place_multi'][i]] = self._handle['count_multi'][i]
        return np.transpose(frame.reshape((4,128,128)),axes=(0,2,1))
    def _modules_for_geom(self, i):
        frame = np.zeros(self._numpix, dtype=np.int64)
        frame[self._handle['place_ones'][i]] = 1
        frame[self._handle['place_multi'][i]] = self._handle['count_multi'][i]
        return frame.reshape((4,128,128))
    def assembled(self, i):
        """Returns assembled frame for given event."""
        if self.geom is None:
            print("Cannot provide assembled image, no geometry has been provided")
        return geom.apply_geom_ij_yx((self.y, self.x), self._modules_for_geom(i))[::-1,::-1]
    @property
    def activepixels(self):
        """assembled boolean image with active pixels = True."""
        if self.geom is None:
            print("Cannot provide active-pixel mask, no geometry has been provided")
        return geom.apply_geom_ij_yx((self.y, self.x), np.ones((4,128,128), dtype=np.bool))[::-1,::-1]
    @property
    def goodpixels(self):
        """assembled boolean image with good pixels = True."""
        if self.geom is None:
            print("Cannot provide good-pixel mask, no geometry has been provided")
        return geom.apply_geom_ij_yx((self.y, self.x), np.transpose(self.goodmask, axes=(0,2,1)))[::-1,::-1]
    
