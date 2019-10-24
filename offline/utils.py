"""
Utility functions

Author: Benedikt Daurer
"""
import geom
import numpy as np

def assemble(modules, geomfile):
    x, y = geom.pixel_maps_from_geometry_file(geomfile)
    image = geom.apply_geom_ij_yx((y, x), np.transpose(modules, axes=(0,2,1)))[::-1,::-1]
    return image
