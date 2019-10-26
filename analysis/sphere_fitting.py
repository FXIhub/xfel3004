#!/usr/bin/env python
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import sys; sys.path.append("../offline/")
import sparse, geom, utils
import spimage
import scipy.optimize as optimize
import scipy.stats as stats
import argparse

parser = argparse.ArgumentParser(prog="Sphere fitting")
parser.add_argument("run", type=int, help="Run number")
args = parser.parse_args()
print("Starting run %d" %args.run)

# Experiment parameters
downsampling = 1
pixelsize = downsampling * 200e-6
distance = 1.62 # From logbook
material = 'sucrose'
phenergy = 6000. #eV
wavelength = (1239.8418746 / phenergy) * 1e-9 # m
saturation_level = 10000
gain = 1 # ADUs per photn

# Sizing parameters
diameter_start  = 200 * 1e-9
intensity_start = 1. * 1e-3 / 1e-12 #[mJ/um2]
brute_evals = 200
rmax = 250 / downsampling
maskradius = 100 / downsampling

# Radial mask
def rmask(r, sh, cx, cy):
    ny, nx = sh
    xx,yy = np.meshgrid(np.arange(nx), np.arange(ny))
    return (xx-cx)**2 + (yy-cy)**2 > (r**2)


infile = "../data/r%04d_lowq.h5" %args.run
with sparse.SmallFrame(infile, geometry="../geometry/b3_lowq.geom", mode="r+") as f:
    mask = f.activepixels[:300,:300]
    sh = mask.shape
    qr = spimage.x_to_qx(np.arange(0,sh[0]/2.), pixelsize/downsampling, distance)

    def model(p):
        diameter, intensity = p
        A = spimage.sphere_model_convert_intensity_to_scaling(intensity, diameter, wavelength, pixelsize, 
                                                              distance, material=material)
        s = spimage.sphere_model_convert_diameter_to_size(diameter, wavelength, pixelsize, distance)
        I = spimage.I_sphere_diffraction(A,qr,s)
        return I

    def costfunc1(p, data, mask):
        return 1-stats.pearsonr(model([p,1e8])[mask],data[mask])[0]

    def costfunc2(p, data, mask, diameter):
        return ((model([diameter,p])[mask] - data[mask])**2).sum()

    # Centering parameters
    centering_maxshift  = 40 / downsampling
    centering_threshold = 0.5
    centering_blur      = 4
    centering_x0 = sh[1]/2.
    centering_y0 = sh[0]/2.

    # Add new entries to the file
    if "diameter" not in f._handle:
        f._handle.create_dataset("diameter", (f.nframes,))
    if "intensity" not in f._handle:
        f._handle.create_dataset("intensity", (f.nframes,))
    if "cx" not in f._handle:
        f._handle.create_dataset("cx", (f.nframes,))
    if "cy" not in f._handle:
        f._handle.create_dataset("cy", (f.nframes,))
    if "error_test" not in f._handle:
        f._handle.create_dataset("error_test", (f.nframes,))
    if "radial_qr" not in f._handle:
        f._handle.create_dataset("radial_qr", (f.nframes, sh[0]//2))
    if "radial_data" not in f._handle:
        f._handle.create_dataset("radial_data", (f.nframes, sh[0]//2))
    if "radial_fit" not in f._handle:
        f._handle.create_dataset("radial_fit", (f.nframes, sh[0]//2))

    # Loop through all data
    for i in range(f.nframes):
        assembled = f.assembled(i)[:300,:300]

        # Step. 1 Finding the center
        x,y = spimage.find_center(assembled,mask,method='blurred', 
                                  x0=centering_x0, y0=centering_y0,dmax=centering_maxshift, 
                                  threshold=centering_threshold, blur_radius=centering_blur)
        #print("Step 1: Found center position (%.2f, %2.f)" %(x,y))

        # Add spherical mask in the center (restricting the sizing to low q)
        mask_sizing = mask & ~rmask(maskradius, mask.shape, mask.shape[1]/2+x, mask.shape[0]/2+y)

        #plt.figure(figsize=(5,5), dpi=100)
        #plt.axis('off')
        #plt.imshow(assembled_sorted[j]*cmask*mask_sizing, norm=colors.LogNorm(), cmap='magma')
        #plt.show()


        # Step 2. Radial average
        centers, radial = spimage.radialMeanImage(assembled, cx=mask.shape[1]/2+x, 
                                          cy=mask.shape[0]/2+y, output_r=True)
        data_r  = radial[:sh[0]//2]
        data_qr = spimage.x_to_qx(centers, pixelsize, distance)[:sh[0]//2]
        mask_qr = data_qr > 50


        # ## Step 3. Fitting size/intensity
        output = optimize.brute(costfunc1, [(20e-9,320e-9)], args=(data_r, mask_qr), Ns=300, full_output=True)
        diameter = output[0]
        res = optimize.minimize(costfunc2, [intensity_start], args=(data_r, mask_qr, diameter), 
                                method="Powell", tol=None, options={'disp':False})
        intensity = res['x']
        fun = res['fun']
        res2 = optimize.minimize(costfunc2, [intensity_start], args=(data_r, mask_qr, diameter/2), 
                                 method="Powell", tol=None, options={'disp':False})
        good = (res2['fun'] - fun)

        #plt.figure()
        #plt.title("good = %f" %good)
        #plt.plot(qr, data_r*mask_qr, label='%d' %i)
        #plt.plot(qr, mask_qr*model([diameter,intensity]))
        #plt.show()

        # Save data to file
        f._handle["diameter"][i] = diameter
        f._handle["intensity"][i] = intensity
        f._handle["cx"][i] = x
        f._handle["cy"][i] = y
        f._handle["error_test"][i] = good
        f._handle["radial_qr"][i] = qr
        f._handle["radial_data"][i] = model([diameter, intensity])
        print("Done with %d/%d" %(i+1,f.nframes))

