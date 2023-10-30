# Graphical Mask Maker for DSSC detector 
Based on CsPadMaskMaker.

This provides a geometry corrected view of the hexagonal pixels of the DSSC detector, each pixel is placed in x,y space then the surrounding area is 'filled' to provide an approximate representation of the hexagonal grid with a small number of pixels. Masked pixels are in data-space, so when masking a pixel on screen this will lookup the corresponding data pixel and then map that change to all on-screen pixels representing a single hexagonal pixel.

The boolean mask is saved to:
    mask.h5
        entry_1/good_pixels

## Installation
To install dependencies through conda, run:
```
$ module load exfel exfel-python
$ conda env create -f environment.yml -n maskmaker # takes ages!
$ conda activate maskmaker 
```

## Example
```
(maskmaker)$ python maskmaker.py /home/amorgan/p003004/scratch/amorgan/scratch/dark/r0290_dark.h5/data/sigma -m badpixel_mask_r0195.h5/entry_1/good_pixels 
```
