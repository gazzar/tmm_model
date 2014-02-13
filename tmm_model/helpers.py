#!/usr/bin/env python

# Copyright (c) 2014, Gary Ruben
# Released under the Modified BSD license
# See LICENSE

"""General helper functions"""

import warnings
from skimage.io import imsave
import numpy as np

# append current directory to the path because we need to find tifffile.py
import sys, os
sys.path.insert(0, os.path.dirname(__file__))


def write_tiff32(filename, im):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # save a 32-bit tiff using either freeimage or tifffile
        im = im.astype(np.float32)
        try:
            imsave(filename, im, plugin='freeimage')
        except (ValueError, RuntimeError):
            imsave(filename, im, plugin='tifffile')

def zero_outside_circle(im):
    im = im.copy()
    rows, cols = im.shape
    assert rows==cols
    side = rows

    mask = np.hypot(*np.ogrid[-side/2:side/2, -side/2:side/2])
    im[mask > side/2] = 0.0
    return im
