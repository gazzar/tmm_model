#!/usr/bin/env python

# Copyright (c) 2014, Gary Ruben
# Released under the Modified BSD license
# See LICENSE

"""General helper functions"""

import warnings
from skimage.io import imsave, imread
import skimage.transform as st
import numpy as np
import matplotlib.pyplot as plt

# append current directory to the path because we need to find tifffile.py
import sys, os
sys.path.insert(0, os.path.dirname(__file__))


def write_tiff32(filename, im):
    """Write a float32 tiff file.

    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # save a 32-bit tiff using either freeimage or tifffile
        im = im.astype(np.float32)
        try:
            imsave(filename, im, plugin='freeimage')
        except (ValueError, RuntimeError):
            imsave(filename, im, plugin='tifffile')


def read_tiff32(filename):
    """Read a float32 tiff file using either freeimage or tifffile.
    In fact, skimage.imread will read *some* 32-bit tiffs via PIL, such as those
    written by tifffile.imsave *but* PIL is fussier than tifffile and wouldn't
    read 32-bit tiffs written from ImageJ.
    
    Arguments:
    filename - full filename path string
    
    Returns:
    float32 image ndarray

    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # load a 32-bit tiff using either freeimage or tifffile
        try:
            im = imread(filename, plugin='freeimage', as_grey=True)
        except (ValueError, RuntimeError):
            im = imread(filename, plugin='tifffile', as_grey=True)
    im = im.astype(np.float32)
    return im


def zero_outside_circle(im):
    """Return a copy of a square image with zeroed values outside an imaginary
    inscribed circle that touches all four sides.

    """
    im = im.copy()
    rows, cols = im.shape
    assert rows==cols
    side = rows

    mask = np.hypot(*np.ogrid[-side/2:side/2, -side/2:side/2])
    im[mask > side/2] = 0.0
    return im


def imshow(im, show=False):
    """Display image in a window.

    """
    plt.matshow(im, cmap='gray')
    plt.colorbar()
    if show:
        plt.show()


def rotate(im, angle):
    """Use the skimage rotate function to rotate an image
    scale into (-1,1) range for rotate op then rescale.
    
    Arguments:
    im - float array
    angle - rotation angle in degrees

    """
    scale = im.max()
    return st.rotate(im/scale, angle) * scale
