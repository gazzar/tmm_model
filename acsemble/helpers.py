#!/usr/bin/python3

from __future__ import absolute_import, division, print_function
from . import config
import logging
import skimage.transform as st
import os
import fnmatch
import re
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from . import tifffile
    from skimage.io import imsave, imread

try:
    from skimage.measure import compare_ssim
    mssim_version = "new"
except ImportError:
    from skimage.measure import structural_similarity as compare_ssim
    mssim_version = "old"

logger = logging.getLogger(__name__)


def write_tiff32(filename, im):
    """Write a float32 tiff file.

    """
    # save a 32-bit tiff using either freeimage or tifffile
    im = im.astype(np.float32)
    if np.any(np.isnan(im)):
        logging.warning('warning: file {} contains NaNs'.format(filename))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imsave(filename, im, plugin='freeimage')
    except (ValueError, RuntimeError):
        tifffile.imsave(filename, im, compress=1)
    logging.info('save tiff: ' + filename)


def read_tiff32(filename):
    """Read a float32 tiff file using either freeimage or tifffile.
    In fact, skimage.imread will read *some* 32-bit tiffs via PIL, such as those
    written by tifffile.imsave *but* PIL is fussier than tifffile and wouldn't
    read 32-bit tiffs written from ImageJ.
    
    Parameters
    ----------
    filename - full filename path string
    
    Returns
    -------
    float32 image ndarray

    """
    # load a 32-bit tiff using either freeimage or tifffile
    try:
        im = imread(filename, plugin='freeimage', as_grey=True)
    except (ValueError, RuntimeError):
        im = tifffile.imread(filename)
    im = im.astype(np.float32)
    logging.info('load tiff: ' + filename)
    return im


def zero_outside_circle(im, symmetric=True, tolerance=0.0):
    """Return a copy of a square image with zeroed values outside an imaginary
    inscribed circle that touches all four sides.

    Parameters
    ----------
    im - 2d ndarray
        image to mask
    symmetric - Boolean
        If True, use the symmetric mask, else use the quick mask
    tolerance - Float
        Positive number of pixels by which mask circle is expanded

    Returns
    -------
    image with region outside inscribed circle zeroed

    """
    im = im.copy()
    rows, cols = im.shape
    assert rows == cols
    side = rows

    if symmetric:
        s = (side - 1) / 2.
        mask = np.hypot(*np.ogrid[-s:s + 1, -s:s + 1]) > s + tolerance + 1
        im[mask] = 0.0
    else:
        mask = np.hypot(*np.ogrid[-side/2:side/2, -side/2:side/2]) > side/2 + tolerance
        im[mask] = 0.0
    return im


def zero_outside_mask(im, mask):
    """Zero the entries in im (i.e. not in a copy of im) that lie in the outer
    part of the mask image (based on the value of mask[0,0])
    
    Parameters
    ----------
    im - 2d ndarray
    mask - 2d ndarray

    """
    s = nd.generate_binary_structure(2, 2)      # 8-connected structure element
    mask = (mask == mask[0, 0])
    la, _ = nd.label(mask, structure=s)
    im[la == la[0, 0]] = 0.0


def imshow(im, show=False, cmap='gray', **kwargs):
    """Display image in a window.

    """
    plt.imshow(im, origin='upper', interpolation='nearest', cmap=cmap,
               **kwargs)
    plt.colorbar()
    if show:
        plt.show()


def rotate(im, angle):
    """Use the skimage rotate function to rotate an image
    scale into (-1,1) range for rotate op then rescale.
    
    Parameters
    ----------
    im - float array
    angle - rotation angle in degrees
        Rotates ccw for an image with (0,0) (row, column) coords in the upper-left or
        equivalently, rotates cw for an image with (0,0) (row, column) coords in the
        lower-left. Note, this is the opposite sense to the mathematical convention for
        rows corresponding to y and cols to x.
                            y
                            ^
                            |
        with z(out-of-page) o--> x
        However, our 2d images lie in the xz-plane
        with y(out-of-page) o--> x
                            |
                            v
                            z
        so actually the rotation sense matches convention for this case. In other words, for
        xz-plane images, don't change the rotation sign. For xy-plane images call with -angle.

    Returns
    -------
    2d array of float (Rotated image)

    """
    assert issubclass(im.dtype.type, np.floating)

    implementation = config.iradon_implementation

    if implementation == 'skimage_iradon':
        return st.rotate(im, angle, center=np.array(im.shape)//2, preserve_range=True)
    elif implementation in ('xlict_recon_mpi_fbp', 'xlict_recon_gridrec'):
        # Check whether rotation angle is close enough to be able to rotate using a rot90
        # call. If so, use rot90 instead. I doubt we'll ever do 10000 rotation angles, so
        # use this to determine the threshold.
        axes = np.array([-360.0, -270.0, -180.0, -90.0, 0.0, 90.0, 180.0, 270.0, 360.0])
        aligned_with_axis = abs(angle - axes) < 360./10000  # faster than np.isclose()
        if np.any(aligned_with_axis):
            return np.rot90(im, k=np.flatnonzero(aligned_with_axis)[0])
        else:
            return st.rotate(im, angle, preserve_range=True)


def match_pattern(pattern, s, normalise_paths=True):
    """Does a glob pattern match against a list of strings. Only the *
    character is currently supported here (no ? or []). Returns a list of
    tuples of (matching_string, matching_glob_component). For example
    >>> match_pattern('a*z', ['abc', 'abcz', 'az'])
    returns
    [('abcz', 'bc'), ('az', '')]

    >>> match_pattern('a*z', ['abc'])
    returns
    []

    Parameters
    ----------
    pattern - string
        A string that defines a match pattern and may use the unix glob *
        character to define a wildcard section
    s - string
        The comparison string to match against the pattern
    normalise_paths - bool
        If True, normalises all path eparators to the native separator for the current os

    Returns
    -------
    list of tuples of (matching_string, matching_glob_component)

    """
    if normalise_paths:
        pattern = os.path.normpath(pattern)
        s = [os.path.normpath(p) for p in s]

    # Create a regex that will capture the glob pattern part
    regex = fnmatch.translate(pattern).replace('.*', '(.*)')
    reobj = re.compile(regex)

    if '*' in pattern:
        matches = []
        for f in s:
            match = reobj.search(f)
            if match is not None:
                matches.append((f, match.group(1)))
    elif pattern in s:
        matches = [(pattern, pattern)]
    else:
        matches = []

    return matches


def mse(im1, im2, mask=None):
    """Returns MSE of masked regions of im1 and im2.

    Parameters
    ----------
    im1, im2 : ndarray of float
        images to compare using MSE metric
    mask : ndarray of boolean
        boolean mask array same shape as im1 and im2

    Returns
    -------
    float

    """
    # assert im1.shape == im2.shape
    # if mask is None:
    #     mask = np.ones(im1.shape, dtype=np.bool)
    # a_diff = im1[mask] - im2[mask]
    # return np.dot(a_diff, a_diff)/a_diff.size

    assert im1.shape == im2.shape
    if mask is None:
        a_diff = im1 - im2
    else:
        assert mask.shape == im1.shape
        a_diff = im1[mask] - im2[mask]
    a_diff = a_diff.ravel()
    return np.dot(a_diff, a_diff)/a_diff.size


def chi2(im1, im2, mask=None, mask_zeros=True, epsilon=1e-18):
    """Returns chi2 of masked regions of im1 as it compares to a reference image im2.
    This is the one-way chi2 metric, which typically assumes data are binned. Doing so to an
    image would avoid the metric blowing up when it encounters zeros in the im2 "expected"
    or reference image.

    Args:
        im1: 2d ndarray of float
        im2: 2d ndarray of float
        mask: 2d ndarray of bool whose shape matches im1 and im2
            If None, the mask is not applied.
        mask_zeros: bool
            If True, applies a mask which is False wherever im2==0, True elsewhere, and
            logically ANDs this with mask.
        epsilon: float (default 1e-18)
            A small, positive value below which values are to be treated as zero

    Returns:
        chi2 metric

    """
    assert im1.shape == im2.shape
    if mask is None:
        mask = np.ones_like(im2, dtype=bool)
    if mask_zeros:
        mask &= im2 > epsilon
    a_diff = im1[mask] - im2[mask]
    chisq = (a_diff ** 2 / im2[mask]).sum()
    return chisq


def mssim(im1, im2):
    """Return mean structural similarity index of im1 and im2
    According to Notes in
    http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.compare_ssim
    "To match the implementation of Wang et. al. [R261], set gaussian_weights to True,
    sigma to 1.5, and use_sample_covariance to False.
    [R261] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality
           assessment: From error visibility to structural similarity. IEEE Transactions on
           Image Processing, 13, 600-612.
           https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf"

    Parameters
    ----------
    im1, im2 : 2d arrays of floats
        images to compare

    Returns
    -------
    float value of mssim as per Wang et al.

    """
    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)
    assert mssim_version in {'old', 'new'}
    if mssim_version == 'old':
        mssim_kwargs = {}
    else:
        mssim_kwargs = {'gaussian_weights': True,
                        'sigma': 1.5, 'use_sample_covariance': False}
    return compare_ssim(im1, im2, **mssim_kwargs)


def tv(u, order=1):
    """Returns the total variation (TV) or BV seminorm of an array u, as defined by
    Osher et al. (2005).
    TV = |u|_{BV} = \int_\Omega |\nabla u|

    The gradient operator just applies a nearest-neighbour difference.
    It seems that both L1- and L2-norms are used for definitions of the |\cdot| operator in
    TV; We default to the L1 norm definition. Note that Wikipedia posits that |\cdot| is
    the L2-norm
    https://en.wikipedia.org/wiki/Total_variation
    However, from my assessment of the literature, both are used:
    http://dsp.stackexchange.com/questions/16271

    Args:
        u: 2d ndarray of float
        order: int (default = 1)
            1 or 2

    Returns:
        float

    """
    dx = np.diff(u, axis=0)
    dy = np.diff(u, axis=1)
    assert order in {1, 2}
    if order == 1:
        tv = np.abs(dx).sum() + np.abs(dy).sum()
    else:
        tv = np.sqrt((dx**2).sum() + (dy**2).sum())
    return tv


def append_to_running_log(filename, text):
    with open(filename, "a") as f:
        f.write(text)
