"""General helper functions"""

from skimage.io import imsave, imread
import skimage.transform as st
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import tifffile


def write_tiff32(filename, im):
    """Write a float32 tiff file.

    """
    # save a 32-bit tiff using either freeimage or tifffile
    im = im.astype(np.float32)
    try:
        imsave(filename, im, plugin='freeimage')
    except (ValueError, RuntimeError):
        tifffile.imsave(filename, im, compress=1)


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
    # load a 32-bit tiff using either freeimage or tifffile
    try:
        im = imread(filename, plugin='freeimage', as_grey=True)
    except (ValueError, RuntimeError):
        im = tifffile.imread(filename)
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


def zero_outside_mask(im, mask):
    """Zero the entries in im (i.e. not in a copy of im) that lie in the outer
    part of the mask image (based on the value of mask[0,0])
    
    Arguments:
    im - 2d ndarray
    mask - 2d ndarray

    """
    s = nd.generate_binary_structure(2, 2)      # 8-connected structure element
    mask = (mask==mask[0,0])
    la, _ = nd.label(mask, structure=s)
    im[la==la[0,0]] = 0.0


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
    
    Arguments:
    im - float array
    angle - rotation angle in degrees

    """
    assert issubclass(im.dtype.type, np.floating)

    scale = im.max()
    return st.rotate(im/scale, angle) * scale
