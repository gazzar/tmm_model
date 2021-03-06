#!/usr/bin/python3

"""Reconstruction code
"""

from __future__ import absolute_import, division, print_function
from . import config
import logging
import os
import numpy as np
from skimage.transform import iradon, iradon_sart
from acsemble import helpers
from acsemble.helpers import write_tiff32
import glob

logger = logging.getLogger(__name__)

UM_PER_CM = 1e4


def reconstruct_and_write(p, el, algorithm, anglelist=None):
    """Reconstruct from a sinogram

    Parameters:
    -----------
    el - name of current element, e.g. 'Fe'
    algorithm - one of f, s
        f - for conventional filtered backprojection with ramp filter
            [skimage iradon()]
        s - sart [skimage iradon_sart()]
    anglelist - list of angles in degrees

    """
    assert algorithm in ['f', 's']

    el_map0 = p.el_maps[el]
    sinogram = el_map0.astype(np.float64)

    # Rescale sinogram pixel quantities based on pixel side length.
    # The sinogram is a map of some quantity q per pixel,
    # which needs to be rescaled to units of [q]/cm.
    sinogram *= UM_PER_CM/p.um_per_px

    if algorithm == 'f':
        # conventional filtered backprojection
        # im = iradon(sinogram, anglelist, circle=True, interpolation='cubic')
        # im = iradon_sart(sinogram, anglelist, image=im)
        im = iradon(sinogram, anglelist, circle=True)

    if algorithm == 's':
        # 1-pass Simultaneous Algebraic Reconstruction Technique (SART)
        im = iradon_sart(sinogram, anglelist)

    # Get the filename that matches the glob pattern for this element
    # and prepend r_ to it
    pattern = p.filename

    matches = helpers.match_pattern(pattern, glob.glob(pattern))
    if matches:
        match_base = [m[0] for m in matches if el == m[1]][0]
    else:
        raise Exception('Element {} not found in {}'.format(el, matches))

    path = os.path.dirname(pattern)
    base = os.path.basename(match_base)
    filename = os.path.join(path, 'r_'+base)

    write_tiff32(filename, im)

    return im


def reconstruct(p, algorithm, anglesfile=None):
    """
    """
    if anglesfile is None:
        anglelist = None
    else:
        anglelist = np.loadtxt(anglesfile)
    for el in p.el_maps:
        reconstruct_and_write(p, el, algorithm, anglelist)


if __name__ == '__main__':
    import phantom

    BASE = r'R:\Science\XFM\GaryRuben\git_repos\tmm_model'

    os.chdir(os.path.join(BASE, r'acsemble\data'))

    p = phantom.Phantom2d(filename='s_golosio*.tiff')

    print(p)

    anglesfile = os.path.join(BASE, r'acsemble\data\angles.txt')
    reconstruct(p, 'f', anglesfile)
#    reconstruct(p, 's', anglesfile)
