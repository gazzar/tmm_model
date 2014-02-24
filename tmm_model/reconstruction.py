#!/usr/bin/env python

# Copyright (c) 2014, Gary Ruben
# Released under the Modified BSD license
# See LICENSE


"""Reconstruction code
"""

import os
import numpy as np
from skimage.transform import iradon, iradon_sart, rotate
from helpers import write_tiff32
import glob, fnmatch


def reconstruct_and_write(el, el_map0, pattern, algorithm, anglelist=None):
    """Reconstruct from a sinogram

    Arguments:
    el - name of current element, e.g. 'Fe'
    el_map0 - numpy float32 array with elemental abundance of el 
    pattern - glob pattern of filenames which match the current one
    algorithm - one of f, s
        f - for conventional filtered backprojection with ramp filter
            [skimage iradon()]
        s - sart [skimage iradon_sart()]
    anglelist - list of angles in degrees

    """
    assert algorithm in ['f', 's']

    sinogram = np.rot90(el_map0).astype(np.float64)
    if algorithm == 'f':
        # conventional filtered backprojection
        im = iradon(sinogram, anglelist, circle=True)

    if algorithm == 's':
        # 1-pass Simultaneous Algebraic Reconstruction Technique (SART)
        im = iradon_sart(sinogram, anglelist)

    # Get the filename that matches the glob pattern for this element
    # and prepend r_ to it
    filenames = ['{}-{}{}'.format(
                   '-'.join(f.split('-')[:-1]), el, os.path.splitext(f)[1])
                 for f in glob.glob(pattern)]
    path, base = os.path.split(fnmatch.filter(filenames, pattern)[0])
    filename = os.path.join(path, 'r_'+base)

    write_tiff32(filename, im)


def reconstruct(p, algorithm, anglesfile=None):
    """
    """
    if anglesfile is None:
        anglelist = None
    else:
        anglelist = np.loadtxt(anglesfile)
    for el in p.el_maps:
        el_map0 = p.el_maps[el]
        reconstruct_and_write(el, el_map0, p.filename, algorithm, anglelist)


if __name__ == '__main__':
    import phantom

    os.chdir(r'R:\Science\XFM\GaryRuben\git_repos\tmm_model\tmm_model\data')

    p = phantom.Phantom2d(filename='s_golosio*.tiff')

    anglesfile = r'R:\Science\XFM\GaryRuben\git_repos\tmm_model\commands\angles.txt'
    reconstruct(p, 'f', anglesfile)
#    reconstruct(p, 's', anglesfile)
