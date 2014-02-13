#!/usr/bin/env python

# Copyright (c) 2014, Gary Ruben
# Released under the Modified BSD license
# See LICENSE


"""Projection class
Implements a class for 2D phantoms. The Golosio phantom is coded in terms of
geometry and composition data, with class methods supplied to instanciate it.
Alternatively, phantoms can be defined as a pair of files, one containing
geometry data as a greyscale bitmap and the other containing the composition
data structure defined in a yaml file.

"""

import os
import numpy as np
from skimage.transform import radon, rotate
from skimage.io import imread, imsave
from helpers import write_tiff32, zero_outside_circle
import glob, fnmatch


def project_and_write(el, el_map0, pattern, algorithm, anglelist):
    """Project and write sinogram for element map el

    Arguments:
    el - name of current element, e.g. 'Fe'
    el_map0 - numpy float32 array with elemental abundance of el 
    pattern - glob pattern of filenames which match the current one
    algorithm - 'r' for conventional radon as implemented in scikits-image
    anglelist - list of angles in degrees

    """
    assert algorithm in ['r']

    im = zero_outside_circle(el_map0)
    if algorithm == 'r':
        # conventional Radon transform
        im = radon(im, anglelist, circle=True).T

    # Get the filename that matches the glob pattern for this element
    # and prepend s_ to it
    filenames = ['{}-{}{}'.format(
                   '-'.join(f.split('-')[:-1]), el, os.path.splitext(f)[1])
                 for f in glob.glob(pattern)]
    path, base = os.path.split(fnmatch.filter(filenames, pattern)[0])
    filename = os.path.join(path, 's_'+base)

    write_tiff32(filename, im)


def project(p, algorithm, anglesfile):
    """
    """
    anglelist = np.loadtxt(anglesfile)
    for el in p.el_maps:
        el_map0 = p.el_maps[el]
        project_and_write(el, el_map0, p.filename, algorithm, anglelist)


if __name__ == '__main__':
    import phantom

    os.chdir(r'R:\Science\XFM\GaryRuben\git_repos\tmm_model\tmm_model\data')

    p = phantom.Phantom2d(filename='golosio*.tiff')

    anglesfile = r'R:\Science\XFM\GaryRuben\git_repos\tmm_model\commands\angles.txt'
    project(p, 'r', anglesfile)
