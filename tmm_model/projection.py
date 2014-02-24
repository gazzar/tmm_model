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
from skimage.transform import radon
#from skimage.io import imread, imsave
from helpers import write_tiff32, zero_outside_circle, rotate
from data_helpers import brain_attenuation
import glob, fnmatch
from helpers import imshow
import matplotlib.pyplot as plt


UM_PER_CM = 1e4

def g_radon(im, anglelist):
    """My attempt at a radon transform. I'll assume the "circle=True"
    condition that skimage makes, i.e. I'm ignoring what goes on outside the
    circular area whose diameter is equal to the side of the square array.

    Arguments:
    im - square 2d float32 array
    anglelist - list of angles in degrees

    """
    sinogram = np.empty((im.shape[0], len(anglelist)), dtype=np.float32)
    for i, angle in enumerate(anglelist):
        im_r = rotate(im, -angle)
        sinogram[:, i] = im_r.sum(axis=0)

    return sinogram


def project_with_absorption(p, matrix_map, anglelist):
    """Generates the sinogram accounting for absorption by the element,
    or compound in the case of the matrix.
    Use the Mass Attenuation Coefficient data from NIST XAAMDI database
    http://physics.nist.gov/PhysRefData/XrayMassCoef/chap2.html

    Arguments:
    p - phantom object
        p.energy - incident beam energy (keV)
        p.um_per_px - length of one pixel of the map (um)
    matrix_map - square 2d float32 array of element or compound abundance
    anglelist - list of angles in degrees

    matrix_map (g cm^-3)
    mu_on_rho units are cm^2 g^-1
    px_side (um)

    """
    sinogram = np.empty((matrix_map.shape[0], len(anglelist)), dtype=np.float32)
    b = brain_attenuation()
    for i, angle in enumerate(anglelist):
        im = rotate(matrix_map, -angle)
        sinogram[:, i] = (im.sum(axis=0) * b.mu_on_rho(p.energy) *
                          p.um_per_px/UM_PER_CM)
    return sinogram


def project_and_write(p, el, el_map0, algorithm, anglelist):
    """Project and write sinogram for element map el

    Arguments:
    p - phantom object
    el - name of current element, e.g. 'Fe'
    el_map0 - numpy float32 array with elemental abundance of el 
    algorithm - one of 'r', 'g', 'a'
        r - conventional radon as implemented in scikits-image
        g - my attempt at a conventional radon transform
        a - my attempt at a radon transform with absorption through the matrix
    anglelist - list of angles in degrees

    """
    assert algorithm in ['r', 'g', 'a']

    im = zero_outside_circle(el_map0)
    if algorithm == 'r':
        # conventional Radon transform
        im = radon(im, anglelist, circle=True)

    if algorithm == 'g':
        # my conventional Radon transform
        im = g_radon(im, anglelist)

    if algorithm == 'a':
        # my Radon transform with absorption
        im = project_with_absorption(p, im, anglelist)

    # Get the filename that matches the glob pattern for this element
    # and prepend s_ to it
    pattern = p.filename
    filenames = ['{}-{}{}'.format(
                   '-'.join(f.split('-')[:-1]), el, os.path.splitext(f)[1])
                 for f in glob.glob(pattern)]
    path, base = os.path.split(fnmatch.filter(filenames, pattern)[0])

    # Write sinogram (absorption map)
    s_filename = os.path.join(path, 's_'+base)
    write_tiff32(s_filename, np.rot90(im,3))            # rotate 90 deg cw


def project(p, algorithm, anglesfile):
    """
    """
    anglelist = np.loadtxt(anglesfile)
    for el in p.el_maps:
        el_map0 = p.el_maps[el]
        project_and_write(p, el, el_map0, algorithm, anglelist)


if __name__ == '__main__':
    import phantom

    DATA_DIR = r'R:\Science\XFM\GaryRuben\git_repos\tmm_model\tmm_model\data'
    os.chdir(DATA_DIR)

    anglesfile = r'R:\Science\XFM\GaryRuben\git_repos\tmm_model\commands\angles.txt'

    '''
    # split golosio into elements+matrix
    p = phantom.Phantom2d(filename='golosio_100.png', matrix_elements='N O C')
    p.split_map(DATA_DIR)
    '''

    p = phantom.Phantom2d(filename='golosio*matrix.tiff', um_per_px=10.0, energy=1)
    project(p, 'a', anglesfile)
