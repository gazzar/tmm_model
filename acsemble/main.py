#!/usr/bin/env python

# Copyright (c) 2014, Gary Ruben
# Released under the Modified BSD license
# See LICENSE

"""The main forward model
A simple forward model of interaction with a phantom
that accounts for self-absorption of entering and exiting rays and produces
Rayleigh, Compton and fluorescence signals in a model of Maia detector.
Initially, the phantom will be 2D so that all interactions occur within the
plane of the incident beam.

A rotation series will be produced for the beam as each position along the
breadth of the phantom, defined by the max. projected breadth over all rotation
angles of the longest dimension of non-0 indexed phase.

Some consideration will be given to parallelisation of the computation.

for each rotation angle
    for each translation
        generate a beam
        # incoming absorption
        propagate from source through all pixels (voxels) until exiting material
        for each ray section intersecting a different material
            reduce intensity by exp(-mu t)
        # Now we have a map of intensity at every pixel in the map
        # scattering and fluorescence
        for each pixel in the current map (phantom at some rotation angle to
                detector)
            for each Maia detector element
                determine ray from map pixel to Maia element and its solid angle
                determine Rayleigh, Compton and main fluorescence contributions
                for each of Rayleigh, Compton and fluorescence contributions
                    for each ray section intersecting a different material
                        reduce intensity by exp(-mu t)

"""

import numpy as np
from ConfigParser import SafeConfigParser as ConfigParser
import maia, phantom


def read_config():
    global filename, side_mm, rotations, translations
    global sample_centre_to_detector_mm

    # read model config
    config = ConfigParser()
    config.read('config.ini')
    filename = config.get('phantom', 'filename')
    side_mm = config.get('phantom', 'side_mm')
    rotations = config.get('scan', 'rotations')
    translations = config.get('scan', 'translations')
    sample_centre_to_detector_mm = config.get(
        'geometry', 'sample_centre_to_detector_mm')


def project_beam(translation):
    """
    propagate from source through all pixels (voxels) until exiting material
    for each ray section intersecting a different material
        reduce intensity by exp(-mu t)
    # Now we have a map of intensity at every pixel in the map
    # scattering and fluorescence
    for each pixel in the current map (phantom at some rotation angle to
            detector)
        for each Maia detector element
            determine ray from map pixel to Maia element and its solid angle
            determine Rayleigh, Compton and main fluorescence contributions
            for each of Rayleigh, Compton and fluorescence contributions
                for each ray section intersecting a different material
                    reduce intensity by exp(-mu t)

    """
    pass


read_config()
p = phantom.Phantom2d(filename=filename)
# p.rotate(45)
# p.show(1)
for angle in np.linspace(0, 360, rotations, endpoint=False):
    p.rotate(angle)
    for t in np.linspace(0, p.rows, translations):
        project_beam(t)