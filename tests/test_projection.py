#!/usr/bin/python3

from __future__ import absolute_import, division, print_function
import six
import context
import unittest
from acsemble.phantom import Phantom2d
from acsemble import projection
import os
import numpy as np

PATH_HERE = os.path.abspath(os.path.dirname(__file__))
GOLOSIO_MAP = os.path.join(PATH_HERE, '..', 'acsemble', 'data', 'golosio_100.png')
YAMLFILE = os.path.join(PATH_HERE, '..', 'acsemble', 'data', 'golosio.yaml')


class OutgoingPhotonEnergyTests(unittest.TestCase):
    def setUp(self):
        self.phantom = Phantom2d(filename=GOLOSIO_MAP, yamlfile=YAMLFILE)

    def test_fluoro(self):
        energy = projection.outgoing_photon_energy(
            event_type = 'fluoro',
            p = self.phantom,
            q = 199,        # outer corner
            el = 'Fe'       # K_alpha is 6.398
        )
        self.assertAlmostEqual(energy, 6.398, places=2)

    def test_rayleigh(self):
        energy = projection.outgoing_photon_energy(
            event_type = 'rayleigh',
            p = self.phantom,
            q = 199,        # outer corner
        )
        self.assertAlmostEqual(energy, self.phantom.energy)

    @unittest.skip
    def test_compton(self):
        energy_outer = projection.outgoing_photon_energy(
            event_type = 'compton',
            p = self.phantom,
            q = 199,        # outer corner
        )
        energy_inner = projection.outgoing_photon_energy(
            event_type = 'compton',
            p = self.phantom,
            q = 248,        # inner corner
        )
        self.assertTrue(energy_outer > energy_inner)      # because backscatter
        self.assertTrue(energy_outer < self.phantom.energy)
        # self.assertAlmostEqual(energy_outer, 14.324, places=3) # Maia Rev A
        self.assertAlmostEqual(energy_outer, 14.282, places=3) # For Maia Rev C

if __name__ == "__main__" :
    import sys
    from numpy.testing import run_module_suite
    run_module_suite(argv=sys.argv)
