import sys, os
PATH_HERE = os.path.abspath(os.path.dirname(__file__))
sys.path = [
            os.path.join(PATH_HERE, '..'),
            os.path.join(PATH_HERE, '..', '..'),    # include path to version.py
           ] + sys.path
import unittest
import phantom
import mlem
import numpy as np
import maia
import config


""" The following pattern is designed to match
Ni_test_phantom-N.tiff,
Ni_test_phantom-Ni.tiff
and
Ni_test_phantom-matrix.tiff

This assumes that these have been generated from the base
Ni_test_phantom.svg and Ni_test_phantom.yaml files via
calling ImageMagick (i.e. run the remap-Ni_test_phantom.py script)

"""
ELEMENT = 'Ni'
MAP_PATTERN = os.path.join(PATH_HERE, '..', 'data', 'Ni_test_phantom-*.tiff')
YAMLFILE = os.path.join(PATH_HERE, '..', 'data', 'Ni_test_phantom.yaml')
UM_PER_CM = 1e4
WIDTH_UM = 100.0
WIDTH_PX = 100
UM_PER_PX = WIDTH_UM / WIDTH_PX
ENERGY_KEV = 15.6


class DensityFromFluoroTests(unittest.TestCase):
    def setUp(self):
        # create a p instance
        self.p = phantom.Phantom2d(
            filename=MAP_PATTERN,
            yamlfile=YAMLFILE,
            um_per_px=UM_PER_PX,
            energy=ENERGY_KEV,
        )
        maia_d = maia.Maia()
        self.q = maia_d.channel(7, 7).index[0]
        self.el = ELEMENT
        config.parse()  # read config file settings

    # See http://stackoverflow.com/questions/16134281/python-mocking-a-method-from-an-imported-module
    # @mock.patch('config.no_out_absorption')
    # @mock.patch('config.no_in_absorption')
    def test_interface(self):
        config.no_out_absorption = False
        config.no_in_absorption = False
        density = mlem.density_from_fluorescence_for_el(self.p, self.q, self.el)
        self.assertTrue(isinstance(density, float))
