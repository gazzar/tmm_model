import sys, os
PATH_HERE = os.path.abspath(os.path.dirname(__file__))
sys.path = [
            os.path.join(PATH_HERE, '..'),
            os.path.join(PATH_HERE, '..', '..'),    # include path to version.py
           ] + sys.path
import config
import unittest
from phantom import Phantom2d
import numpy as np


GOLOSIO_MAP = os.path.join(PATH_HERE, '..', 'data', 'golosio_100.png')
YAMLFILE = os.path.join(PATH_HERE, '..', 'data', 'golosio.yaml')


class LoadMapTests(unittest.TestCase):
    def setUp(self):
        self.phantom = Phantom2d(filename=GOLOSIO_MAP, yamlfile=YAMLFILE)

    def test_simple_load(self):
        self.assertTrue(isinstance(self.phantom, Phantom2d))

    def test_correct_load(self):
        self.assertEqual(self.phantom.rows, 100)
        self.assertEqual(self.phantom.cols, 100)
        self.assertEqual(self.phantom.um_per_px, 1.0)
        self.assertEqual(self.phantom.phantom_array.min(), 0)
        self.assertEqual(self.phantom.phantom_array.max(), 3)

class RotationTests(unittest.TestCase):
    def setUp(self):
        self.phantom = Phantom2d(filename=GOLOSIO_MAP, yamlfile=YAMLFILE)

    def test_0deg(self):
        p = self.phantom.rotate(0)
        self.assertTrue(np.allclose(p, self.phantom.phantom_array))

    def test_360deg(self):
        p = self.phantom.rotate(360)
        self.assertTrue(np.allclose(p, self.phantom.phantom_array))

    def test_90deg(self):
        p = self.phantom.rotate(90)
        self.assertTrue(np.allclose(p, np.rot90(self.phantom.phantom_array)))

class YamlFileSanityTests(unittest.TestCase):
    def setUp(self):
        self.phantom = Phantom2d(filename=GOLOSIO_MAP, yamlfile=YAMLFILE)

    def verify_weights_sum_to_one(self):
        for compound in self.phantom.compounds.values():
            self.assertAlmostEqual(sum(compound[1].values()), 1.0)


if __name__ == "__main__":
    from numpy.testing import run_module_suite
    run_module_suite()
()