import sys, os

PATH_HERE = os.path.abspath(os.path.dirname(__file__))
sys.path = [
            os.path.join(PATH_HERE, '..'),
            os.path.join(PATH_HERE, '..', '..'),    # include path to version.py
           ] + sys.path
import config
import unittest
import data_helpers
import helpers
import numpy as np
from phantom import Phantom2d


GOLOSIO_MAP = os.path.join(PATH_HERE, '..', 'data', 'golosio_100.png')
YAMLFILE = os.path.join(PATH_HERE, '..', 'data', 'golosio.yaml')

class MatrixPropertiesTests(unittest.TestCase):
    #TODO: Test that matrix compound is normalised; test assertion in reader
    #      is tripped properly
    @unittest.skip('')
    def test_read_compound(self):
        p = Phantom2d(filename=GOLOSIO_MAP, yamlfile=YAMLFILE)
        self.phantom = p
        matches = data_helpers.MatrixProperties(p)

    @unittest.skip('')
    def test_no_match(self):
        matches = helpers.match_pattern('a*z', ['abc'])
        self.assertSequenceEqual(matches, [])

if __name__ == "__main__":
    from numpy.testing import run_module_suite
    run_module_suite()
