import sys, os
PATH_HERE = os.path.abspath(os.path.dirname(__file__))
sys.path = [
            os.path.join(PATH_HERE, '..'),
            os.path.join(PATH_HERE, '..', '..'),    # include path to version.py
           ] + sys.path
import config
import unittest
import numpy as np
import iradons


class IRadonTests(unittest.TestCase):
    def test_skimage_iradon(self):
        p = np.array([[1.0]])
        r = iradons.iradon(p, angles=np.r_[0.0, 1.0])
        self.assertTrue(np.allclose(r, r))


if __name__ == "__main__":
    from numpy.testing import run_module_suite
    run_module_suite()
