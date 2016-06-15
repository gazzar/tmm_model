#!/usr/bin/python3

from __future__ import absolute_import, division, print_function
import six
import context
import unittest
import numpy as np
from acsemble import mlem


def test_noisify():
    im = np.array([[1.0, 2.0], [3.0, 4.0]])
    noisy_im1 = mlem.noisify(im, frac=0.1)
    noisy_im2 = mlem.noisify(im, frac=0.5)
    assert(np.dot(noisy_im1, noisy_im1).sum() < np.dot(noisy_im2, noisy_im2).sum())


if __name__ == "__main__" :
    import sys
    from numpy.testing import run_module_suite
    run_module_suite(argv=sys.argv)
