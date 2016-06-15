#!/usr/bin/python3

from __future__ import absolute_import, division, print_function
import six
import context
import acsemble
from nose.tools import assert_raises
import unittest
from acsemble import config
from acsemble import data_helpers


@unittest.skip('')
def test_matrixproperty_creation():
    #TODO: Test that matrix compound is normalised; test assertion in reader
    #      is tripped properly
    p = Phantom2d(filename=GOLOSIO_MAP, yamlfile=YAMLFILE)
    self.phantom = p
    matches = data_helpers.MatrixProperties(p)

@unittest.skip('')
def test_no_match(self):
    matches = helpers.match_pattern('a*z', ['abc'])
    self.assertSequenceEqual(matches, [])


if __name__ == "__main__" :
    import sys
    from numpy.testing import run_module_suite
    run_module_suite(argv=sys.argv)
