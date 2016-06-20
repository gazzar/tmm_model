#!/usr/bin/python3

from __future__ import absolute_import, division, print_function
import six
import context
import unittest
from acsemble import helpers
import numpy as np
import os


class MatchPatternTests(unittest.TestCase):
    def test_match_pattern_one(self):
        matches = helpers.match_pattern('a*z', ['abz'])
        self.assertSequenceEqual(matches, [('abz', 'b')])
        matches = helpers.match_pattern('a*z.ext', ['abz.ext'])
        self.assertSequenceEqual(matches, [('abz.ext', 'b')])

    def test_match_pattern_many(self):
        matches = helpers.match_pattern('a*z', ['abc', 'abcz', 'az'])
        self.assertSequenceEqual(matches, [('abcz', 'bc'), ('az', '')])

    def test_match_nopattern(self):
        matches = helpers.match_pattern('abc', ['abc', 'abcz', 'az'])
        self.assertSequenceEqual(matches, [('abc', 'abc')])

    def test_no_match(self):
        matches = helpers.match_pattern('a*z', ['abc'])
        self.assertSequenceEqual(matches, [])

def test_match_path_pattern():
    matches = helpers.match_path_pattern('x/y\\a*z', ['x/y\\abz'])
    assert(matches == [(os.path.join('x', 'y', 'abz'), 'b')])

if __name__ == "__main__" :
    import sys
    from numpy.testing import run_module_suite
    run_module_suite(argv=sys.argv)
