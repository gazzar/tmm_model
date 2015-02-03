import sys, os
PATH_HERE = os.path.abspath(os.path.dirname(__file__))
sys.path = [os.path.join(PATH_HERE, '..')] + sys.path
import unittest
import helpers
import numpy as np


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

if __name__ == '__main__':
    import nose
    nose.main()