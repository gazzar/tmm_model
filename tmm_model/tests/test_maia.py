#!/usr/bin/env python

# Copyright (c) 2013, Gary Ruben
# Released under the Modified BSD license
# See LICENSE

import sys, os
PATH_HERE = os.path.abspath(os.path.dirname(__file__))
sys.path = [os.path.join(PATH_HERE, '..')] + sys.path
import unittest
from maia import Maia
import numpy as np


MD_DATA = 'mask-Mo-3x100-R-10-T-04-gap-75-grow-09-tune-103-safe-01-glue-025-1.csv'
SIZE = 4


class CorrectLoadTests(unittest.TestCase):
    def setUp(self):
        self.det = Maia()

    def test_simple_load(self):
        self.assertTrue(isinstance(self.det, Maia))


class CoordinateTests(unittest.TestCase):
    def setUp(self):
        self.det = Maia()

    def test_compare_coords(self):
        import maia_funcs

        # compare Matt's and Chris's detector element id maps
        cr_map = self.det.make_map(lambda:self.det.maia_data.index, -1.0)
        md_map = np.array(maia_funcs.getPixelMap())
        self.assertTrue(np.allclose(cr_map, md_map))


class SolidAngleTests(unittest.TestCase):
    def setUp(self):
        self.det = Maia()
        self.d_mm = 10.0
        self.a_mm = 5.0
        self.b_mm = 15.0
        self.A_mm = 25.0
        self.B_mm = 35.0

    def test_sa_rect1(self):
        a, b = self.a_mm, self.b_mm
        sa1 = self.det._rect_solid_angle(a, b, self.d_mm)
        sa2 = self.det._rect_solid_angle(b, a, self.d_mm)
        self.assertTrue(np.allclose(sa1, sa2))

    def test_sa_rect1(self):
        a, b = self.a_mm, self.b_mm
        sa1 = self.det._rect_solid_angle(a, b, self.d_mm)
        sa2 = self.det._rect_solid_angle(b, a, self.d_mm)
        self.assertTrue(np.allclose(sa1, sa2))

    def test_sa_rect2(self):
        a, b = self.a_mm, self.b_mm
        A, B = self.A_mm, self.B_mm
        sa1 = self.det._get_solid_angle(A, B, a, b, self.d_mm)
        sa2 = self.det._get_solid_angle(B, A, b, a, self.d_mm)
        self.assertTrue(np.allclose(sa1, sa2))
        ab_rect_sa = self.det._rect_solid_angle(a, b, self.d_mm)
        self.assertTrue(sa1 < ab_rect_sa)

    def test_sa_rect3(self):
        a, b = self.a_mm, self.b_mm
        A, B = self.A_mm, self.B_mm
        sa1 = self.det._get_solid_angle(A, B, a, b, self.d_mm)
        sa2 = self.det._get_solid_angle(A, B, b, a, self.d_mm)
        self.assertFalse(np.allclose(sa1, sa2))

    def test_sa_rect4(self):
        a, b = self.a_mm, self.b_mm
        A, B = self.A_mm, self.B_mm
        sa1 = self.det._get_solid_angle(A, B, a, b, self.d_mm)
        sa2 = self.det._get_solid_angle(B, A, a, b, self.d_mm)
        self.assertFalse(np.allclose(sa1, sa2))

if __name__ == '__main__':
    import nose
    nose.main()