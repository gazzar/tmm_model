#!/usr/bin/python3

from __future__ import absolute_import, division, print_function
import six
import context

import unittest
from acsemble.maia import Maia, Pad
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import acsemble.transformations as tx
import numpy as np
from acsemble import config

def setup_module():
    config.config_init()


MD_DATA = 'mask-Mo-3x100-R-10-T-04-gap-75-grow-09-tune-103-safe-01-glue-025-1.csv'
SIZE = 4


class CorrectLoadTests(unittest.TestCase):
    def setUp(self):
        self.det = Maia()

    def test_simple_load(self):
        self.assertTrue(isinstance(self.det, Maia))

    def tearDown(self):
        """The Pad class maintains an internal check to ensure pad objects
        aren't replaced. Clear the internal check state.

        """
        Pad.clear_pads()


class ChannelSelectionTests(unittest.TestCase):
    def setUp(self):
        Pad.clear_pads()
        self.det = Maia()

    def test_quadrant_selection(self):
        quadrant = list(self.det.channel_selection(quadrant=0))
        self.assertTrue(len(quadrant)==384/4)

    def test_quadrants_selection(self):
        quadrant = list(self.det.channel_selection(quadrant=(0, 2)))
        self.assertTrue(len(quadrant)==384/2)

    def test_row0_selection(self):
        row = list(self.det.channel_selection(row=0))
        self.assertTrue(len(row)==20)

    def test_row10_selection(self):
        row = list(self.det.channel_selection(row=10))
        self.assertTrue(len(row)==16)

    def test_col0_selection(self):
        col = list(self.det.channel_selection(col=0))
        self.assertTrue(len(col)==20)

    def test_col10_selection(self):
        col = list(self.det.channel_selection(col=10))
        self.assertTrue(len(col)==16)

    def test_row0col0_selection(self):
        ch = list(self.det.channel_selection(row=0, col=0))
        self.assertTrue(len(ch)==1)

    def test_rowscols_selection(self):
        ch = list(self.det.channel_selection(row=(0, 2), col=(0, 1)))
        self.assertTrue(len(ch)==4)

    def test_rowscols_selection2(self):
        ch = list(self.det.channel_selection(row=(0, 8), col=(0, 8)))
        self.assertTrue(len(ch)==3)

    def test_rowscols_range(self):
        ch = list(self.det.channel_selection(row=range(9), col=range(9)))
        self.assertTrue(len(ch)==80)        # 9x9-1=80

    def tearDown(self):
        """The Pad class maintains an internal check to ensure pad objects
        aren't replaced. Clear the internal check state.

        """
        Pad.clear_pads()

class PadTests(unittest.TestCase):
    def setUp(self):
        # pad lies along -ve z-axis (normal doesn't matter)
        self.pad_geometry = (0, 0, 0, 1, 1)  # x,y,z,w,h

    def test_null_transform(self):
        self.pad = Pad(ident=1, pad_geometry=self.pad_geometry,
                       detector_centre_mm=[0, 0, 0],
                       detector_unit_normal=[0, 0, 1])
        r = self.pad._get_pad_transform_matrix(detector_centre_mm=[0, 0, 0])
        np.testing.assert_allclose(r, np.eye(4))

    @unittest.skip
    def test_rotation(self):
        # Check that rotation matrix calculated to reorient pad matches the
        # expected coordinate system of the transformations.py module.
        self.pad = Pad(ident=1, pad_geometry=self.pad_geometry,
                       detector_centre_mm=[0, 0, 0],
                       detector_unit_normal=[0, 1, 0])
        r = self.pad._get_pad_transform_matrix(detector_centre_mm=[0, 0, 0])
        # Matrix to rotate about an axis defined by a point and direction is
        # rotation_matrix(angle, direction, point=None):
        # To rotate from [0,0,1] to [0,1,0], rotate by 90deg about [1,0,0]
        tx_r1 = tx.rotation_matrix(-np.pi/2, [1,0,0])
        tx_r2 = tx.rotation_matrix(np.pi/2, [-1,0,0])

    def test_spherical_from_cartesian1(self):
        self.pad = Pad(ident=1, pad_geometry=self.pad_geometry,
                       detector_centre_mm=[0, 0, 1],
                       detector_unit_normal=[0, 0, 1])
        self.assertAlmostEqual(self.pad.theta, 0.0)

    def test_spherical_from_cartesian2(self):
        self.pad = Pad(ident=1, pad_geometry=self.pad_geometry,
                       detector_centre_mm=[1, 0, 0],
                       detector_unit_normal=[0, 0, 1])
        self.assertAlmostEqual(self.pad.theta, np.pi/2)
        self.assertAlmostEqual(self.pad.phi, 0.0)

    def test_spherical_from_cartesian3(self):
        self.pad = Pad(ident=1, pad_geometry=self.pad_geometry,
                       detector_centre_mm=[0, 1, 0],
                       detector_unit_normal=[0, 0, 1])
        self.assertAlmostEqual(self.pad.theta, np.pi/2)
        self.assertAlmostEqual(self.pad.phi, np.pi/2)

    def test_spherical_from_cartesian4(self):
        self.pad = Pad(ident=1, pad_geometry=self.pad_geometry,
                       detector_centre_mm=[0, 0, -1],
                       detector_unit_normal=[0, 0, 1])
        self.assertAlmostEqual(self.pad.theta, np.pi)

    def test_spherical_from_cartesian5(self):
        self.pad = Pad(ident=1, pad_geometry=self.pad_geometry,
                       detector_centre_mm=[-1, 0, 0],
                       detector_unit_normal=[0, 0, 1])
        self.assertAlmostEqual(self.pad.theta, np.pi/2)
        self.assertAlmostEqual(self.pad.phi, np.pi)

    def test_spherical_from_cartesian6(self):
        self.pad = Pad(ident=8, pad_geometry=self.pad_geometry,
                       detector_centre_mm=[0, -1, 0],
                       detector_unit_normal=[0, 0, 1])
        self.assertAlmostEqual(self.pad.theta, np.pi/2)
        self.assertAlmostEqual(self.pad.phi, -np.pi/2)

    def tearDown(self):
        """The Pad class maintains an internal check to ensure pad objects
        aren't replaced. Clear the internal check state.

        """
        Pad.clear_pads()


class SolidAngleTests(unittest.TestCase):
    def setUp(self):
        self.det = Maia()
        self.pad = self.det.pads[1]

        # A - x-coord of detector element corner closest to the origin
        # B - y-coord of detector element corner closest to the origin
        # a - width of detector element
        # b - height of detector element
        # d - distance to plane of detector
        self.A_mm = np.min(self.pad.vertices[:-1,0], axis=0)
        self.B_mm = np.min(self.pad.vertices[:-1,1], axis=0)
        self.a_mm = self.pad.width
        self.b_mm = self.pad.height
        self.d_mm = self.pad.vertices[3,2]

    def test_solid_angle(self):
        omega_pad = self.pad.solid_angle()
        omega_det_pad = self.det._get_solid_angle(
            self.A_mm, self.B_mm, self.a_mm, self.b_mm, self.d_mm)
        self.assertAlmostEqual(omega_pad, omega_det_pad)

    def test_all_solid_angles(self):
        for p in self.det.pads.values():
            A_mm = np.min(np.abs(p.vertices[:-1,0]), axis=0)
            B_mm = np.min(np.abs(p.vertices[:-1,1]), axis=0)
            a_mm = p.width
            b_mm = p.height
            d_mm = p.vertices[3,2]
            omega_pad = p.solid_angle()
            omega_det_pad = self.det._get_solid_angle(A_mm, B_mm, a_mm, b_mm,
                                                      d_mm)
            self.assertAlmostEqual(omega_pad, omega_det_pad)

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

    def test_sa_rect5(self):
        A, B = 0.0, 0.0
        a1, b1 = 2.0, 1.0
        a2, b2 = 2.0, -1.0
        a3, b3 = -2.0, 1.0
        a4, b4 = -2.0, -1.0
        # get solid angles of rectangles, none of which cross the origin
        sa1 = self.det._get_solid_angle(A, B, a1, b1, self.d_mm)
        sa2 = self.det._get_solid_angle(A, B, a2, b2, self.d_mm)
        sa3 = self.det._get_solid_angle(A, B, a3, b3, self.d_mm)
        sa4 = self.det._get_solid_angle(A, B, a4, b4, self.d_mm)

        # get solid angle of a corresponding rectangle crossing the origin
        pad_geometry = (0, 0, 0, 4.0, 2.0)  # x,y,z,w,h
        p = Pad(ident=384, pad_geometry=pad_geometry,
                detector_centre_mm=[0, 0, self.d_mm],
                detector_unit_normal=[0, 0, 1])
        sa5 = p.solid_angle()

        self.assertTrue(np.allclose(sa1+sa2+sa3+sa4, sa5))

    def test_sa_rect6(self):
        A, B = 0.0, 0.0
        a1, b1 = 1.0, 1.0
        # get solid angles of rectangles, none of which cross the origin
        sa1 = self.det._get_solid_angle(A, B, a1, b1, self.d_mm)

        # get solid angle of a corresponding rectangle crossing the origin
        pad_geometry = (0, 0, 0, 1.0, 1.0)  # x,y,z,w,h
        p = Pad(ident=384, pad_geometry=pad_geometry,
                detector_centre_mm=[0.5, 0.5, self.d_mm],
                detector_unit_normal=[0, 0, 1])
        sa5 = p.solid_angle()

        self.assertTrue(np.allclose(sa1, sa5))

    def test_sa_rect7(self):
        # get solid angle of a rectangle
        pad_geometry = (0.5, 0.5, 0, 1.0, 1.0)  # x,y,z,w,h
        p = Pad(ident=385, pad_geometry=pad_geometry,
                detector_centre_mm=[0.0, 0.0, self.d_mm],
                detector_unit_normal=[0, 0, 1])
        sa1 = p.solid_angle()

        # get solid angle of a corresponding rectangle oriented towards the same
        # sphere centre
        p = Pad(ident=386, pad_geometry=pad_geometry,
                detector_centre_mm=[self.d_mm, 0.0, 0.0],
                detector_unit_normal=[1, 0, 0])
        sa2 = p.solid_angle()
        self.assertTrue(np.allclose(sa1, sa2))

        # get solid angle of a corresponding rectangle oriented towards the same
        # sphere centre
        p = Pad(ident=387, pad_geometry=pad_geometry,
                detector_centre_mm=[0.0, 0.0, -self.d_mm],
                detector_unit_normal=[0, 0, 1])
        sa3 = p.solid_angle()

        self.assertTrue(np.allclose(sa1, sa3))

    def test_sa_rect8(self):
        # get solid angle of a rectangle
        pad_geometry = (0, 0, 0, 2.0, 2.0)  # x,y,z,w,h
        p = Pad(ident=384, pad_geometry=pad_geometry,
                detector_centre_mm=[0.0, 0.5, self.d_mm],
                detector_unit_normal=[0, 0, 1])
        sa1 = p.solid_angle()

        # get solid angle of a corresponding rectangle oriented towards the same
        # sphere centre
        p = Pad(ident=385, pad_geometry=pad_geometry,
                detector_centre_mm=[0.5, 0.0, self.d_mm],
                detector_unit_normal=[0, 0, 1])
        sa2 = p.solid_angle()

        self.assertTrue(np.allclose(sa1, sa2))

    def test_sa_rect9(self):
        A, B = 0.0, 0.0
        a1, b1 = 1.0, 1.0
        # get solid angles of rectangles, none of which cross the origin
        sa1 = self.det._get_solid_angle(A, B, a1, b1, self.d_mm)

        # get solid angle of a corresponding rectangle crossing the origin
        pad_geometry = (0.5, 0.5, 0, 1.0, 1.0)  # x,y,z,w,h
        p = Pad(ident=384, pad_geometry=pad_geometry,
                detector_centre_mm=[0.0, 0.0, self.d_mm],
                detector_unit_normal=[0, 0, 1])
        sa2 = p.solid_angle()

        self.assertTrue(np.allclose(sa1, sa2))

        # get solid angle of a corresponding rectangle crossing the origin
        pad_geometry = (0.0, 0.0, 0, 1.0, 1.0)  # x,y,z,w,h
        p = Pad(ident=385, pad_geometry=pad_geometry,
                detector_centre_mm=[0.5, 0.5, self.d_mm],
                detector_unit_normal=[0, 0, 1])
        sa3 = p.solid_angle()

        self.assertTrue(np.allclose(sa1, sa3))

    def tearDown(self):
        """The Pad class maintains an internal check to ensure pad objects
        aren't replaced. Clear the internal check state.

        """
        Pad.clear_pads()

if __name__ == "__main__" :
    import sys
    from numpy.testing import run_module_suite
    run_module_suite(argv=sys.argv)
