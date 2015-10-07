import sys, os
PATH_HERE = os.path.abspath(os.path.dirname(__file__))
sys.path = [os.path.join(PATH_HERE, '..')] + sys.path
import unittest
from maia import Maia, Pad
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import transformations as tx
import numpy as np


MD_DATA = 'mask-Mo-3x100-R-10-T-04-gap-75-grow-09-tune-103-safe-01-glue-025-1.csv'
SIZE = 4


class CorrectLoadTests(unittest.TestCase):
    def setUp(self):
        self.det = Maia()

    def test_simple_load(self):
        self.assertTrue(isinstance(self.det, Maia))


class ChannelSelectionTests(unittest.TestCase):
    def setUp(self):
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


class PadTests(unittest.TestCase):
    def test_null_rotation(self):
        # Check that rotation matrix to reorient to a unit normal
        # oriented along z is the identity matrix.
        self.pad = Pad(id=1, centre_xyz=[0,0,0], unit_normal=[0,0,1],
                       width=1, height=1)
        r = self.pad._get_pad_rotation_matrix()
        np.testing.assert_allclose(r, np.eye(4))

    def test_rotation(self):
        # Check that rotation matrix calculated to reorient pad matches the
        # expected coordinate system of the transformations.py module.
        self.pad = Pad(id=2, centre_xyz=[0,0,0], unit_normal=[0,1,0],
                       width=1, height=1)
        r = self.pad._get_pad_rotation_matrix()
        # Matrix to rotate about an axis defined by a point and direction is
        # rotation_matrix(angle, direction, point=None):
        # To rotate from [0,0,1] to [0,1,0], rotate by 90deg about [1,0,0]
        tx_r1 = tx.rotation_matrix(-np.pi/2, [1,0,0])
        tx_r2 = tx.rotation_matrix(np.pi/2, [-1,0,0])
        pass

if __name__ == '__main__':
    import nose
    nose.main()