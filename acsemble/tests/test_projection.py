import sys, os
PATH_HERE = os.path.abspath(os.path.dirname(__file__))
sys.path = [os.path.join(PATH_HERE, '..')] + sys.path
import unittest
from phantom import Phantom2d, golosio_compounds, golosio_geometry
import projection
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

class DrawGolosioPhantomTest(unittest.TestCase):
    def test_make_golosio_phantom(self):
        PX_SIDE = 200
        p = Phantom2d(size=(PX_SIDE,PX_SIDE), um_per_px=1.0/PX_SIDE)
        for thing in golosio_geometry:
            p.add_shape(thing)

class CheckValuesTests(unittest.TestCase):
    def setUp(self):
        self.phantom = Phantom2d(filename=GOLOSIO_MAP, yamlfile=YAMLFILE)

    def test_golosio_compound_at_0_0(self):
        p_0_0 = self.phantom.compound_record(golosio_compounds, row=0, col=0)
        self.assertEqual(p_0_0[1][0][0], 'N')

        d_0_0 = self.phantom.density(golosio_compounds, row=0, col=0)
        self.assertEqual(d_0_0, 1.2e-3)

    def test_golosio_compound_at_35_35(self):
        p_35_35 = self.phantom.compound_record(golosio_compounds, row=35, col=35)
        self.assertEqual(p_35_35[1][2][1], 0.4)     # Ca_20 conc. for compound 2

        d_35_35 = self.phantom.density(golosio_compounds, row=35, col=35)
        self.assertEqual(d_35_35, 3.5)

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

class OutgoingPhotonEnergyTests(unittest.TestCase):
    def setUp(self):
        self.phantom = Phantom2d(filename=GOLOSIO_MAP, yamlfile=YAMLFILE)

    def test_fluoro(self):
        energy = projection.outgoing_photon_energy(
            event_type = 'fluoro',
            p = self.phantom,
            q = 199,        # outer corner
            el = 'Fe'       # K_alpha is 6.398
        )
        self.assertAlmostEqual(energy, 6.398, places=2)

    def test_rayleigh(self):
        energy = projection.outgoing_photon_energy(
            event_type = 'rayleigh',
            p = self.phantom,
            q = 199,        # outer corner
        )
        self.assertAlmostEqual(energy, self.phantom.energy)


    def test_compton(self):
        energy_outer = projection.outgoing_photon_energy(
            event_type = 'compton',
            p = self.phantom,
            q = 199,        # outer corner
        )
        energy_inner = projection.outgoing_photon_energy(
            event_type = 'compton',
            p = self.phantom,
            q = 248,        # inner corner
        )
        self.assertTrue(energy_outer > energy_inner)      # because backscatter
        self.assertTrue(energy_outer < self.phantom.energy)
        self.assertAlmostEqual(energy_outer, 14.324, places=3)

if __name__ == '__main__':
    import nose
    nose.main()