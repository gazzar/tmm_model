"""mlem algorithm
Implements the iterative mlem algorithm, performing projection and
backprojection in a loop.

My advice for learning about MLEM is to look at two books:
[1] G. L. Zeng, Medical image reconstruction: A Conceptual Tutorial.
                Springer, 2010.
[2] M. N. Wernick and J. N. Aarsvold, Emission Tomography: The Fundamentals of
                                      PET and SPECT. Academic Press, 2004.

"""

from __future__ import print_function

import sys, os

# Set environ so that matplotlib uses v2 interface to Qt, because we are
# using mayavi's mlab interface in maia.py
os.environ.update(
    {'QT_API': 'pyqt', 'ETS_TOOLKIT': 'qt4'}
)

import config           # keep this near the top of the imports
import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
import imageio
import projection
import data_helpers


UM_PER_CM = 1e4


class Mlem(object):
    def __init__(self, p, el, projector, backprojector, g_j, angles=None):
        """
        p : Phantom2d reference
        el : The element this mlem loop refers to
        projector : function
            Function that takes an image and transforms it to
            projection space.
        backprojector : function
            Function that takes a sinogram and transforms it to
            image space.
        g_j : observed sinogram data.
        angles : ndarray of angles (in degrees) for projector and backprojector.

        """
        self.p = p
        self.el = el
        self.project = projector
        self.backproject = backprojector
        self.angles = angles
        self.g_j = g_j
        self.epsilon = np.mean(np.abs(g_j)) / 1000.0
        self.sum_g_j = g_j.sum()
        self.i = 0
        self.f = np.ones_like(backprojector(g_j, angles))
        self.weighting = backprojector(np.ones_like(g_j), angles)
        assert self.f.shape == self.weighting.shape
        self.weighting.clip(self.epsilon, out=self.weighting)

    def iterate(self):
        # From Lalush and Wernick (See [2]);
        # f^\hat <- (f^\hat / |\sum h|) * \sum h * (g_j / g)          ... (*)
        # where g = \sum (h f^\hat)                                   ... (**)
        #
        # self.f is the current estimate f^\hat
        # The following g from (**) is equivalent to g = \sum (h f^\hat)
        g = self.project(self.f, angles=self.angles)

        g.clip(min=self.epsilon, out=g)

        if config.show_images:
            self.g = g      # Save this for diagnostic purposes
            self.imsave_g()
            im = self.backproject(g, angles=self.angles)
            imageio.imsave('mlem_d_%03d.tif' % self.i, im.astype(np.float32))
        # form parenthesised term (g_j / g) from (*)
        r = self.g_j / g

        # backproject to form \sum h * (g_j / g)
        g_r = self.backproject(r, angles=self.angles)

        # Renormalise backprojected term / \sum h)
        # Normalise the individual pixels in the reconstruction
        self.f *= g_r / self.weighting

        if config.show_images:
            self.imsave_f()

        # print some progress output
        if 0:
            print('.', end='')
        else:
            self.stats()

        self.i += 1

    def _imshow(self, im, show=True):
        plt.figure()
        plt.imshow(im, interpolation='nearest', cmap='YlGnBu_r')
        plt.title('iteration:%d' % self.i)
        plt.colorbar()
        if show:
            plt.show()

    def stats(self):
        print(
            'i={i} f_sum={f_sum} f_min={f_min} f_max={f_max} '
            'g_j_sum={g_j_sum}'.format(
                i=self.i,
                f_sum=self.f.sum(),
                f_min=self.f.min(),
                f_max=self.f.max(),
                g_j_sum=self.sum_g_j,
            )
        )

    def imsave_f(self, name_pattern='mlem_f_%03d.tif'):
        imageio.imsave(name_pattern % self.i, self.f.astype(np.float32))

    def imsave_g(self, name_pattern='mlem_g_%03d.tif'):
        imageio.imsave(name_pattern % self.i, self.g.astype(np.float32))


def noisify(im, frac=0.1):
    # Add Poisson detector noise; Clip just above 0 to avoid scipy bug:
    # See https://github.com/scipy/scipy/issues/1923
    im.clip(im.max() / 1e6, out=im)
    i_max = im.max()
    sigma = frac * sp.ndimage.standard_deviation(im) / i_max
    # rescale for noise addition, create noisy version and rescale
    im /= i_max * (sigma ** 2)
    im = stats.poisson.rvs(im) * i_max * (sigma ** 2)
    return im


def density_from_attenuation(p, attenuation_map):
    """
    Take an attenuation map [cm^-1] and recast it as a matrix map with density
    units [g/cm3] by dividing by the mass attenuation coefficient of the matrix.

    Applying this to a dimensionless projected attenuation map sinogram []
    will convert it to areal density units [g/cm2]

    Parameters
    ----------
    p : Phantom2d object
    attenuation_map : 2d ndarray of float

    Returns
    -------
    2d ndarray of float with same dimensions as attenuation_map

    """
    d = data_helpers.MatrixProperties(p)
    density_map = attenuation_map / d.ma(p.energy)
    return density_map


def density_from_fluorescence_for_el(p, q, el):
    """
    Make a calibration of the ratio of fluorescence irradiance to areal
    density. For sinograms with units of quantity q per cm2, these
    ultimately need to be rescaled to units of [q]/cm2. To do this, this
    function, I project and sum at one angle to get the ratio of the
    integral of the resulting map with the original. This function's return
    value should be multiplied by a calculated emission map (in corresponding
    irradiance units) to convert the map to projected density.

    Parameters
    ----------
    p : Phantom2d object
    q : int
        Maia detector pad id
    el : string
        element, e.g. 'Ni', 'Fe', 'Zn'

    Returns
    -------
    float
        conversion factor [g/irradiance_unit]

    """
    # override absorption settings temporarily to get the response with abs off
    conf_o = config.no_out_absorption
    conf_i = config.no_in_absorption
    config.no_out_absorption = True
    config.no_in_absorption = True
    sinogram = projection.project_sinogram(event_type='fluoro', p=p, q=q,
                                           anglelist=[0], el=el)
    # restore overridden absorption settings
    config.no_out_absorption = conf_o
    config.no_in_absorption = conf_i

    # Rescale sinogram pixel quantities based on pixel side length.
    # Note: I don't need to do this, assuming the length scale is not changing
    # sinogram *= (UM_PER_CM / p.um_per_px) ** 2

    # Now just integrate the density [g/cm3] in the elemental map.
    mass = p.el_maps[el]

    return mass.sum() / sinogram.sum()


def density_from_attenuation_for_matrix(p):
    conf_o = config.no_out_absorption
    conf_i = config.no_in_absorption
    config.no_out_absorption = True
    config.no_in_absorption = True
    sinogram = projection.absorption_sinogram(p=p, anglelist=[0])
    # restore overridden absorption settings
    config.no_out_absorption = conf_o
    config.no_in_absorption = conf_i

    # Rescale sinogram pixel quantities based on pixel side length.
    # Note: I don't need to do this, assuming the length scale is not changing
    # sinogram *= (UM_PER_CM / p.um_per_px) ** 2

    # Now just integrate the density [g/cm3] in the elemental map.
    mass = p.el_maps['matrix']

    return mass.sum() / sinogram.sum()


if __name__ == '__main__':

    import phantom
    from maia import Maia
    from skimage.transform import iradon, iradon_sart
    import os

    '''
    def projector(x, angles=None):
        """
        skimage's radon transform

        """
        # x = filters.gaussian_filter(x, 1)
        x[circular_mask] = 0    # Must be identically zero to use the
                                # circle=True skimage radon kwarg
        y = radon(x, theta=angles, circle=True)
        y *= conversion_factor_for_el
        return y
    '''

    def projector_model(x, angles=None):
        el = mlem.el
        p = mlem.p
        x[circular_mask] = 0    # Must be identically zero to use the
                                # circle=True skimage radon kwarg
        p.el_maps[el] = x
        y = projection.project_sinogram('fluoro', p, q, angles, el)
        y *= conversion_factor_for_el

        return y

    def backprojector(x, angles=None):
        """
        skimage's fbp-based inverse radon transform

        """
        x /= conversion_factor_for_el

        # TODO: I think I need to do this because I suspect the conversion
        # TODO: factor removes the length unit.
        # Rescale sinogram pixel quantities based on pixel side length.
        # The sinogram is a map of some quantity q per pixel,
        # which needs to be rescaled to units of [q]/cm.
        x *= UM_PER_CM/p.um_per_px

        # On the next line, the filter=None selection is *very* important
        # as using the ramp filter causes the scheme to diverge!
        if config.backprojector == 'fbp':
            y = iradon(x, theta=angles, circle=True, filter=None)
        elif config.backprojector == 'sart_no_prior':
            y = iradon_sart(x, theta=angles)
        y[circular_mask] = 0
        return y

    PATH_HERE = os.path.abspath(os.path.dirname(__file__))
    ELEMENT = 'Ni'
    MAP_PATTERN = os.path.join(PATH_HERE, 'data', 'Ni_test_phantom2-*.tiff')
    YAMLFILE = os.path.join(PATH_HERE, 'data', 'Ni_test_phantom2.yaml')
    UM_PER_CM = 1e4
    # WIDTH_UM = 100.0
    WIDTH_UM = 2000.0
    WIDTH_PX = 100
    UM_PER_PX = WIDTH_UM / WIDTH_PX
    ENERGY_KEV = 15.6

    config.parse()  # read config file settings

    p = phantom.Phantom2d(
        filename=MAP_PATTERN,
        yamlfile=YAMLFILE,
        um_per_px=UM_PER_PX,
        energy=ENERGY_KEV,
    )
    # maia_d = Maia()
    maia_d = projection.maia_d
    q = maia_d.channel(7, 7).index[0]
    # angles = np.linspace(0, 180, np.pi * p.shape[0], endpoint=False)
    angles = np.linspace(0, 360, np.pi * p.rows, endpoint=False)

    # If/when we have an experimentally acquired absorption sinogram, we need
    # to obtain the matrix map from it, which requires a conversion factor,
    # by which to multiply the sinogram:
    conversion_factor_for_matrix = density_from_attenuation_for_matrix(p)
    # In this example, however, we already have the matrix map, so just use
    # that directly.

    # We do, however, need to convert fluorescence to equivalent mass for this
    # case:
    conversion_factor_for_el = density_from_fluorescence_for_el(p, q,
                                                                el=ELEMENT)

    # Create the "input data" i.e. the simulated-experimentally-acquired
    # sinograms
    im = projection.absorption_sinogram(p, angles)  # absorption (CT) sinogram
    # Create a container object for the input data
    sinograms = phantom.Phantom2d(
        shape=im.shape,
        um_per_px=UM_PER_PX,
        energy=ENERGY_KEV,
    )
    # Substitute the absorption (CT) sinogram for the matrix.
    sinograms.insert_map('matrix', im)
    # Now do all non-matrix elements.
    for el in p.el_maps:
        if el == 'matrix':
            continue
        im = projection.project_sinogram('fluoro', p, q, angles, el)
        im *= density_from_fluorescence_for_el(p, q, el)

        sinograms.insert_map(el, im)

    #########################
    ## OK, we're set up now

    # Make a circular mask to allow use of skimage's radon transform with
    # circle=True
    s = (p.el_maps['matrix'].shape[1] - 1) / 2.
    circular_mask = np.hypot(*np.ogrid[-s:s + 1, -s:s + 1]) > s - 0.5

    g_j = sinograms.el_maps[ELEMENT]
    mlem = Mlem(p, ELEMENT, projector_model, backprojector, g_j, angles=angles)
    # mlem = Mlem(projector, backprojector, g_j, angles=angles)

    # mlem.imsave_f()
    for im in range(50):
        mlem.iterate()
    # mlem.imsave_f()
