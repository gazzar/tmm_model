#!/usr/bin/python3

"""mlem algorithm
Implements the iterative mlem algorithm, performing projection and
backprojection in a loop.

My advice for learning about MLEM is to look at two books:
[1] G. L. Zeng, Medical image reconstruction: A Conceptual Tutorial.
                Springer, 2010.
[2] M. N. Wernick and J. N. Aarsvold, Emission Tomography: The Fundamentals of
                                      PET and SPECT. Academic Press, 2004.

"""

from __future__ import absolute_import, division, print_function
import six
from . import config
import logging
logger = logging.getLogger(__name__)

import sys, os

# Set environ so that matplotlib uses v2 interface to Qt, because we are
# using mayavi's mlab interface in maia.py
os.environ.update(
    {'QT_API': 'pyqt', 'ETS_TOOLKIT': 'qt4'}
)

import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
import skimage.transform as st
import skimage.filters as sf
import scipy.ndimage as sn
from . import projection
from . import data_helpers
from . import helpers
import click

UM_PER_CM = 1e4

# if config.mlem_profile:
#     from profilehooks import profile
# else:
#     profile = lambda x: x


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
        self.r_sum = 0.0
        self.r_median = 0.0
        self.el = el
        self.project = projector
        self.backproject = backprojector
        self.angles = angles
        self.g_j = g_j
        self.epsilon = np.mean(np.abs(g_j)) / 1000.0
        self.sum_g_j = g_j.sum()
        self.i = 0
        # As per conventional MLEM, set initial self.f estimate to all ones to avoid bias.
        self.f = np.ones((g_j.shape[0], g_j.shape[0]))           # units of self.f [g/cm3]
        self.weighting = backprojector(np.ones_like(g_j), angles) # units of weighting [???]
        # The backprojector forces 0's outside the inscribed circle after backprojection.
        # This will cause divide-by-zero errors if the numerator has zeros in this region.
        # Avoid this by clipping everything to +epsilon.
        assert self.f.shape == self.weighting.shape
        self.weighting.clip(self.epsilon, out=self.weighting)

    # @profile
    def iterate(self):
        # From Lalush and Wernick (See [2]);
        # f^\hat <- (f^\hat / |\sum h|) * \sum h * (g_j / g)          ... (*)
        # where g = \sum (h f^\hat)                                   ... (**)
        #
        # self.f is the current estimate f^\hat
        # H == h_{mj} is called the system matrix;
        #   - projects from image space to projection space.
        #   - represents the probability that a photon emitted at voxel j is detected in
        #     bin m (See Glick and Soares for justification of unmatched H and H').
        # The following g from (**) is equivalent to g = \sum (h f^\hat)

        # Note: The projector internally forces 0's outside the inscribed circle before
        # projecting.
        g = self.project(self.f, angles=self.angles)    # units of self.f [g/cm3], g [g/cm2]
        g.clip(min=self.epsilon, out=g)

        # save diagnostics
        self.g = g      # Save this for logging
        self.log_stats()
        if config.save_g_images:
            self.imsave_g()
        if config.save_d_images:
            im = self.backproject(g, angles=self.angles)
            self.imsave_d(im)

        # form parenthesised term (g_j / g) from (*)
        r = self.g_j / g                                # units of r [dimensionless]
        self.r_sum = r.sum()            # Store for stats output
        self.r_median = np.median(r)    # Store for stats output

        # backproject to form \sum h * (g_j / g)
        # Note: The backprojector internally forces 0's outside the inscribed circle after
        # backprojection. That's OK here because it's on the numerator so won't cause
        # divide-by-zero.
        g_r = self.backproject(r, angles=self.angles)   # units of g_r [???]

        # Renormalise backprojected term / \sum h)
        # Normalise the individual pixels in the reconstruction
        self.f *= g_r / self.weighting

        if config.save_f_images:
            self.imsave_f()

        if config.mlem_save_similarity_metrics:
            self._save_similarity_metrics()

        self.i += 1

    def _save_similarity_metrics(self):
        im = self.f
        im_ref = self.reference_image
        mse = helpers.mse(im, im_ref)
        helpers.append_to_running_log(filename=config.mlem_mse_path,
                                      text='{:04}\t{}\n'.format(self.i, mse))
        mssim = helpers.mssim(im, im_ref)
        helpers.append_to_running_log(filename=config.mlem_mssim_path,
                                      text='{:04}\t{}\n'.format(self.i, mssim))

    def _imshow(self, im, show=True):
        plt.figure()
        plt.imshow(im, interpolation='nearest', cmap='YlGnBu_r')
        plt.title('iteration:%d' % self.i)
        plt.colorbar()
        if show:
            plt.show()

    def log_stats(self):
        logger.info(
            '{i}\t{f_sum}\t{f_min}\t{f_max}\t{r_sum}\t{r_median}\t{g_sum}\t{g_j_sum}'.format(
                i=self.i,
                f_sum=self.f.sum(),
                f_min=self.f.min(),
                f_max=self.f.max(),
                r_sum=self.r_sum,
                r_median=self.r_median,
                g_sum=self.g.sum(),
                g_j_sum=self.sum_g_j,
            )
        )

    def imsave_dfg(self, pattern, im):
        filename = os.path.join(config.mlem_im_path, pattern % self.i)
        helpers.write_tiff32(filename, im)

    def imsave_d(self, im):
        self.imsave_dfg('mlem_d_%03d.tif', im)

    def imsave_f(self):
        self.imsave_dfg('mlem_f_%03d.tif', self.f)

    def imsave_g(self):
        self.imsave_dfg('mlem_g_%03d.tif', self.g)

    def set_reference_image(self, im):
        self.reference_image = im


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


def density_from_fluorescence_for_el(p, q, maia_d, el):
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
    maia_d : Maia() instance
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
                                        maia_d=maia_d, anglelist=[0], el=el)
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
    sinogram = projection.absorption_sinogram(p=p, anglelist=[0])
    # Rescale sinogram pixel quantities based on pixel side length.
    sinogram *= UM_PER_CM / p.um_per_px

    # Now just integrate the density [g/cm3] in the elemental map.
    mass = p.el_maps['matrix']

    return mass.sum() / sinogram.sum()
