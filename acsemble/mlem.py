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

if config.mlem_profile:
    from profilehooks import profile
else:
    profile = lambda x: x

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
        self.weighting = backprojector(np.ones_like(g_j), angles)   # units of weighting [???]
        # The backprojector forces 0's outside the inscribed circle after backprojection.
        # This will cause divide-by-zero errors if the numerator has zeros in this region.
        # Avoid this by clipping everything to +epsilon.
        assert self.f.shape == self.weighting.shape
        self.weighting.clip(self.epsilon, out=self.weighting)

    @profile
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
    sinogram = projection.absorption_sinogram(p=p, anglelist=[0])
    # Rescale sinogram pixel quantities based on pixel side length.
    sinogram *= UM_PER_CM / p.um_per_px

    # Now just integrate the density [g/cm3] in the elemental map.
    mass = p.el_maps['matrix']

    return mass.sum() / sinogram.sum()


@click.command()
@click.option('--config-path', '-c', default=None, help='Path to config file')
def main():
    from . import phantom
    from . import iradons
    import os
    import glob

    def projector_model(x, angles=None):
        """A function that computes and returns the fluorescence sinogram of the element
        self.el to which the calling mlem object corresponds. This function is attached to
        the mlem object during its construction.

        Parameters
        ----------
        x: Density map of the element of interest with units [g/cm3].
        angles: 1d array/vector of projection angles [degree].

        Returns
        -------
        sinogram: 2d ndarray of floats [g/cm2]

        """
        x = np.copy(x)
        el = mlem.el
        p = mlem.p
        x[circular_mask] = 0    # Must be identically zero to use the
                                # circle=True skimage radon kwarg
        p.el_maps[el] = x
        y = projection.project_sinogram('fluoro', p, q, angles, el)
        y *= conversion_factor_for_el

        return y

    def backprojector(x, angles=None, **kwargs):
        """Wraps an inverse-radon-transform direct-backprojector; basically fbp without a
        filter. This takes a fluorescence sinogram and returns the direct-backprojected
        model estimate.

        Parameters
        ----------
        x : sinogram: 2d ndarray of floats [g/cm2]
        angles : 1d array/vector of projection angles [degree].
        kwargs : dictionary
            If

        Returns
        -------
        2d ndarray of floats representing a density map of the element of interest with
        units [g/cm3].

        """
        if kwargs:
            filter = kwargs['filter']
        else:
            filter = config.xlict_recon_mpi_fbp_filter
        x = np.copy(x)
        im_x = x / conversion_factor_for_el

        # Rescale sinogram pixel quantities based on pixel side length.
        # The sinogram is a map of some quantity q per pixel,
        # which needs to be rescaled to units of [q]/cm.
        im_x *= UM_PER_CM/p.um_per_px

        y = iradons.iradon(im_x, angles, filter=filter)
        y[circular_mask] = 0
        return y

    # Some paths and constants
    PATH_HERE = os.path.abspath(os.path.dirname(__file__))
    UM_PER_CM = 1e4
    yamlfile = config.map_elements
    map_width_um = config.map_width_um
    filenames = helpers.match_pattern(config.map_pattern, glob.glob(config.map_pattern))

    # Read the matrix sinogram. Reconstruct it because we need to use the attenuation map
    # in our forward model.
    matrix_im_filename = [i[0] for i in filenames if i[1]=='matrix'][0]
    matrix_im = helpers.read_tiff32(matrix_im_filename)
    matrix_im = matrix_im.T

    # Set endpoint=False because MdJ trimmed the 360 deg projection off the sinogram.
    matrix_angles = np.linspace(0, 360, matrix_im.shape[1], endpoint=False)
    im_recon = iradons.iradon_absorption(matrix_im, matrix_angles)
    # Write it out for posterity
    filename = os.path.join(config.mlem_im_path, 'matrix_xtract_recon.tif')
    helpers.write_tiff32(filename, im_recon)

    # Downsample the matrix map to the same shape as the element map. To do that we need to
    # read the el map, just to get its dimensions.
    el_im_filename = [i[0] for i in filenames if i[1]==config.element][0]
    el_im = helpers.read_tiff32(el_im_filename)
    width_px = el_im.shape[1]
    um_per_px = map_width_um / width_px

    if config.matrix_blur:
        im_recon_length_scale_factor = float(matrix_im.shape[0]) / width_px
        # Note: I verified that sn.filters.gaussian_filter doesn't renormalise the data;
        # the sum and mean are unchanged by the operation.
        im_recon = sn.filters.gaussian_filter(im_recon, sigma=4/im_recon_length_scale_factor)
    im_recon.clip(min=0.0, out=im_recon)

    # Now downsample the matrix to the same shape as the element map.
    im_recon = st.resize(im_recon, (width_px, width_px), preserve_range=True)
    # Write it out for posterity/diagnostic-purposes.
    filename = os.path.join(config.mlem_im_path, 'matrix_recon.tif')
    helpers.write_tiff32(filename, im_recon)

    # Read the experimental input data
    # We read the matrix sinogram here, but it isn't used in any test of convergence by the
    # mlem iteration. Instead, the reconstructed, downsampled map is fixed in the forward
    # model estimate.
    sinograms = phantom.Phantom2d(
        filename=config.map_pattern,
        yamlfile=yamlfile,
        um_per_px=um_per_px,
        energy=config.energy_keV,
    )
    # Maia sinograms can have -ve values but mlem doesn't allow that, so just clip the data.
    # This is biasing the data, but only by a little bit.
    sinograms.rescale_element(config.element)
    sinograms.clip_positive()
    for el, el_map in sinograms.el_maps.iteritems():
        assert np.all(el_map >= 0.0)

    assert type(config.angles_compute) == bool
    if config.angles_compute:
        assert type(config.angles_closed_range) == bool
        angles = np.linspace(0, config.angles_max, sinograms.cols,
                             endpoint=config.angles_closed_range)
    else:
        raise Exception('no angles')

    # Create an initial model estimate
    model_shape = (sinograms.rows, sinograms.rows)
    p = phantom.Phantom2d(
        yamlfile=yamlfile,
        um_per_px=um_per_px,
        energy=config.energy_keV,
        shape=model_shape,
    )
    # If/when we have an experimentally acquired absorption sinogram, we need
    # to obtain the matrix map from it, which requires a conversion factor,
    # by which to multiply the sinogram:
    p.matrix = helpers.MatrixProperties(p)

    # For now, I will smooth the ricegrain matrix and rescale it according to the
    # proportionality constant k = config.density_per_compton_count [g cm^-3 count^-1]
    # Units:
    # im_recon: count / unscaled_voxel
    # k = config.density_per_compton_count: g cm^-3 count^-1
    # rescaled im_recon: g cm^-3 / rescaled_voxel
    im_recon *= config.density_per_compton_count
    filename = os.path.join(config.mlem_im_path, 'matrix_recon_g_per_cm3.tif')
    helpers.write_tiff32(filename, im_recon)

    p.insert_map('matrix', im_recon)
    p.insert_map(config.element, np.ones(model_shape))

    # Now create a detector instance
    maia_d = projection.maia_d
    q = config.detector_pads[0]

    # We need to convert the detected signal from the fluorescence model to an equivalent
    # density so that units are consistent around the iterative loop.
    config.i = 0        # Kludge; a global variable
    conversion_factor_for_el = density_from_fluorescence_for_el(p, q, el=config.element)

    #########################
    ## OK, we're set up now

    # Make a circular mask to allow use of skimage's radon transform with circle=True
    s = (p.el_maps['matrix'].shape[1] - 1) / 2.
    circular_mask = np.hypot(*np.ogrid[-s:s + 1, -s:s + 1]) > s - 0.5

    g_j = sinograms.el_maps[config.element]
    mlem = Mlem(p, config.element, projector_model, backprojector, g_j, angles=angles)

    # mlem requires that all values are positive. Verify that before launching.
    for el, el_map in mlem.p.el_maps.iteritems():
        assert np.all(el_map >= 0.0)
    # sinograms.clip_positive()
    # for el, el_map in sinograms.el_maps.iteritems():
    #     assert np.all(el_map >= 0.0)

    logger.info('i\tf_sum\tf_min\tf_max\tr_sum\tg_sum\tg_j_sum')

    im = backprojector(g_j, angles=angles, filter=config.xlict_recon_reference_fbp_filter)
    filename = os.path.join(
                    config.mlem_im_path,
                    '{}_fbp_{}_recon.tif'.format(
                        config.element,
                        config.xlict_recon_reference_fbp_filter
                    )
                )
    helpers.write_tiff32(filename, im)

    im = helpers.read_tiff32(config.mlem_reference_image_path)
    im = im[::-1]   # temporarily flip the image up-down until I fix this inversion bug.
    mlem.set_reference_image(im)

    for im in range(101):
        config.i = im
        mlem.iterate()


if __name__ == '__main__':
    main()
