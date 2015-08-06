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

import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
import imageio


class Mlem(object):
    def __init__(self, projector, backprojector, g_j, angles=None):
        """
        projector : function
            Function that takes an image and transforms it to
            projection space.
        backprojector : function
            Function that takes a sinogram and transforms it to
            image space.
        g_j : observed sinogram data.
        angles : ndarray of angles (in degrees) for projector and backprojector.

        """
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
        # From Lalush and Wernick;
        # f^\hat <- (f^\hat / |\sum h|) * \sum h * (g_j / g)          ... (*)
        # where g = \sum (h f^\hat)                                   ... (**)
        #
        # self.f is the current estimate f^\hat
        # The following g from (**) is equivalent to g = \sum (h f^\hat)
        g = self.project(self.f, angles=self.angles)

        g.clip(min=self.epsilon, out=g)

        # form parenthesised term (g_j / g) from (*)
        r = self.g_j / g

        # backproject to form \sum h * (g_j / g)
        g_r = self.backproject(r, angles=self.angles)

        # Renormalise backprojected term / \sum h)
        # Normalise the individual pixels in the reconstruction
        self.f *= g_r / self.weighting

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
        print('i={i} f_sum={f_sum} f_min={f_min} f_max={f_max} '
              'g_j_sum={g_j_sum}'.format(
                  i = self.i,
                  f_sum = self.f.sum(),
                  f_min = self.f.min(),
                  f_max = self.f.max(),
                  g_j_sum = self.sum_g_j,
                  )
              )

    def imsave_f(self):
        imageio.imsave('mlem_mpl_%03d.tif' % self.i, self.f.astype(np.float32))


def noisify(projectionData, frac=0.1):
    # Add Poisson detector noise; Clip just above 0 to avoid scipy bug:
    # See https://github.com/scipy/scipy/issues/1923
    projectionData.clip(projectionData.max()/1e6, out=projectionData)
    I_MAX = projectionData.max()
    sigma = frac * sp.ndimage.standard_deviation(projectionData) / I_MAX
    # rescale for noise addition, create noisy version and rescale
    projectionData = projectionData / (I_MAX * (sigma**2))
    projectionData = stats.poisson.rvs(projectionData) * I_MAX * (sigma**2)
    return projectionData


if __name__ == '__main__':

    import skimage
    from skimage.transform import radon, iradon, downscale_local_mean
    from skimage.util import pad
    from skimage import data, filters


    def projector(x, angles=None):
        '''
        skimage's radon transform

        '''
        # x = filters.gaussian_filter(x, 1)
        x[circular_mask] = 0    # Must be identically zero to use the
                                # circle=True skimage radon kwarg
        y = radon(x, theta=angles, circle=True)
        return y


    def backprojector(x, angles=None):
        '''
        skimage's fbp-based inverse radon transform

        '''
        # On the next line, the filter=None selection is *very* important
        # as using the ramp filter causes the scheme to diverge!
        y = iradon(x, theta=angles, circle=True, filter=None)
        y[circular_mask] = 0
        return y


    '''
    # Make a phantom that is 0 everywhere except for a small 3x3 block
    f0 = np.zeros((32, 32))
    f0[16:19, 16:19] = 1
    '''

    # '''
    # lena
    f0 = np.zeros((128, 128))
    b = 24
    f0[b:128-b, b:128-b] = data.lena()[178:338:2, 174:334:2].sum(axis=2)
    f0 = noisify(f0)
    # '''

    angles = np.linspace(0, 180, np.pi*f0.shape[0], endpoint=False)

    s = (f0.shape[1] - 1) / 2.
    circular_mask = np.hypot(*np.ogrid[-s:s+1, -s:s+1]) > s-0.5

    g_j = projector(f0, angles=angles)
    mlem = Mlem(projector, backprojector, g_j, angles=angles)

    mlem.imsave_f()
    for im in range(8):
        mlem.iterate()
    mlem.imsave_f()
