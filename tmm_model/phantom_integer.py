#!/usr/bin/env python

# Copyright (c) 2014, Gary Ruben
# Released under the Modified BSD license
# See LICENSE


"""Phantom class
Implements a class for 2D phantoms. The Golosio phantom is coded in terms of
geometry and composition data, with class methods supplied to instanciate it.
Alternatively, phantoms can be defined as a pair of files, one containing
geometry data as a greyscale bitmap and the other containing the composition
data structure defined in a yaml file.

"""

import os
import numpy as np
from skimage.io import imread, imsave
from skimage import img_as_ubyte
from skimage.transform import rotate
import matplotlib.pyplot as plt
from collections import Iterable
import yaml

# suppress spurious skimage warning
import warnings

#-------------------------------------------------------------------------------
# Phantoms
#-------------------------------------------------------------------------------
"""Phantom from Fig. 3 of Golosio et al. [1]

[1] Golosio et al., Internal Elemental Microanalysis Combining
X-Ray Fluorescence, Compton and Transmission Tomography, J. App. Phys. 94 (2003)

"""
# Composition
golosio_compounds = {
    # Compound: [Density (g cm^-3), [[Z1, Concentration1 (weight frac)],
    #                                [Z2, Concentration2 (weight frac)], ... ]]
    0: [1.2e-3, [['N', 0.78], [ 'O', 0.21], ['Ar', 0.01]]], # air
    1: [2.0,    [['C', 0.3 ], [ 'O', 0.6 ], ['Si', 0.1 ]]],
    2: [3.5,    [['O', 0.3 ], ['Si', 0.1 ], ['Ca', 0.4 ], ['Fe', 0.2]]],
    3: [3.5,    [['O', 0.3 ], ['Si', 0.1 ], ['Ca', 0.3 ], ['Fe', 0.3]]],
}

# Geometry
golosio_geometry = [
    # [shape, compound, topleft, bottomright]
    ['square',  0, (0.0, 0.0), (1.0, 1.0)],         # air
    ['square',  1, (0.2, 0.2), (0.8, 0.8)],         # substrate

    ['square',  2, (0.26, 0.26), (0.44, 0.44)],     # upper left

    ['circle',  2, (0.26, 0.56), (0.44, 0.74)],     # upper right

    ['circle',  2, (0.56, 0.26), (0.74, 0.44)],     # lower left
    ['diamond', 1, (0.58, 0.28), (0.72, 0.42)],

    ['square',  2, (0.56, 0.56), (0.74, 0.74)],     # lower right
    ['circle',  3, (0.59, 0.59), (0.71, 0.71)],
]
#-------------------------------------------------------------------------------


class Phantom2d(object):
    """
    Internally, the phantom is a 2d array of integer compound ids that define
    the composition at each pixel.

    """
    def __init__(self, size=None, scale=1.0, filename=None):
        """
        Arguments:
        size - (rows,cols) in px
        scale - units_per_px, e.g. 0.1 defines a scale of 0.1 units per px
        filename - if not None, passed to self.read_map() to create the phantom
            from a greyscale image map filename.png and compound data
            filename.yaml

        """
        assert size is not None or filename is not None
        if filename is not None:
            self.phantom_array_0deg = self._read_map(filename)
            self.compounds = self._read_composition(filename)
        else:
            self.rows, self.cols = size
            assert type(self.rows) is int and type(self.cols) is int
            self.scale = float(scale)
            self.phantom_array_0deg = np.zeros(size, dtype=int)

        self.filename = filename
        self.phantom_array = self.phantom_array_0deg.copy()


    def coordinate_scale(self, val):
        """Rescale a coordinate value from "world" coords (0.0-1.0)
        to pixel coordinates.

        Arguments:
        val - coordinate value (0.0-1.0)

        Returns:
        rescaled value

        """
        if self.scale is None:
            result = val
        else:
            if isinstance(val, Iterable):
                container_type = type(val)
                result = container_type(v/self.scale for v in val)
            else:
                result = val / self.scale
        return result


    def rotate(self, angle):
        """Rotate the phantom_array

        Arguments:
        angle - angle in degrees (ccw) to rotate the phantom_array

        """
        # scikit-image generates the rotated output array as a float array
        # which must be scaled back to uint8 using the img_as_ubyte function.
        # Also, it raises a warning, which needs to be ignored - sheesh!

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # order=0 forces nearest-neighbour interpolation.
            self.phantom_array = img_as_ubyte(
                rotate(self.phantom_array_0deg, angle, order=0))
       

    def add_shape(self, thing):
        """Add a 'thing'; a square, circular or diamond-shaped patch of some
        integer, which refers to a compound. This is used as a lookup index into
        a table of compound compositions.

        """
        shape, compound, topleft, bottomright = thing
        topleft = self.coordinate_scale(topleft)
        bottomright = self.coordinate_scale(bottomright)

        tlrow, tlcol = topleft
        brrow, brcol = bottomright
        cr, cc = np.array([topleft, bottomright]).mean(axis=0)
        radius = (brrow - tlrow) / 2.0
        tlrow, tlcol, brrow, brcol = \
            [int(i) for i in (tlrow, tlcol, brrow, brcol)]

        # After writing this, I discovered skimage has a drawing module, which
        # would probably produce better results
        if shape == 'square':
            self.phantom_array_0deg[tlrow:brrow+1, tlcol:brcol+1] = compound

        if shape == 'circle':
            assert type(topleft) is tuple and type(bottomright) is tuple
            p = np.hypot(*np.ogrid[-cr:self.rows-cr, -cc:self.cols-cc])
            self.phantom_array_0deg[p <= radius] = compound

        if shape == 'diamond':
            intrad = int(radius)
            diamond = np.tri(intrad*2, intrad*2, intrad, dtype=int)
            diamond[intrad:] = diamond[:intrad][::-1]
            diamond[:,:intrad] = diamond[:,intrad:][:,::-1]

            diamond_mask = np.zeros_like(self.phantom_array, dtype=bool)
            diamond_mask[tlrow+2:tlrow+intrad*2+2,
                         tlcol+1:tlcol+intrad*2+1] = diamond

            self.phantom_array_0deg[diamond_mask] = compound

        self.phantom_array = self.phantom_array_0deg


    def _read_map(self, filename):
        """Read an image map (a greyscale png) whose pixel values
           correspond to keys in a lookup table of compounds.

        Arguments:
        filename - string or file handle filename.png

        Returns:
        integer array

        """
        im = imread(filename+'.png', as_grey=True)
        self.rows, self.cols = im.shape
        self.scale = 1.0
        return im


    @staticmethod
    def _read_composition(filename):
        """Read phantom compound composition data from a yaml file.

        Arguments:
        filename - string or file handle corresponding to a file
                   filename[_ext].yaml
                   If the filename contains the optional _ext portion, this will
                   be ignored. This enables a composition map to apply to many
                   geometry maps.

        Returns:
        composition data - see golosio_compounds above for example

        """
        pathname, basename = os.path.split(filename)
        filename = os.path.join(pathname, basename.split('_')[0])
        with open(filename+'.yaml') as f:
            comp = yaml.load(f)
        return comp


    def split_map(self, dirname):
        """Split the single greyscale lookup map + composition dictionary into
        a series of data files that mirror the files output by GeoPIXE. An
        example might be that a phantom defined as a file pair (golosio.png,
        golosio.yaml) will generate a series of files golosio-x.tiff, one for
        each x in [N, O, Ar, C, Si, Ca, Fe]
        
        Arguments:
        dirname - path to directory to contain elemental tiff maps

        """
        filename = os.path.basename(self.filename)

        # enumerate all elements in compounds
        materials = [i[1] for i in self.compounds.values()]
        elements = set([el for i in materials for el in i])

        # generate a map for each element
        maps = {}
        for el in elements:
            maps[el] = np.zeros(self.phantom_array_0deg.shape, dtype=np.float32)

        for compound in self.compounds:
            density, weights = self.compounds[compound]
            for el in weights:
                m = maps[el]
                m[self.phantom_array==compound] = density * weights[el]

        # write them out
        for el in elements:
            outfile = os.path.join(dirname, '{}-{}.tiff'.format(filename, el))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # save a 32-bit tiff using either freeimage or tifffile
                try:
                    imsave(outfile, maps[el], plugin='freeimage')   # 32-bit
                except (ValueError, RuntimeError):
                    imsave(outfile, maps[el], plugin='tifffile')    # 32-bit

        '''
        for el in elements:
            plt.matshow(maps[el])
            plt.colorbar()
            plt.title(el)
        plt.show()
        '''


    def compound_record(self, compounds, row, col):
        """Return the compound record for the specified row and col.

        Arguments:
        compounds - The lookup structure containing the compound information.
                 See e.g. golosio_compounds for the dictionary format.
        row, col - Array indices of the pixel in the phantom.

        Returns:
        compound record for that pixel value.

        """
        assert row < self.rows and col < self.cols
        ix = self.phantom_array[row, col]
        return compounds[ix]


    def density(self, compounds, row, col):
        """Return the density at the specified row and col.

        Arguments:
        compounds - The lookup structure containing the compound information.
                 See e.g. golosio_compounds for the dictionary format.
        row, col - Array indices of the pixel in the phantom

        Returns:
        density (g cm^-3)

        """
        return self.compound_record(compounds, row, col)[0]


    def show(self, show=False):
        """Display the map using matplotlib.

        """
        plt.imshow(self.phantom_array, origin='upper',
                   interpolation='nearest', cmap='gray')
        plt.colorbar()
        if show:
            plt.show()


if __name__ == '__main__':
    GOLOSIO_MAP = os.path.join('data', 'golosio_100')
    p = Phantom2d(filename=GOLOSIO_MAP)
    p.split_map('data')
