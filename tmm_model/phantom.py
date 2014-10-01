#!/usr/bin/env python

# Copyright (c) 2014, Gary Ruben
# Released under the Modified BSD license
# See LICENSE


"""Phantom class
Implements a class for 2D phantoms. The Golosio phantom is coded in terms of
geometry and composition data, with class methods supplied to instantiate it.
Alternatively, phantoms can be defined as a pair of files, one containing
geometry data as a greyscale bitmap and the other containing the composition
data structure defined in a yaml file.

"""

from __future__ import print_function
import os
import numpy as np
from imageio import imread
from skimage import img_as_ubyte
from skimage.transform import rotate
import matplotlib.pyplot as plt
from collections import Iterable
from helpers import write_tiff32, read_tiff32
import yaml
import glob

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
    Internally, the phantom is a dict of 2d float32 arrays keyed by compound
    element name. This dict can be formed by reading a series of named tiffs
    or by reading a 2d integer index array (a .png file) and a matching dict of
    compound ids that define the composition at each pixel. If the latter, the
    dict of float32 arrays must be created locally.

    """
    def __init__(self, size=None, um_per_px=1.0, energy=15, filename=None,
                  matrix_elements=''):
        """
        Arguments:
        size - (rows,cols) in px
        um_per_px - um_per_px, e.g. 0.1 defines a scale of 0.1 um/px
        energy - incident energy (keV)
        filename - if not None, passed to self.read_map() to create the phantom
            from a greyscale image map filename.png and compound data
            filename.yaml
        matrix_elements - space-separated string of elements in the matrix.
            default = empty-string ""

        """
        assert size is not None or filename is not None
        self.el_maps = {}       # container for elemental maps
        self.matrix_elements = matrix_elements
        self.filename = filename
        self.energy = energy
        self.um_per_px = float(um_per_px)
        if filename is not None:
            basedir, basename = os.path.split(filename)
            basename_noext, ext = os.path.splitext(basename)
            assert ext in ['.png', '.tif', '.tiff']
            if ext == '.png':
                # If filename has a png extension, first read the .png map and
                # .yaml composition then write the per-element tiffs
                self.phantom_array = self._read_map(filename)
                self.compounds = self._read_composition(filename)
                self.split_map(basedir)
                filename = os.path.join(basedir, basename_noext+'*.tiff')
            else:
                self.phantom_array = np.zeros(size, dtype=int)
            # read tiffs
            self._read_tiffs(filename)
        else:
            self.rows, self.cols = size
            assert type(self.rows) is int and type(self.cols) is int
            self.phantom_array = np.zeros(size, dtype=int)


    def coordinate_scale(self, val):
        """Rescale a coordinate value from "world" coords (0.0-1.0)
        to pixel coordinates.

        Arguments:
        val - coordinate value (0.0-1.0)

        Returns:
        rescaled value

        """
        if self.um_per_px is None:
            result = val
        else:
            if isinstance(val, Iterable):
                container_type = type(val)
                result = container_type(v/self.um_per_px for v in val)
            else:
                result = val / self.um_per_px
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
            ub = img_as_ubyte(rotate(self.phantom_array, angle, order=0))
        return ub


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
            self.phantom_array[tlrow:brrow+1, tlcol:brcol+1] = compound

        if shape == 'circle':
            assert type(topleft) is tuple and type(bottomright) is tuple
            p = np.hypot(*np.ogrid[-cr:self.rows-cr, -cc:self.cols-cc])
            self.phantom_array[p <= radius] = compound

        if shape == 'diamond':
            intrad = int(radius)
            diamond = np.tri(intrad*2, intrad*2, intrad, dtype=int)
            diamond[intrad:] = diamond[:intrad][::-1]
            diamond[:,:intrad] = diamond[:,intrad:][:,::-1]

            diamond_mask = np.zeros_like(self.phantom_array, dtype=bool)
            diamond_mask[tlrow+2:tlrow+intrad*2+2,
                         tlcol+1:tlcol+intrad*2+1] = diamond

            self.phantom_array[diamond_mask] = compound


    def _read_map(self, filename):
        """Read an image map (a greyscale png) whose pixel values
           correspond to keys in a lookup table of compounds.

        Parameters
        ----------
        filename : str or file handle
            The filename to read.

        Returns
        -------
        int array

        Raises
        ------
        IOError if there was a problem opening the file.

        """
        try:
            im = imread(filename)[:,:,0]
        except IOError:
            print('Could not open' + filename)
            raise
        # print(filename, im.max(), im.shape)
        self.rows, self.cols = im.shape
        self.um_per_px = 1.0
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


    def _read_tiffs(self, pattern):
        """Read all tiff files matching the glob pattern
        
        Arguments:
        pattern - a glob pattern for tiff files, e.g. '*.tiff'

        """
        _, basename = os.path.split(pattern)
        filenames = glob.glob(pattern)
        for f in filenames:
            assert('-' in f)
            basename, _ = os.path.splitext(f)
            el = basename.split('-')[-1]
            im = read_tiff32(f)
            self.rows, self.cols = im.shape

            self.el_maps[el] = im


    def split_map(self, dirname):
        """Split the single greyscale lookup map + composition dictionary into
        a series of data files that mirror the files output by GeoPIXE. An
        example might be that a phantom defined as a file pair (golosio.png,
        golosio.yaml) will generate a series of files golosio-x.tiff, one for
        each x in [N, O, Ar, C, Si, Ca, Fe].
        If self.matrix_elements is not the empty string, any elements it
        contains will be added to a single map called filename-matrix.tiff
        
        Arguments:
        dirname - path to directory to contain elemental tiff maps

        """
        filename = os.path.basename(self.filename)
        filename = os.path.splitext(filename)[0]

        # enumerate all elements in compounds
        materials = [i[1] for i in self.compounds.values()]
        elements = set([el for i in materials for el in i])

        # generate a map for each element
        maps = {}
        # for el in elements:
        non_matrix_elements = elements.difference(self.matrix_elements.split())
        for el in non_matrix_elements:
            maps[el] = np.zeros(self.phantom_array.shape, dtype=np.float32)

        if self.matrix_elements:
            maps['matrix'] = np.zeros(self.phantom_array.shape,
                                      dtype=np.float32)

        for compound in self.compounds:
            density, weights = self.compounds[compound]
            # Normalize weights so that they sum to 1
            sum_of_weights = sum(weights.values())
            weights = {k:v/sum_of_weights for k,v in weights.iteritems()}
            # Distribute elements to the individual maps or the matrix
            for el in weights:
                if el in self.matrix_elements.split():
                    # add to the matrix map
                    m = maps['matrix']
                else:
                    # add to the individual element map
                    m = maps[el]
                compound_mask = (self.phantom_array & compound).astype(bool)
                m[compound_mask] = density * weights[el]

        # Write tiffs
        for el in maps:
            outfile = os.path.join(dirname, '{}-{}.tiff'.format(filename, el))
            write_tiff32(outfile, maps[el])


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
    # MAP = os.path.join('data', 'golosio_100.png')
#     GOLOSIO_MAP = os.path.join('data', 'golosio_100*.tiff')
    MAP = r'R:\Science\XFM\GaryRuben\projects\TMM\work\data' \
                  r'\phantom1_100.png'
    p = Phantom2d(filename=MAP, matrix_elements='H C N O Na P S Cl K')
    p.split_map('data')
