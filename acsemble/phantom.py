"""Phantom class
Implements a class for 2D phantoms. The Golosio phantom is coded in terms of
geometry and composition data, with class methods supplied to instantiate it.
Alternatively, phantoms can be defined as a pair of files, one containing
geometry data as a greyscale bitmap and the other containing the composition
data structure defined in a yaml file.

"""

from __future__ import print_function
import os
import config
import logging
import numpy as np
from imageio import imread
from skimage import img_as_ubyte
from skimage.transform import rotate
import matplotlib.pyplot as plt
from collections import Iterable
import helpers
from helpers import write_tiff32, read_tiff32
from data_helpers import MatrixProperties
import yaml
import glob
import textwrap

# suppress spurious skimage warning
import warnings

#-------------------------------------------------------------------------------
# Phantoms
#-------------------------------------------------------------------------------

class Phantom2d(object):
    """
    Internally, the phantom is a dict of 2d float32 arrays keyed by compound
    element name. This dict can be formed by reading a series of named tiffs
    or by reading a 2d integer index array (a .png file) and a matching dict of
    compound ids that define the composition at each pixel. If the latter, the
    dict of float32 arrays must be created locally.
    The values in the tiff density-maps have units of g/cm3. Together with the
    um_per_px value,

    """
    def __init__(self, shape=None, um_per_px=1.0, energy=15, filename=None,
                 yamlfile=''):
        """Constructor

        Parameters
        ----------
        shape : (rows,cols) in px
        um_per_px : um_per_px, e.g. 0.1 defines a scale of 0.1 um/px
        energy : incident energy (keV)
        filename : string
            if not None, passed to self.read_map() to create the phantom
            from a greyscale image map filename.png and compound data
            filename.yaml
        yamlfile : string
            used with filename; contains compound data, e.g. 'filename.yaml'

        """
        assert shape is not None or filename is not None
        # The el_maps field is a container for the elemental maps. One map is
        # special; the 'matrix' map represents the density map of the
        # matrix compound.
        # The key:value pairs are
        #     key: string containing element name, e.g. 'Zn'
        #     value: density map array (g/cm3)
        # Example:
        #   {'matrix':<nxm ndarray>, 'Zn':<nxm ndarray>, 'Fe':<nxm ndarray>}
        self.el_maps = {}       #
        #
                                # One of these has the key 'matrix'
        self.filename = filename
        self.yamlfile = yamlfile
        self.energy = energy
        self.um_per_px = float(um_per_px)
        if filename is not None:
            self.matrix = MatrixProperties(self)
            basedir, basename = os.path.split(filename)
            basename_noext, ext = os.path.splitext(basename)
            assert ext in ['.png', '.tif', '.tiff']
            if ext == '.png':
                # If filename has a png extension, first read the .png map and
                # .yaml composition then write the per-element tiffs
                self.phantom_array = self._read_map(filename)
                self.compounds = self._read_composition(self.yamlfile)
                self.split_map(basedir)
                filename = os.path.join(basedir, basename_noext+'-*.tiff')
            else:
                self.phantom_array = np.zeros(shape, dtype=int)

            # read tiffs
            self._read_tiffs(filename)
        else:
            # No filename was specified; just create a minimal Phantom2d
            # object to be populated by the instantiating code.
            self.rows, self.cols = shape
            assert isinstance(self.rows, (int, long)) and \
                   isinstance(self.cols, (int, long))
            self.phantom_array = np.zeros(shape, dtype=int)

    def __str__(self):
        """String representation of a Phantom2d object.

        Returns
        -------
        str
            The string representation.

        """
        rp_filename = 'None' if (self.filename is None) else self.filename
        rp_yamlfile = self.yamlfile if self.yamlfile else "''"
        repstr = '''
            Phantom2d
                id: {id}
                filename: {filename}
                el_maps (keys): {el_maps}
                yamlfile: {yamlfile}
                energy: {energy}
                um_per_px: {um_per_px}
                rows, cols: ({rows}, {cols})
                '''.format(
                    id = id(self),
                    el_maps = self.el_maps.keys(),
                    filename = rp_filename,
                    yamlfile = rp_yamlfile,
                    energy = self.energy,
                    um_per_px = self.um_per_px,
                    rows = self.rows, cols = self.cols,
                )
        return textwrap.dedent(repstr)

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
            diamond[:, :intrad] = diamond[:, intrad:][:, ::-1]

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
            im = imread(filename)
            if len(im.shape) == 3:
                # RGBA greyscale image; just use the 'R' layer
                im = im[:, :, 0]
            assert len(im.shape) == 2
        except IOError:
            print('Could not open' + filename)
            raise
        # print(filename, im.max(), im.shape)
        self.rows, self.cols = im.shape
        self.um_per_px = 1.0
        return im

    @staticmethod
    def _read_composition(yamlfile):
        """Read phantom compound composition data from a yaml file.

        Arguments:
        yamlfile - string or file handle corresponding to a file
                   filename.yaml

        Returns:
        composition data - see golosio_compounds above for example

        """
        with open(yamlfile) as f:
            comp = yaml.load(f)
        try:
            comp = comp['compound']
        except KeyError:
            pass
        return comp

    def _read_tiffs(self, pattern):
        """Read all tiff files matching the glob pattern.
        Our sinogram assumes the matlab convention, which
        is sino-x=angles, sino-y=x-projections.
        If this is not the case, the user can request this code transpose the sinograms on
        import by setting the mlem_transpose_on_read option in the config file.
        
        Parameters
        ----------
        pattern - a glob pattern for tiff files, e.g. '*.tiff'

        """
        filenames = helpers.match_pattern(pattern, glob.glob(pattern))

        for f, el in filenames:
            im = read_tiff32(f)
            if config.mlem_transpose_on_read:
                im = im.T
            self.rows, self.cols = im.shape
            self.el_maps[el] = im

    def insert_map(self, el, im):
        """Inserts the specified element (or compound) density map into el_maps.

        Parameters
        ----------
        el : str
            name to use for key
        im : 2d ndarray of float
            density map

        """
        rows, cols = im.shape
        assert(self.rows == rows and self.cols == cols)
        assert(el not in self.el_maps)
        self.el_maps[el] = im

    def delete_map(self, el):
        self.el_maps.pop(el)

    def split_map(self, dirname):
        """Split the single greyscale lookup map + composition dictionary into
        a series of data files that mirror the files output by GeoPIXE. An
        example might be that a phantom defined as a file pair (golosio.png,
        golosio.yaml) will generate a series of files golosio-x.tiff, one for
        each x in [N, O, Ar, C, Si, Ca, Fe].
        If self.matrix_elements is not the empty string, any elements it
        contains will be added to a single map called filename-matrix.tiff
        
        Parameters
        ----------
        dirname : path to directory to contain elemental tiff maps

        """
        filename = os.path.basename(self.filename)
        filename = os.path.splitext(filename)[0]

        # enumerate all elements in compounds
        materials = [i[1] for i in self.compounds.values()]
        elements = set([el for i in materials for el in i])

        # generate a map for each element
        maps = {}
        # for el in elements:
        non_matrix_elements = elements.difference(self.matrix.cp.keys())
        for el in non_matrix_elements:
            maps[el] = np.zeros(self.phantom_array.shape, dtype=np.float32)

        if self.matrix.cp:
            maps['matrix'] = np.zeros(self.phantom_array.shape,
                                      dtype=np.float32)

        for compound in self.compounds:
            density, weights = self.compounds[compound]
            # Normalize weights so that they sum to 1
            sum_of_weights = sum(weights.values())
            weights = {k:v/sum_of_weights for k,v in weights.iteritems()}

            # Distribute elements to the individual maps or the matrix
            for el in weights:
                if el in self.matrix.cp:
                    # add to the matrix map
                    m = maps['matrix']
                else:
                    # add to the individual element map
                    m = maps[el]
                compound_mask = (self.phantom_array & compound).astype(bool)
                # apply density, so tiff-maps will have units of g/cm3
                m[compound_mask] += density * weights[el]

        # Write tiffs
        for el in maps:
            outfile = os.path.join(dirname, '{}-{}.tiff'.format(filename, el))
            write_tiff32(outfile, maps[el])

    def compound_record(self, compounds, row, col):
        """Return the compound record for the specified row and col.

        Parameters
        ----------
        compounds : The lookup structure containing the compound information.
                 See e.g. golosio_compounds for the dictionary format.
        row, col : Array indices of the pixel in the phantom.

        Returns
        -------
        compound record for that pixel value.

        """
        assert row < self.rows and col < self.cols
        ix = self.phantom_array[row, col]
        return compounds[ix]

    def density(self, compounds, row, col):
        """Return the density at the specified row and col.

        Arguments:
        compounds : The lookup structure containing the compound information.
                 See e.g. golosio_compounds for the dictionary format.
        row, col : Array indices of the pixel in the phantom

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

    def clip_positive(self):
        """Rewrite all arrays, clipping them to only contain positive values.

        """
        for el, el_map in self.el_maps.iteritems():
            el_map.clip(min=0.0, out=el_map)

if __name__ == '__main__':
    # MAP = os.path.join('data', 'golosio_100.png')
#     GOLOSIO_MAP = os.path.join('data', 'golosio_100*.tiff')
    MAP = r'R:\Science\XFM\GaryRuben\projects\TMM\work\data' \
                  r'\phantom1_100.png'
    p = Phantom2d(filename=MAP)#, matrix_elements='H C N O Na P S Cl K')
    p.split_map('data')
