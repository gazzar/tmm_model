#!/usr/bin/env python

# Copyright (c) 2014, Gary Ruben
# Released under the Modified BSD license
# See LICENSE

import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


"""Maia detector class"""

PATH_HERE = os.path.abspath(os.path.dirname(__file__))
# sys.path = [os.path.join(PATH_HERE, '..')] + sys.path
MAIA_DATA = os.path.join(PATH_HERE, 'Maia_384A.csv')


class Maia(object):
    """Represents the detector geometry and provides visualisation routines.

    """
    def __init__(self):
        # Read Chris Ryan's detector data
        self.maia_data = pd.read_csv(MAIA_DATA, skipinitialspace=True,
                                      header=12)


    def rect_solid_angle(self, a, b, d):
        """Return the solid angle of a rectangle with one corner at the origin.

        Arguments:
        a - width of detector element
        b - height of detector element
        d - distance to plane of detector

        Returns:
        solid angle (sr)

        """
        alpha = a / (2.0 * d)
        beta = b / (2.0 * d)
        fact = np.sqrt((1+alpha**2+beta**2) / ((1+alpha**2)*(1+beta**2)))
        omega = 4.0 * (np.arccos(fact))

        return omega


    def get_omega(self, A, B, a, b, d):
        """Return the solid angle of a rectangular detector element of size
        width a * height b that does not lie across either the x=0 or y=0 axes
        and whose closest point to the origin lies at (x, y) = (A, B)

        From RJ Mathar, Solid Angle of a Rectangular Plate,
        Note 2 at http://www.mpia-hd.mpg.de/~mathar/public
        http://www.mpia-hd.mpg.de/~mathar/public/mathar20051002.pdf

        Arguments:
        A - x-coord of detector element corner closest to the origin
        B - y-coord of detector element corner closest to the origin
        a - width of detector element
        b - height of detector element
        d - distance to plane of detector

        Returns:
        solid angle (sr)

        """
        omega1 = self.rect_solid_angle(2.0*(A+a), 2.0*(B+b), d)
        omega2 = self.rect_solid_angle(2.0*A, 2.0*(B+b), d)
        omega3 = self.rect_solid_angle(2.0*(A+a), 2.0*B, d)
        omega4 = self.rect_solid_angle(2.0*A, 2.0*B, d)

        omega = (omega1 - omega2 - omega3 + omega4) / 4.0

        return omega

    # Create a vectorised version
    v_getOmega = np.vectorize(get_omega, excluded=['self', 'd'])


    def make_map(self, func, fill_value=0.0):
        """Returns a 20x20 map of the detector with the specified function
        populating the map.

        Arguments:
        func - A function evaluated for each self.maia_data row and column entry
            Examples:
            lambda : np.log(self.maia_data['width'])    # log width of element
            lambda : self.maia_data['width'] * self.maia_data['height'] # area
        fill_value - value to initialise the map to (default 0.0)

        Returns:
        20x20 numpy float32 array

        """
        map2d = np.zeros((20, 20)) + fill_value
        map2d[self.maia_data['Row'],
              self.maia_data['Column']] = func()
        return map2d


    def det_show(self, a, cmap='hot'):
        """Display a 20x20 detector map

        """
        plt.imshow(a, interpolation='nearest', origin='lower', cmap=cmap)
        plt.colorbar()


if __name__ == '__main__':
    from tests import maia_funcs

    det = Maia()    # Make a detector instance

    # Geometry
    d_mm = 10.0
    det_px_mm = 0.4

    # Location of CSV file that has the collimator info in it.
    dirPath = "."
    csvPath = \
        'mask-Mo-3x100-R-10-T-04-gap-75-grow-09-tune-103-safe-01-glue-025-1.csv'
    csvPath = os.path.join('tests', csvPath)

    # Get the solid angle distribution.
    md_sa_map, md_area_map = maia_funcs.getCollAreas(dirPath, csvPath, d_mm,
                                                     det_px_mm)

    # Calculate and append area and solid angle columns
    # then make some 2D maps for plotting and comparison

    a_mm = det.maia_data['width']
    b_mm = det.maia_data['height']
    A_mm = abs(det.maia_data['X']) - a_mm/2
    B_mm = abs(det.maia_data['Y']) - b_mm/2

    det.maia_data['Area mm2'] = a_mm * b_mm
    det.maia_data['Solid Angle'] = det.v_getOmega(det, A_mm, B_mm,
                                                   a_mm, b_mm, d_mm)

    cr_area_map = det.make_map(lambda:det.maia_data['Area mm2'])
    cr_sa_map = det.make_map(lambda:det.maia_data['Solid Angle'])

    # % diff b/w Matt's and Chris's solid angle maps (normalised to Chris's values
    det.det_show(100*(md_sa_map-cr_sa_map)/cr_sa_map, cmap='winter')
    plt.show()
