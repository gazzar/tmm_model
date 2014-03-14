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
MAIA_DATA = os.path.join(PATH_HERE, 'data', 'Maia_384A.csv')


class Maia(object):
    """Represents the detector geometry and provides visualisation routines.

    """
    def __init__(self, d_mm=10.0):
        # Read Chris Ryan's detector data
        self.maia_data = pd.read_csv(MAIA_DATA, skipinitialspace=True,
                                      header=12)
        self.d_mm = d_mm
        self.rows = 20
        self.cols = 20
        self.shape = (self.rows, self.cols)

        # Calculate and append area and solid angle columns
        # then make some 2D maps for plotting and comparison
    
        a_mm = self.maia_data['width']
        b_mm = self.maia_data['height']
        A_mm = abs(self.maia_data['X']) - a_mm/2
        B_mm = abs(self.maia_data['Y']) - b_mm/2
    
        self.maia_data['area_mm2'] = a_mm * b_mm
        self.maia_data['omega'] = self.v_getOmega(self, A_mm, B_mm,
                                                        a_mm, b_mm, d_mm)
        self.maia_data['angle_X_rad'] = np.arctan(self.maia_data.X/d_mm)
        self.maia_data['angle_Y_rad'] = np.arctan(self.maia_data.Y/d_mm)


    def _rect_solid_angle(self, a, b, d):
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


    def _get_solid_angle(self, A, B, a, b, d):
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
        omega1 = self._rect_solid_angle(2.0*(A+a), 2.0*(B+b), d)
        omega2 = self._rect_solid_angle(2.0*A, 2.0*(B+b), d)
        omega3 = self._rect_solid_angle(2.0*(A+a), 2.0*B, d)
        omega4 = self._rect_solid_angle(2.0*A, 2.0*B, d)

        omega = (omega1 - omega2 - omega3 + omega4) / 4.0

        return omega

    # Create a vectorised version
    v_getOmega = np.vectorize(_get_solid_angle, excluded=['self', 'd'])


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
        map2d = np.zeros((self.rows, self.cols)) + fill_value
        map2d[self.maia_data['Row'],
              self.maia_data['Column']] = func()
        return map2d


    def element(self, row, col):
        """Return Dataframe for detector element at row, col index

        """
        return self.maia_data[(self.maia_data['Row']==row) &
                             (self.maia_data['Column']==col)]


    def area(self, row, col):
        """Return area of maia element row, col

        """
        el = self.element(row, col)
        return el.iloc[0].area_mm2


    def yx(self, row, col):
        """Return (Y, X) centre coords (mm) of maia element row, col

        """
        el = self.element(row, col)
        y, x = el.iloc[0][['Y', 'X']]
        return y, x


    def yx_angles_radian(self, row, col):
        """Return angles along Y and X to centre of maia element
        row, col

        """
        el = self.element(row, col)
        yr, xr = el.iloc[0][['angle_Y_rad', 'angle_X_rad']]
        return yr, xr


    def solid_angle(self, row, col):
        """Return solid angle of maia element row, col

        """
        el = self.element(row, col)
        '''
        a_mm, b_mm, y, x = el.iloc[0][['width', 'height', 'Y', 'X']]
        A_mm = abs(x) - a_mm/2
        B_mm = abs(y) - b_mm/2

        omega = self.get_omega(A_mm, B_mm, a_mm, b_mm, self.d_mm)
        return omega
        '''
        return el.iloc[0].omega


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

    print det.solid_angle(9, 7)

    # Location of Matt's CSV file that has the collimator info in it.
    dirPath = "."
    csvPath = \
        'mask-Mo-3x100-R-10-T-04-gap-75-grow-09-tune-103-safe-01-glue-025-1.csv'
    csvPath = os.path.join('tests', csvPath)

    # Get the solid angle distribution.
    md_sa_map, md_area_map = maia_funcs.getCollAreas(dirPath, csvPath, d_mm,
                                                     det_px_mm)

    cr_area_map = det.make_map(lambda:det.maia_data.area_mm2)
    cr_sa_map = det.make_map(lambda:det.maia_data.omega)

    # % diff b/w Matt's and Chris's solid angle maps (normalised to Chris's values
    det.det_show(100*(md_sa_map-cr_sa_map)/cr_sa_map, cmap='winter')
    #det.det_show(md_sa_map, cmap='winter')

    #angles_map = det.make_map(lambda:det.maia_data.angle_Y_rad)
    #det.det_show(angles_map, cmap='summer')

    plt.show()

