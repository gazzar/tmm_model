import sys, os

# Set environ so that mayavi uses Qt instead of wx
os.environ.update(
    {'QT_API': 'pyqt', 'ETS_TOOLKIT': 'qt4'}
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mayavi import mlab


"""Maia detector class"""

PATH_HERE = os.path.abspath(os.path.dirname(__file__))
# sys.path = [os.path.join(PATH_HERE, '..')] + sys.path
MAIA_DATA = os.path.join(PATH_HERE, 'data', 'Maia_384C.csv')


class Pad(object):
    """Represents a single detector pad

    """
    ids = set()

    def __init__(self, id, centre_xyz, unit_normal, width, height):
        assert len(centre_xyz) == 3
        assert len(unit_normal) == 3
        self.centre_xyz = np.array(centre_xyz, dtype=float)
        self.unit_normal = np.array(unit_normal, dtype=float) / \
                           np.linalg.norm(unit_normal)

        width = float(width)
        height = float(height)
        ids_len = len(Pad.ids)
        Pad.ids.add(id)
        assert len(Pad.ids) == ids_len + 1
        self.id = id
        self.width = width
        self.height = height
        self.T = np.eye(4)      # Start with identity transform matrix
        self.vertices = self._vertices_from_params()

    def _vertices_from_params(self):
        w = self.width
        h = self.height
        cx, cy, cz = self.centre_xyz
        # pad corner coords in the pad coord system whose plane normal is 0,0,1
        vertices = np.array([
            (cx + w / 2.0, cy + h / 2.0, cz, 1),
            (cx - w / 2.0, cy + h / 2.0, cz, 1),
            (cx + w / 2.0, cy - h / 2.0, cz, 1),
            (cx - w / 2.0, cy - h / 2.0, cz, 1),
        ])
        return np.dot(self.T, vertices)

    def _get_pad_rotation_matrix(self):
        '''
        # Find the rotation matrix R that rotates 0,0,1 to self.unit_normal
        # See http://math.stackexchange.com/questions/180418/
        #         calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
        a = np.array([0.0, 0.0, 1.0])
        v = np.cross(a, self.unit_normal)
        s = np.norm(v)
        '''
        # http://gamedev.stackexchange.com/questions/20097/
        # how-to-calculate-a-3x3-rotation-matrix-from-2-direction-vectors
        # Note: My vectors are normalised so I don't normalize the results as
        # done in the stackoverflow answer.
        R = np.eye(4)
        nhat = [0.0, 0.0, 1.0]
        if not np.allclose(nhat, self.unit_normal):
            R[0, 0:3] = np.array(nhat)
            R[2, 0:3] = mz = np.cross(nhat, self.unit_normal)
            R[1, 0:3] = np.cross(mz, nhat)
        return R

    def show3d(self, show_id=False):
        """Show the pad in 3d using the Mayavi mlab triangular_mesh function.

        A simple example of showing a square lying in the z=0 plane:

        #    0    1    2    3      <- indices used in triangles
        x = [0.0, 1.0, 0.0, 1.0]
        y = [0.0, 0.0, 1.0, 1.0]
        z = [0.0, 0.0, 0.0, 0.0]
        triangles = [(0,1,2), (1,2,3)]
        mlab.triangular_mesh(x, y, z, triangles)

        The pad is rectangular, requiring two triangles to show it, with coords
        at indices referred to in the triangles list of tuples.

        """
        x, y, z, w_ = self.vertices.T
        triangles = [(0,1,2), (1,2,3)]
        mlab.triangular_mesh(x, y, z, triangles)
        if show_id:
            cx, cy, cz = self.centre_xyz
            mlab.text3d(cx-0.2, cy, cz+0.1, str(self.id), scale=0.2,
                        orient_to_camera=False, orientation=[0,0,1])


class Maia(object):
    """Represents the detector geometry and provides visualisation routines.

    """

    def __init__(self, d_mm=10.0):
        """
        d_mm : float
            Distance in mm from specimen centre to detector face

        """
        # Read Chris Ryan's detector data
        self.maia_data = pd.read_csv(MAIA_DATA, index_col='Data',
                                     skipinitialspace=True, header=12)
        self.pads = self.make_pads(self.maia_data)
        self.d_mm = d_mm
        self.rows = 20
        self.cols = 20
        self.shape = (self.rows, self.cols)

        # Calculate and append area and solid angle columns
        # then make some 2D maps for plotting and comparison
        x = self.maia_data.X
        y = self.maia_data.Y

        a_mm = self.maia_data['width']
        b_mm = self.maia_data['height']
        A_mm = abs(x) - a_mm / 2
        B_mm = abs(y) - b_mm / 2

        # Ensure that the pads don't cross the x or y axes because there are
        # simplifying assumptions in the code that rely on this fact.
        assert (A_mm >= 0.0).all()
        assert (B_mm >= 0.0).all()

        self.maia_data['area_mm2'] = a_mm * b_mm
        self.maia_data['omega'] = self.v_getOmega(self, A_mm, B_mm,
                                                  a_mm, b_mm, d_mm)
        self.maia_data['angle_X_rad'] = np.arctan(x / d_mm)
        self.maia_data['angle_Y_rad'] = np.arctan(y / d_mm)

        self.maia_data['theta'] = np.pi / 2 + np.arctan(d_mm / np.hypot(x, y))
        self.maia_data['phi'] = np.arctan2(y, x)

    def make_pads(self, maia_data):
        """Create pad objects corresponding to maia_data Pandas dataframe

        Arguments:
        maia_data - Pandas dataframe corresponding to Chris Ryan's csv data

        Returns:
        list of Pad objects

        """
        pads = []
        for id, p in maia_data.iterrows():
            pads.append(Pad(id, (p.X, p.Y, p.Z), [0,0,1], p.width, p.height))
        return pads

    def show3d(self, *args, **kwargs):
        for p in self.pads:
            p.show3d(*args, **kwargs)

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
        fact = np.sqrt(
            (1 + alpha ** 2 + beta ** 2) / ((1 + alpha ** 2) * (1 + beta ** 2)))
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
        omega1 = self._rect_solid_angle(2.0 * (A + a), 2.0 * (B + b), d)
        omega2 = self._rect_solid_angle(2.0 * A, 2.0 * (B + b), d)
        omega3 = self._rect_solid_angle(2.0 * (A + a), 2.0 * B, d)
        omega4 = self._rect_solid_angle(2.0 * A, 2.0 * B, d)

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
        map2d[self.maia_data.Row,
              self.maia_data.Column] = func()
        return map2d


    def channel(self, row, col):
        """Return Dataframe for detector element at row, col index

        """
        return self.maia_data[(self.maia_data.Row == row) &
                              (self.maia_data.Column == col)]


    def area(self, row, col):
        """Return area of maia element row, col

        """
        el = self.channel(row, col)
        return el.iloc[0].area_mm2


    def yx(self, row, col):
        """Return (Y, X) centre coords (mm) of maia element row, col

        """
        el = self.channel(row, col)
        y, x = el.iloc[0][['Y', 'X']]
        return y, x


    def yx_angles_radian(self, row, col):
        """Return angles along Y and X to centre of maia element
        row, col

        """
        el = self.channel(row, col)
        yr, xr = el.iloc[0][['angle_Y_rad', 'angle_X_rad']]
        return yr, xr


    def solid_angle(self, row, col):
        """Return solid angle of maia element row, col

        """
        el = self.channel(row, col)
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


    def channel_selection(self, quadrant=None, row=None, col=None):
        """A generator for the Maia channel IDs (which are also the
        Pandas dataframe IDs) of all Maia channels in the specified
        group, where the group is specified by quadrant, row or
        column values or ranges.

        Parameters
        ----------
        quadrant : int, default None
            one or more from the set (0-3). e.g. [0, 2].
        row : int, default None
            The Maia channel row or rows (0-19), e.g. 1, 10, (1, 9), [1, 3, 5]
                                                range(5,9).
        col : int, default None
            The Maia channel column (0-19). See row examples.

        Yields
        ------
        All channel IDs in the group.

        """
        df = self.maia_data
        if quadrant is not None:
            df = df[np.in1d(df.Quadrant, quadrant)]
        if row is not None:
            df = df[np.in1d(df.Row, row)]
        if col is not None:
            df = df[np.in1d(df.Column, col)]

        for channel in df.index.values:
            yield channel


    def maia_data_column_from_id(self, channel_id, column_name):
        """Return the entry from the specified channel id and column in the
        maia_data Pandas dataframe.

        Parameters
        ----------
        channel_id : int
            Pandas Dataframe index corresponding to the Maia "Data" value.
        column_name : str
            csv column name string, e.g. "Quadrant".

        Returns
        -------
        Requested value (float).

        """
        return self.maia_data.loc[channel_id][column_name]


if __name__ == '__main__':
    from tests import maia_funcs

    det = Maia()  # Make a detector instance
    det.show3d(show_id=False)
    mlab.show()

    '''
    # Geometry
    d_mm = 10.0
    det_px_mm = 0.4

    print det.solid_angle(7, 7)

    # Location of Matt's CSV file that has the collimator info in it.
    dirPath = "."
    csvPath = \
        'mask-Mo-3x100-R-10-T-04-gap-75-grow-09-tune-103-safe-01-glue-025-1.csv'
    csvPath = os.path.join('tests', csvPath)

    # Get the solid angle distribution.
    md_sa_map, md_area_map = maia_funcs.getCollAreas(dirPath, csvPath, d_mm,
                                                     det_px_mm)

    cr_area_map = det.make_map(lambda: det.maia_data.area_mm2)
    cr_sa_map = det.make_map(lambda: det.maia_data.omega)

    # % diff b/w Matt's and Chris's solid angle maps (normalised to Chris's values
    det.det_show(100 * (md_sa_map - cr_sa_map) / cr_sa_map, cmap='winter')
    # det.det_show(md_sa_map, cmap='winter')

    #angles_map = det.make_map(lambda:det.maia_data.angle_Y_rad)
    #det.det_show(angles_map, cmap='summer')
    plt.show()
    '''