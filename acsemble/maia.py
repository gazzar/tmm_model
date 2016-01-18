import sys, os

# Set environ so that mayavi uses Qt instead of wx
os.environ.update(
    {'QT_API': 'pyqt', 'ETS_TOOLKIT': 'qt4'}
)

import config           # keep this near the top of the imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import transformations as tx

from mayavi import mlab


"""Maia detector class"""

PATH_HERE = os.path.abspath(os.path.dirname(__file__))
# sys.path = [os.path.join(PATH_HERE, '..')] + sys.path
MAIA_DATA = os.path.join(PATH_HERE, 'data', 'Maia_384C.csv')


class Pad(object):
    """Represents a single detector pad. The pad is created in the
    constructor using the size and position data from the pad_geometry tuple,
    then transformed via affine transformations to its absolute position in
    space using the detector centre and unit normal vectors. Fixed geometry
    attribute values are then immediately computed, so this is only done once at
    construction.

    Attributes
    ----------
    id : int
    width : float
    height : float
    pad_unit_normal : 3-element nd-array of floats; vector
    pad_centre_xyz : 3-element nd-array of floats; vector
    area_mm2 : float
    T : geometric transform matrix that transforms coordinates from their
        initial defined locations to the final pad location
    vertices : 5 row x 4 col nd-array of floats
        Final absolute coords of the four pad corners and the pad centre.
        Columns are x,y,z,1 where the column of 1's is a convenience value
        that arises when performing 4x4 geometric transform matrix operations.

    """
    # ids is a collection that tracks instantiated pad ids to ensure we never
    # create a pad with the same id twice.
    ids = set()

    def __init__(self, id, pad_geometry, detector_centre_mm,
                 detector_unit_normal):
        """
        Parameters
        ----------
        id : int
            pad id
        pad_geometry : tuple of floats
            (X, Y, Z, width, height)
        detector_centre_mm : tuple of floats
            (x, y, z)
        detector_unit_normal : tuple of floats
            (x, y, z)

        """
        px, py, pz, width, height = pad_geometry
        self.width = width = float(width)
        self.height = height = float(height)

        assert len(detector_centre_mm) == 3
        assert len(detector_unit_normal) == 3
        self.pad_unit_normal = (np.array(detector_unit_normal, dtype=float) /
                                     np.linalg.norm(detector_unit_normal))

        # Verify that we haven't allocated this id previously
        ids_len = len(Pad.ids)
        Pad.ids.add(id)
        assert len(Pad.ids) == ids_len + 1, ids_len

        self.id = id
        self.area_mm2 = width * height

        # On the next line, angle is the angle between the unit z-vector and
        # the detector_unit_normal vector.
        self.T = self._get_pad_transform_matrix(detector_centre_mm)
        self.vertices = self._vertices_from_params(px, py, pz, self.T)
        # Store centre coords of transformed pad coords
        px, py, pz = self.pad_centre_xyz = self.vertices[-1]

        # solid angle presented to the coordinate system origin
        self.omega = self.solid_angle()

        # Angles of the pad centre to the positive z-axis
        self.angle_X_rad = np.arctan2(px, pz)
        self.angle_Y_rad = np.arctan2(py, pz)

        # Theta and Phi spherical coordinate properties
        self.theta = np.arccos(pz / np.linalg.norm(self.pad_centre_xyz))
        self.phi = np.arctan2(py, px)

    @staticmethod
    def clear_pads():
        Pad.ids = set()

    def _vertices_from_params(self, cx, cy, cz, t):
        """Returns 3d coords of the four pad corners and centre:

        Parameters
        ----------
        cx, cy, cz : centre coordinates about which to perform transform
        t : 4 x 4 geometric transform matrix applied to the vertices

        Returns
        -------
        5 x 3 array of floats
            x1, y1, z1
            x2, y2, z2
            x3, y3, z3
            x4, y4, z4
            cx, cy, cz

        """
        w = self.width
        h = self.height
        # pad corner coords in the pad coord system whose plane normal is 0,0,1
        vertices = np.transpose(np.array([
            (cx + w / 2.0, cy + h / 2.0, cz, 1),    # corner 1
            (cx - w / 2.0, cy + h / 2.0, cz, 1),    # corner 2
            (cx + w / 2.0, cy - h / 2.0, cz, 1),    # corner 3
            (cx - w / 2.0, cy - h / 2.0, cz, 1),    # corner 4
            (cx,           cy,           cz, 1),    # centre
        ]))
        t = np.dot(t, vertices)[:-1]    # transform and strip superfluous 1's
        return np.transpose(t)

    def _get_pad_transform_matrix(self, detector_centre_mm):
        nhat = np.array([0.0, 0.0, 1.0])

        angle = tx.angle_between_vectors(nhat, self.pad_unit_normal)
        if (np.allclose(nhat, self.pad_unit_normal) or
            np.allclose(-nhat, self.pad_unit_normal)):
            # vectors are parallel or antiparallel
            direction = np.array([1.0, 0.0, 0.0])
        else:
            direction = np.cross(nhat, self.pad_unit_normal)
        R1 = tx.rotation_matrix(angle, direction)
        T1 = tx.translation_matrix(detector_centre_mm)
        T = tx.concatenate_matrices(T1, R1)
        return T

    def solid_angle(self):
        """Returns the solid angle of the detector pad w.r.t. a sphere
        centred at the origin.

        Returns
        -------
        float

        """
        # Non-overlapping triangles have vertex indices = [(0,1,2), (1,2,3)]
        # so calculate them separately and add them. The solid angle
        # calculation assumes the component pad triangles are non-overlapping.
        vectors = self.vertices[:4]
        norms = np.linalg.norm(vectors, axis=1)
        # unit vectors
        r0, r1, r2, r3 = vectors / norms[np.newaxis].T

        # triangle 1
        n1 = np.abs(np.dot(np.cross(r0, r1), r2))
        d1 = np.dot(r0, r1) + np.dot(r1, r2) + np.dot(r2, r0) + 1.0
        # triangle 2
        n2 = np.abs(np.dot(np.cross(r2, r1), r3))
        d2 = np.dot(r1, r2) + np.dot(r2, r3) + np.dot(r3, r1) + 1.0

        omega = 2.0 * (np.arctan(n1/d1) + np.arctan(n2/d2))
        return omega

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
        x, y, z = self.vertices.T
        triangles = [(0,1,2), (1,2,3)]
        mlab.triangular_mesh(x, y, z, triangles)
        if show_id:
            cx, cy, cz = self.pad_centre_xyz + self.pad_unit_normal * 0.1
            # orientation are angles "referenced to the z axis"
            mlab.text3d(cx, cy, cz, str(self.id), scale=0.2,
                        orient_to_camera=True,
                        # orient_to_camera=False,
                        # orientation=tx.euler_from_matrix(self.T),
            )


class Singleton(object):
    """From Brian Bruggeman's answer here
    http://stackoverflow.com/questions/31875/is-there-a-simple-elegant-way-to
    -define-singletons-in-python

    """
    def __new__(cls, *args, **kwds):
        it = cls.__dict__.get("__it__")
        if it is not None:
            return it
        cls.__it__ = it = object.__new__(cls)
        it.init(*args, **kwds)
        return it
    def init(self, *args, **kwds):
        pass


# class Maia(object):
class Maia(Singleton):
    """Represents the detector geometry and provides visualisation routines.
    We want this to be a proper singleton class.

    """
    def __init__(self, centre_mm=(0,0,10.0), unit_normal=(0,0,1)):
        """
        Parameters
        ----------
        centre_mm : 3-tuple of float
            Detector face centre (x, y, z) coords in mm
        unit_normal : array-like
            Unit vector perpendicular to detector face
        path : str
            Path to csv file containing detector pad data in Chris Ryan's Maia
            data csv format

        """
        assert len(centre_mm) == 3
        assert len(unit_normal) == 3

        # Read detector data
        path = config.detector_csv
        self.maia_data = pd.read_csv(path, index_col='Data',
                                     skipinitialspace=True, header=12)
        self.unit_normal = (np.array(unit_normal, dtype=float) /
                                np.linalg.norm(unit_normal))
        self.centre_mm = centre_mm
        self.pads = self.make_pads(self.maia_data, centre_mm, unit_normal)

    def make_pads(self, maia_data, maia_centre_mm, maia_unit_normal):
        """Create pad objects corresponding to maia_data Pandas dataframe

        Parameters
        ----------
        maia_data - Pandas dataframe corresponding to Chris Ryan's csv data

        Returns
        -------
        dict of Pad objects

        """
        pads = {}
        for id, p in maia_data.iterrows():
            pad_geometry = (p.X, p.Y, p.Z, p.width, p.height)
            pads[id] = Pad(id, pad_geometry, maia_centre_mm, maia_unit_normal)
        return pads

    def show3d(self, *args, **kwargs):
        # Show pads
        for p in self.pads.itervalues():
            p.show3d(*args, **kwargs)

        # Show origin (0,0,0)
        mlab.points3d([0], mode='axes', scale_mode='none')

    def _rect_solid_angle(self, a, b, d):
        """Return the solid angle of a rectangle with one corner at the origin.

        Parameters
        ----------
        a : float
            width of detector element
        b : float
            height of detector element
        d : float
            distance to plane of detector

        Returns
        -------
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
        This is only correct when the detector lies perpendicular to the beam
        axis. As pads can now be oriented arbitrarily, the solid angle
        computation within the pad class is to be used instead of this. This
        is retained to provide a unit test validation method.

        From RJ Mathar, Solid Angle of a Rectangular Plate,
        Note 2 at http://www.mpia-hd.mpg.de/~mathar/public
        http://www.mpia-hd.mpg.de/~mathar/public/mathar20051002.pdf

        Parameters
        ----------
        A - x-coord of detector element corner closest to the origin
        B - y-coord of detector element corner closest to the origin
        a - width of detector element
        b - height of detector element
        d - distance to plane of detector

        Returns
        -------
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

    def channel(self, row, col):
        """Return Dataframe for detector element at row, col index

        """
        return self.maia_data[(self.maia_data.Row == row) &
                              (self.maia_data.Column == col)]

    def area(self, id):
        """Return area of maia element row, col

        """
        return self.pads[id].area_mm2

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
    # MAIA_SINGLE_PAD_DATA = os.path.join(PATH_HERE, 'data',
    #                                     'pseudo_maia_as_one_square_pad.csv')

    # det = Maia(centre_mm=(0,0,10), unit_normal=(0,0,1))  # Detector instance
    # det = Maia(centre_mm=(5,5,0), unit_normal=(0,-1,0))
    # det = Maia(centre_mm=(0,0,10), unit_normal=(-1,0,-1))

    det = Maia(centre_mm=(0,0,-10), unit_normal=(0,0,1),
               path=config.detector_csv)

    det.show3d(show_id=True)
    mlab.show()
