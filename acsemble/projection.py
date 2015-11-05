"""Projection class
Implements a class for 2D phantoms. The Golosio phantom is coded in terms of
geometry and composition data, with class methods supplied to instantiate it.
Alternatively, phantoms can be defined as a pair of files, one containing
geometry data as a greyscale bitmap and the other containing the composition
data structure defined in a yaml file.

"""

from __future__ import print_function
import config
import os
import logging
import glob
import fnmatch

import numpy as np
from numpy import exp, pi
import scipy.constants as sc
from scipy.special import expm1
import xraylib as xrl

from helpers import (write_tiff32, zero_outside_circle, rotate, imshow)
from data_helpers import MatrixProperties
from maia import Maia
import helpers
from progressbar import ProgressBar


UM_PER_CM = 1e4
J_PER_KEV = 1e3 * sc.eV
deg_to_rad = lambda x: x / 180 * pi
rad_to_deg = lambda x: x * 180 / pi

maia_d = Maia()  # Maia detector object (should be a singleton)


def absorption_sinogram(p, anglelist):
    """Generates the absorption sinogram for absorption by the full
    elemental content of the Phantom2d object.

    Parameters
    ----------
    p : Phantom2d object
    anglelist : list of float
        Ordered list of sinogram projection angles in degrees.

    Returns
    -------
    array of float
        Sinogram of requested scattering or fluorescence.
        This is a 2d x-theta map of dimensionless values.

    """
    sinogram = np.empty((p.cols, len(anglelist)))
    if config.show_progress:
        pbar = ProgressBar(maxval=len(anglelist)-1).start()
    for i, angle in enumerate(anglelist):
        if config.show_progress:
            pbar.update(i)

        increasing_ix = True   # Set True to accumulate cmam along increasing y
        n_map = irradiance_map(p, angle, n0=1.0, increasing_ix=increasing_ix)
        if increasing_ix:
            sinogram[:, i] = np.log(n_map[0] / n_map[-1])
        else:
            sinogram[:, i] = np.log(n_map[-1] / n_map[0])
    return sinogram


def outgoing_cmam(p, q, angle, energy, increasing_ix=True):
    """Compute and return the outgoing cumulative multiplicative absorption map
    (cmam), aka xi'.

    Parameters
    ----------
    p : Phantom2d object
        p.energy - incident beam photon energy (keV).
        p.um_per_px - length of one pixel of the map (um).
    q : int
        Maia detector channel id
    angle : float
        Tomography projection angle (degree)
    energy : float
        outgoing radiation energy (keV)
    increasing_ix : bool, optional
        If False, performs cumulative sum in opposite direction (in direction of
        decreasing y-index). Note, for the standard Radon transform,
        this direction is unimportant, but for the Radon transform with
        attenuation, we should project in the beam propagation direction.
        (default True).

    Returns
    -------
    2d ndarray of float
        cumulative multiplicative absorption map.

    """
    # The linear absorption map mu = ma_M * mu_M + sum_k ( ma_k * mu_k )
    mu = p.matrix.ma(energy) * p.el_maps['matrix']
    for el in p.el_maps:
        if el == 'matrix':
            continue
        Z = xrl.SymbolToAtomicNumber(el)
        mu += xrl.CS_Total(Z, energy) * p.el_maps[el]

    # project at angle theta by rotating the phantom by angle -theta and
    # projecting along the z-axis (along columns)

    # rotate by the sum of the projection angle and local detector angle
    # Get angle to rotate maps so that propagation toward detector plane
    # corresponds with direction to maia detector element
    phi_x = maia_d.pads[q].angle_X_rad
    phix_deg = rad_to_deg(phi_x)

    # Apply local rotation, accumulation and rotation operators

    # Apply R_{-theta} operator to mu
    # Apply R_{-phi} operator
    mu = rotate(mu, -(angle + phix_deg))

    # Apply C_z operator
    if increasing_ix:
        mu = np.cumsum(mu, axis=0)
    else:
        mu = np.cumsum(mu[::-1], axis=0)[::-1]

    # Apply R_{phi} operator
    mu = rotate(mu, phix_deg)
    t = p.um_per_px / UM_PER_CM

    phi_y = maia_d.pads[q].angle_Y_rad
    return mu * t / np.cos(phi_y)


def project_sinogram(event_type, p, q, anglelist, el=None):
    """Generates the sinogram of the requested element accounting for
    absorption by the Phantom2d composition and the geometry.

    Parameters
    ----------
    event_type : string
        One of ['rayleigh', 'compton', 'fluoro'].
    p : Phantom2d object
        p.energy - incident beam photon energy (keV).
        p.um_per_px - length of one pixel of the map (um).
    q : int
        Maia detector channel id
    anglelist : list of float
        Ordered list of sinogram projection angles in degrees.
    el : string, optional
        Name of element (e.g. 'Fe') used if projecting that element's
        fluorescence.

    Returns
    -------
    array of float
        Sinogram of requested scattering or fluorescence.

    """
    assert event_type in ['rayleigh', 'compton', 'fluoro']

    sinogram = np.empty((p.cols, len(anglelist)))
    if config.show_progress:
        if len(anglelist) > 1:
            pbar = ProgressBar(maxval=len(anglelist)-1).start()
        else:
            pbar = ProgressBar(maxval=1).start()
    for i, angle in enumerate(anglelist):
        if config.show_progress:
            pbar.update(i)

        increasing_ix = True   # Set True to accumulate cmam along increasing y
        n_map = irradiance_map(p, angle, n0=1.0, increasing_ix=increasing_ix)
        e_map = channel_fluoro_map(p, q, n_map, angle, el)
        energy = outgoing_photon_energy(event_type, p, q, el)
        c = outgoing_cmam(p, q, angle, energy, increasing_ix=increasing_ix)
        if config.no_out_absorption:
            sinogram[:, i] = e_map.sum(axis=0)
        else:
            sinogram[:, i] = (e_map * np.exp(-c)).sum(axis=0)
    return sinogram


def irradiance_map(p, angle, n0=1.0, increasing_ix=True, matrix_only=False):
    """Generates the image-sized map of irradiance [1/(cm2 s)] at each 2d pixel
    for a given angle accounting for absorption at the incident energy by the
    full elemental distribution.

    Parameters
    ----------
    p : phantom object
        p.energy - incident beam energy (keV)
        p.um_per_px - length of one pixel of the map (um)
    angle : float
        angle in degrees (0 degrees is with the source below the object,
        projecting parallel to the y-axis).
    n0 : float
        incident irradiance (default 1.0).
    increasing_ix : bool, optional
        If False, performs cumulative sum in opposite direction (in direction of
        decreasing y-index). Note, for the standard Radon transform,
        this direction is unimportant, but for the Radon transform with
        attenuation, we should project in the beam propagation direction.
        (default True).
    matrix_only : bool, optional
        If True, the map only considers the matrix and doesn't consider the
        other elements in the model.
        (default False).

    Returns
    -------
    2d ndarray of float
        The irradiance map.

    """
    # matrix_map = zero_outside_circle(p.el_maps['matrix'])

    # The linear absorption map mu0 = ma_M * mu_M + sum_k ( ma_k * mu_k )
    mu0 = p.matrix.ma(p.energy) * p.el_maps['matrix']
    if not matrix_only:
        for el in p.el_maps:
            if el == 'matrix':
                continue
            Z = xrl.SymbolToAtomicNumber(el)
            mu0 += xrl.CS_Total(Z, p.energy) * p.el_maps[el]

    # project at angle theta by rotating the phantom by angle -theta and
    # projecting along the z-axis (along columns)
    im = rotate(mu0, -angle)     # rotate by angle degrees ccw
    t = p.um_per_px / UM_PER_CM
    # accumulate along z-axis (image defined in xz-coordinates, so accumulate
    # along image rows), consistent with matlab sinogram convention.
    # See http://www.mathworks.com/help/images/radon-transform.html
    if increasing_ix:
        cmam = t * np.cumsum(im, axis=0)
    else:
        cmam = t * np.cumsum(im[::-1], axis=0)[::-1]
    if config.no_in_absorption:
        n_map = n0 + np.zeros_like(cmam)
    else:
        n_map = n0 * exp(-cmam)
    return n_map


def scattering_ma(event_type, p, row, col):
    """Return the Rayleigh or Compton differential mass attenuation coefficients
    (cm2/g/sr) for icru44 brain tissue with density described by the 'matrix'
    map of phantom p into the Maia detector element indexed by row, col. This
    needs to be multiplied later by the Maia-channel-dependent solid angle.

    Arguments:
    event_type - string, one of ['rayleigh', 'compton']
    p - phantom instance (matrix plus elements)
    row, col - maia detector element indices

    Returns:
    The (rayleigh, compton) differential mass attenuation coefficient

    """
    assert event_type in ['rayleigh', 'compton']

    # Get spherical angles (polar theta & azimuthal phi) to detector element
        # Get spherical angles (polar theta & azimuthal phi) to detector element
    theta = maia_d.pads[q].theta
    phi = maia_d.pads[q].phi

    compound = p.matrix.cp  # elemental data for matrix compound

    # Get the contribution to Rayleigh and Compton scattering from each
    # element in the matrix compound and sum these.
    ma = 0.0
    for el in compound:
        # Assuming propagation along z, coordinate system used by the DCSP_Rayl
        # and DCSP_Compt methods is shown here:
        # http://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/
        # 3D_Spherical.svg/200px-3D_Spherical.svg.png
        # i.e. spherical coordinates with polar angle theta, azimuthal angle phi
        z = xrl.SymbolToAtomicNumber(compound[el])

        # Mass attenuation coefficients from cross-sections. See
        # http://physics.nist.gov/PhysRefData/XrayMassCoef/chap2.html
        # Units of the following expression:
        # elemental fraction by weight * differential mass attenuation coefft
        # unitless * cm2/g/sr
        if event_type == 'rayleigh':
            f = xrl.DCSP_Rayl
        else:
            f = xrl.DCSP_Compt
        ma += compound[el] * f(z, p.energy, theta, phi)
    return ma


def compton_scattered_energy(energy_in, q):
    """Energy of the Compton photons scattered into the direction of the Maia
    detector element.

    Parameters
    ----------
    energy_in : float
        Incident beam photon energy (keV).
    q : int
        Maia detector channel id

    Returns
    -------
    float
        Energy of scattered photons (keV)

    """
    # Get polar scattering angle (theta) to detector element
    # Get spherical angles (polar theta & azimuthal phi) to detector element
    theta = maia_d.pads[q].theta

    # energy_out = 1.0 / (1.0/energy_in +
    # (1 - np.cos(theta))*J_PER_KEV/sc.m_e/sc.c/sc.c)
    return xrl.ComptonEnergy(energy_in, theta)


'''
def old_emission_map(event_type, p, n_map, angle, el=None):
    """Compute the maia-detector-shaped map of Rayleigh, Compton or K-edge
    fluorescence from the element map for an incident irradiance map n_map.

    Parameters
    ----------
    event_type : string
        One of ['rayleigh', 'compton', 'fluoro'].
    p : Phantom2d object
        p.energy - incident beam photon energy (keV).
        p.um_per_px - length of one pixel of the map (um).
    n_map : 2d ndarray of float
        Map of incident irradiance.
    angle : float
        Stage rotation angle (degrees).
    el : string, optional
        Name of element (e.g. 'Fe') used if projecting that element's
        fluorescence.

    Returns
    -------
    1d ndarray of float
        The accumulated flux in Maia detector row 7 for the requested edge.

    """
    assert event_type in ['rayleigh', 'compton', 'fluoro']

    # Get matrix map and, for fluorescence, the map for the requested element,
    # and rotate them to the same angle as the intensity map since these all
    # need to be in registration.
    matrix_map = zero_outside_circle(p.el_maps['matrix'])
    matrix_map_r = rotate(matrix_map, -angle)
    del matrix_map

    k_alpha_energy = -1
    if event_type == 'fluoro':
        edge_map = zero_outside_circle(p.el_maps[el])
        edge_map_r = rotate(edge_map, -angle)
        del edge_map

        # Do this here because we're outside the detector channel loop.
        # Get Z for the fluorescing element and check that its K_alpha is
        # below the incident energy.
        el_z = xrl.SymbolToAtomicNumber(el)
        line = xrl.KA_LINE
        k_alpha_energy = xrl.LineEnergy(el_z, line)
        assert k_alpha_energy < p.energy

    # 2d accumulator for results
    accumulator = np.empty((maia_d.shape[1], n_map.shape[0]))

    # Iterate over maia detector elements in theta, i.e. maia columns
    # This should be parallelizable
    row = 7
    # Get multiplication factor for additional distance that an outgoing photon
    # must travel as it passes out-of-plane to the detector.
    delta_theta_yx_radian = maia_d.yx_angles_radian(row, col=0)
    delta_theta_y_radian = delta_theta_yx_radian[0]
    y_distance_factor = 1.0 / np.cos(delta_theta_y_radian)

    for channel_id in maia_d.channel_selection(row=row):
        col = maia_d.maia_data_column_from_id(channel_id, 'Column')
        # Get angle to rotate maps so that propagation toward detector plane
        # corresponds with direction to maia detector element
        delta_theta_yx_radian = maia_d.yx_angles_radian(row, col)
        delta_theta_x = rad_to_deg(delta_theta_yx_radian[1])

        # For every maia detector element, get the solid angle (parallelize?)
        # Orient the maps toward the maia element
        # TODO: check sign of delta: +ve or -ve?
        imap_rm = rotate(n_map, -delta_theta_x)
        matrix_map_rm = rotate(matrix_map_r, -delta_theta_x)

        # Rotate the geometry so that the detector is on the bottom, so we can
        # integrate by stepping through the row indices.
        imap_rm = np.rot90(imap_rm)
        matrix_map_rm = np.rot90(matrix_map_rm)

        # Solid angle of Maia channel
        omega = maia_d.solid_angle(row, col)

        # mass attenuation coefft. (cm2/g)
        if event_type == 'fluoro':
            edge_map_rm = rotate(edge_map_r, -delta_theta_x)
            edge_map_rm = np.rot90(edge_map_rm)

            # Simulate fluorescence event:
            # Generate the initial fluorescence intensity
            # Start with the highest-energy edge or a mean factor accounting for
            # all edges first (move to other edges later?)
            mac = fluoro_ma(p, el_z)

            # Scale for propagation over one voxel
            # *_mac_t = *_mac * p.um_per_px/UM_PER_CM
            # (cm3/g/sr) =   (cm2/g/sr) * cm
            mac_t = mac * p.um_per_px / UM_PER_CM * y_distance_factor
            # Generate outgoing radiation.
            # This is the fluorescence intensity map.
            imap_rm *= -expm1(-edge_map_rm * omega * mac_t)
            del edge_map_rm
        else:
            mac = scattering_ma(event_type, p, row, col)
            mac_t = mac * p.um_per_px / UM_PER_CM * y_distance_factor
            # Generate outgoing radiation.
            # This is the scattering radiation intensity map.
            imap_rm *= -expm1(-matrix_map_rm * omega * mac_t)

        # Now we've "evented," we use the mass attenuation coefficients of the
        # matrix for propagation with absorption out to the detector, but this
        # is at a new energy depending on the event type.
        energy = {
            'rayleigh': p.energy,
            'compton': compton_scattered_energy(p.energy, row, col),
            'fluoro': k_alpha_energy,
        }[event_type]
        mac_t = brain.ma(energy) * p.um_per_px / UM_PER_CM * y_distance_factor

        # Propagate all intensity to the detector, accumulating (+) and
        # absorbing [exp(-mu/rho rho t)] as we go.
        cmam_matrix = np.cumsum(matrix_map_rm, axis=0) * mac_t
        i_out = (imap_rm * exp(-cmam_matrix * omega)).sum(axis=0)

        # Store the result for this detector element.
        accumulator[col] = i_out

    return accumulator
'''


def outgoing_photon_energy(event_type, p, q=None, el=None):
    """Return the interaction-type-dependent outgoing photon energy in keV.

    Parameters
    ----------
    event_type : string
        One of ['rayleigh', 'compton', 'fluoro'].
    p : Phantom2d object
        p.energy - incident beam photon energy (keV).
        p.um_per_px - length of one pixel of the map (um).
    q : int (optional, required for compton interaction)
        Maia detector channel id
    el : string (optional, required for fluoro interaction)
        Name of fluorescing element (e.g. 'Fe').

    Returns
    -------
    float
        Outgoing photon energy (keV)

    """
    assert event_type in ['rayleigh', 'compton', 'fluoro']

    def k_alpha_energy(el):
        el_z = xrl.SymbolToAtomicNumber(el)
        line = xrl.KA_LINE
        energy = xrl.LineEnergy(el_z, line)
        #assert energy < p.energy
        return energy

    if event_type == 'rayleigh':
        energy = p.energy
    elif event_type == 'compton':
        energy = compton_scattered_energy(p.energy, q)
    else:           # 'fluoro'
        energy = k_alpha_energy(el)

    return energy


def fluoro_emission_map(p, n_map, angle, el):
    """Compute the maia-detector-shaped map of K-edge
    fluorescence from the element map for an incident irradiance map n_map.

    Parameters
    ----------
    p : Phantom2d object
        p.energy - incident beam photon energy (keV).
        p.um_per_px - length of one pixel of the map (um).
    n_map : 2d ndarray of float
        Map of incident irradiance.
    angle : float
        Stage rotation angle (degrees).
    el : string
        Name of fluorescing element (e.g. 'Fe').

    Returns
    -------
    2d ndarray of float
        The fluorescence emission map for the requested edge.

    """
    # edge_map = zero_outside_circle(p.el_maps[el])
    edge_map = p.el_maps[el]
    edge_map_r = rotate(edge_map, -angle)
    del edge_map

    # Get Z for the fluorescing element
    el_z = xrl.SymbolToAtomicNumber(el)
    line = xrl.KA_LINE

    # Sanity check that el K_alpha is below the incident energy.
    k_alpha_energy = xrl.LineEnergy(el_z, line)
    #assert k_alpha_energy < p.energy
    if k_alpha_energy >= p.energy:
        Q = 0.0
    else:
        # Simulate fluorescence event:
        # CS_FluorLine_Kissel_Cascade is the XRF cross section Q_{i,YX} in Eq. (12)
        # of Schoonjans et al.
        Q = xrl.CS_FluorLine_Kissel_Cascade(el_z, line, p.energy)
        # print(el, end=' ')

    # 2d array for results
    emission_map = n_map * Q * edge_map_r * p.um_per_px / UM_PER_CM

    return emission_map


def channel_emission_map(event_type, p, q, n_map, angle, el):
    """Select and defer to the interaction-type-specific emission map
    computation.

    Parameters
    ----------
    event_type : string
        One of ['rayleigh', 'compton', 'fluoro'].
    p : Phantom2d object
        p.energy - incident beam photon energy (keV).
        p.um_per_px - length of one pixel of the map (um).
    q : int
        Maia detector channel id
    n_map : 2d ndarray of float
        Map of incident irradiance.
    angle : float
        Stage rotation angle (degrees).
    el : string
        Name of fluorescing element (e.g. 'Fe').

    Returns
    -------
    2d ndarray of float
        The fluorescence emission map for the requested edge.

    """
    assert event_type in ['rayleigh', 'compton', 'fluoro']

    if event_type == 'fluoro':
        e_map = channel_fluoro_map(p, q, n_map, angle, el)
    elif event_type == 'rayleigh':
        e_map = channel_rayleigh_map(p, q, n_map, angle)
    elif event_type == 'compton':
        e_map = channel_compton_map(p, q, n_map, angle)
    else:
        raise RuntimeError('Brain explodes')

    return e_map


def channel_fluoro_map(p, q, n_map, angle, el):
    """Compute the maia-detector-shaped map of K-edge
    fluorescence from the element map for an incident irradiance map n_map.

    Parameters
    ----------
    p : Phantom2d object
        p.energy - incident beam photon energy (keV).
        p.um_per_px - length of one pixel of the map (um).
    q : int
        Maia detector channel id
    n_map : 2d ndarray of float
        Map of incident irradiance.
    angle : float
        Stage rotation angle (degrees).
    el : string
        Name of fluorescing element (e.g. 'Fe').

    Returns
    -------
    2d ndarray of float
        The fluorescence emission map for the requested edge.

    """
    solid_angle = maia_d.pads[q].omega
    return (solid_angle / 4 / np.pi *
            fluoro_emission_map(p, n_map, angle, el))


def channel_rayleigh_map(p, q, n_map, angle):
    """Compute the maia-detector-shaped map of Rayleigh scattering
    from the element map for an incident irradiance map n_map.

    Parameters
    ----------
    p : Phantom2d object
        p.energy - incident beam photon energy (keV).
        p.um_per_px - length of one pixel of the map (um).
    q : int
        Maia detector channel id
    n_map : 2d ndarray of float
        Map of incident irradiance.
    angle : float
        Stage rotation angle (degrees).
    el : string
        Name of fluorescing element (e.g. 'Fe').

    Returns
    -------
    2d ndarray of float
        The fluorescence emission map for the requested edge.

    """
    solid_angle = maia_d.pads[q].omega

    # Get spherical angles (polar theta & azimuthal phi) to detector element
    theta = maia_d.pads[q].theta
    phi = maia_d.pads[q].phi

    energy = outgoing_photon_energy('rayleigh', p)

    # get a list of all elements el and their weights w_el

    # The absorption map mu = ma_M * mu_M + sum_k ( ma_k * mu_k )
    mu = p.matrix.ma(energy) * p.el_maps['matrix']
    for el in p.el_maps:
        if el == 'matrix':
            continue
        Z = xrl.SymbolToAtomicNumber(el)
        mu += p.matrix.cp[el] * xrl.DCSP_Rayl(Z, energy, theta, phi) * \
              p.el_maps[el]

    ma += xrl.DCSP_Rayl(Z, energy, theta, phi)

    # edge_map = zero_outside_circle(p.el_maps[el])
    edge_map_r = rotate(edge_map, -angle)
    del edge_map

    # Get Z for the fluorescing element
    el_z = xrl.SymbolToAtomicNumber(el)
    line = xrl.KA_LINE

    # Sanity check that el K_alpha is below the incident energy.
    k_alpha_energy = xrl.LineEnergy(el_z, line)
    assert k_alpha_energy < p.energy

    # Simulate fluorescence event:
    # CS_FluorLine_Kissel_Cascade is the XRF cross section Q_{i,YX} in Eq. (12)
    # of Schoonjans et al.
    Q = xrl.CS_FluorLine_Kissel_Cascade(el_z, line, p.energy)

    # 2d array for results
    emission_map = n_map * Q * edge_map_r * p.um_per_px / UM_PER_CM

    return emission_map


def write_sinogram(im, p, event_type, el='matrix'):
    """Project and write sinogram for element map el
    Creates a filename by prepending s_ to the filename and appending r or c
    if writing the rayleigh or compton sinogram, respectively.

    Parameters
    ----------
    im : 2d ndarray of float
        sinogram.
    p : Phantom2d object
    event_type : string
        One of ['absorption', 'rayleigh', 'compton', 'fluoro'].
    el : string
        name of current element, e.g. 'Fe'. (default 'matrix').

    """
    # Get the filename that matches the glob pattern for this element
    # and prepend s_ to it
    pattern = p.filename

    matches = helpers.match_pattern(pattern, glob.glob(pattern))
    if matches:
        # Just get the filename that's a match for the element el
        match_base = [m[0] for m in matches if el==m[1]][0]
    else:
        raise Exception('Element {} not found in {}'.format(el, matches))
    path = os.path.dirname(pattern)
    base = os.path.basename(match_base)
    s_filename = os.path.join(path, 's_'+base)

    # Write sinogram (absorption map)

    # append r, c, f, a to -matrix suffix so sinograms read in as unique images
    if '-matrix' in s_filename:
        s_filename = s_filename.replace('-matrix', '-matrix-' + event_type[0])
    write_tiff32(s_filename, im)

    return im


def project(p, event_type):
    """This is the entry point for the sinogram project script.
    Performs the projection at all specified angles and writes the resulting
    sinograms out as tiffs.

    Parameters
    ----------
    p : Phantom2d object
    event_type : string
        One of 'afrc'.

    """
    # globals().update(args)  # expose args_namespace

    assert event_type in 'afrc'
    event_type = {'a': 'absorption', 'f': 'fluoro', 'r': 'rayleigh',
                  'c': 'compton'}[event_type]

    anglelist = np.loadtxt(config.anglesfile)
    q = config.detector_pads[0]
    if event_type == 'absorption':
        im = absorption_sinogram(p, anglelist)
        s = write_sinogram(im, p, event_type)
    elif event_type == 'fluoro':
        # fluorescence sinogram
        for el in p.el_maps:
            if el == 'matrix':
                continue
            im = project_sinogram(event_type, p, q, anglelist, el)
            s = write_sinogram(im, p, event_type, el)
    else:
        # event_type is 'rayleigh' or 'compton'
        # Absorption, Rayleigh or Compton scattering sinogram of matrix
        im = project_sinogram(event_type, p, q, anglelist)
        s = write_sinogram(im, p, event_type)

    return s


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import phantom

    config.parse()  # read config file settings

    BASE = r'R:\Science\XFM\GaryRuben\projects\TMM'
    DATA_DIR = os.path.join(BASE, r'work\20141016')
    os.chdir(DATA_DIR)

    el = 'Cu'

    p = phantom.Phantom2d(filename='phantom1*.tiff',
                          yamlfile='phantom1-%s.yaml'%el,
                          um_per_px=12.52,
                          energy=9)

    anglelist = np.loadtxt(config.anglesfile, dtype=int)
    # sinogram = absorption_sinogram(p, anglelist, el, show_progress=True)
    # sinogram = project_sinogram('rayleigh', p, anglelist, el, show_progress=True)

    q = config.detector_pads[0]
    sinogram = project_sinogram('fluoro', p, q, anglelist, el)

    # np.save('sino'+el, sinogram)
    imshow(sinogram, extent=[0, 360, 0, 99], aspect=2, cmap='cubehelix')
    plt.xlabel('rotation angle (deg)')
    plt.ylabel('x')
    plt.show()


    for event_type in 'arcf':
        event_name = {'a': 'absorption', 'f': 'fluoro', 'r': 'rayleigh',
                      'c': 'compton'}[event_type]
        project(p, event_type)
