#!/usr/bin/env python

# Copyright (c) 2014, Gary Ruben
# Released under the Modified BSD license
# See LICENSE


"""Projection class
Implements a class for 2D phantoms. The Golosio phantom is coded in terms of
geometry and composition data, with class methods supplied to instantiate it.
Alternatively, phantoms can be defined as a pair of files, one containing
geometry data as a greyscale bitmap and the other containing the composition
data structure defined in a yaml file.

"""

from __future__ import print_function
import sys
import os
import glob
import fnmatch

import numpy as np
from numpy import exp, pi
import scipy.constants as sc
from scipy.special import expm1
import xraylib as xrl

from helpers import (write_tiff32, zero_outside_circle, rotate, imshow)
from data_helpers import brain_properties, reweight_compound
from maia import Maia


UM_PER_CM = 1e4
J_PER_KEV = 1e3 * sc.eV
deg_to_rad = lambda x: x / 180 * pi
rad_to_deg = lambda x: x * 180 / pi

brain = brain_properties()  # brain data object instance
maia_d = Maia()  # Maia detector object singleton


def project_sinogram(event_type, p, anglelist, el=None, show_progress=False):
    """Generates the sinogram of the requested element accounting for
    absorption by the matrix defined by the matrix_map, and the geometry.
    Physical units used:
        matrix_map (g/cm^3)
        ma (cm^2/g)
        px_side (um)

    Parameters
    ----------
    event_type : string
        One of ['absorption', 'rayleigh', 'compton', 'fluoro'].
    p : Phantom2d object
        p.energy - incident beam photon energy (keV).
        p.um_per_px - length of one pixel of the map (um).
    anglelist : list of float
        Ordered list of sinogram projection angles in degrees.
    el : string, optional
        Name of element (e.g. 'Fe') used if projecting that element's
        fluorescence.
    show_progress : bool, optional
        Display progress as an updating percentage iff True.

    Returns
    -------
    array of float
        Sinogram of requested scattering or fluorescence.

    """
    assert event_type in ['absorption', 'rayleigh', 'compton', 'fluoro']

    sinogram = np.empty((p.cols, len(anglelist)))
    for i, angle in enumerate(anglelist):
        if show_progress:
            sys.stdout.write("\r{:.0%}".format(float(i) / len(anglelist)))
            sys.stdout.flush()

        increasing_ix = True   # Set True to accumulate cmam along increasing y
        n_map = irradiance_map(p, angle, n0=1.0, increasing_ix=increasing_ix)
        if event_type == 'absorption':
            if increasing_ix:
                sinogram[:, i] = np.log(n_map[0] / n_map[-1])
            else:
                sinogram[:, i] = np.log(n_map[-1] / n_map[0])
        else:
            e_map = emission_map(event_type, p, n_map, angle, el)
            sinogram[:, i] = e_map.sum(axis=0)
    return sinogram


def irradiance_map(p, angle, n0=1.0, increasing_ix=True):
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

    Returns
    -------
    2d ndarray of float
        The irradiance map.

    """
    # matrix_map = zero_outside_circle(p.el_maps['matrix'])

    # The linear absorption map mu0 = ma_M * mu_M + sum_k ( ma_k * mu_k )
    mu0 = brain.ma(p.energy) * p.el_maps['matrix']
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
    y, x = maia_d.yx(row, col)
    theta = pi / 2 + np.arctan(maia_d.d_mm / np.hypot(x, y))
    phi = np.arctan2(y, x)

    compound = brain.cp_brain_icru44  # elemental data for brain

    # Get the contribution to Rayleigh and Compton scattering from each
    # element in the matrix compound and sum these.
    ma = 0.0
    for el in compound:
        # Assuming propagation along z, coordinate system used by the DCSP_Rayl
        # and DCSP_Compt methods is shown here:
        # http://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/
        # 3D_Spherical.svg/200px-3D_Spherical.svg.png
        # i.e. spherical coordinates with polar angle theta, azimuthal angle phi
        z = compound[el].Z

        # Mass attenuation coefficients from cross-sections. See
        # http://physics.nist.gov/PhysRefData/XrayMassCoef/chap2.html
        # Units of the following expression:
        # elemental fraction by weight * differential mass attenuation coefft
        # unitless * cm2/g/sr
        if event_type == 'rayleigh':
            f = xrl.DCSP_Rayl
        else:
            f = xrl.DCSP_Compt
        ma += compound[el].fraction * f(z, p.energy, theta, phi)
    return ma


def fluoro_ma(p, el_z):
    """Return the fluorescence differential mass attenuation coefficient
    (cm2/g/sr) for the specified element. This needs to be multiplied later by
    the Maia-channel-dependent solid angle.

    Parameters
    ----------
    p : Phantom2d instance
        matrix plus elements.
    el_z : int
        Z of element el.

    Returns
    -------
    float
        Differential fluorescence mass attenuation coefficient (cm2/g/sr).

    """
    line = xrl.KA_LINE

    # CS_FluorLine_Kissel_Cascade is the XRF cross section Q_{i,YX} in Eq. (12)
    # of Schoonjans et al.
    # Units of the following expression:
    # differential mass attenuation coefft
    # 1.0/sr * cm2/g
    fluoro = 1.0 / 4 / pi * xrl.CS_FluorLine_Kissel_Cascade(el_z, line,
                                                              p.energy)
    return fluoro


def compton_scattered_energy(energy_in, row, col):
    """Energy of the Compton photons scattered into the direction of the Maia
    detector element.

    Parameters
    ----------
    energy_in : float
        Incident beam photon energy (keV).
    row, col: int
        Maia detector element indices.

    Returns
    -------
    float
        Energy of scattered photons (keV)

    """
    # Get polar scattering angle (theta) to detector element
    y, x = maia_d.yx(row, col)
    theta = pi / 2 + np.arctan(maia_d.d_mm / np.hypot(x, y))

    # energy_out = 1.0 / (1.0/energy_in +
    # (1 - np.cos(theta))*J_PER_KEV/sc.m_e/sc.c/sc.c)
    return xrl.ComptonEnergy(energy_in, theta)


def emission_map(event_type, p, n_map, angle, el=None):
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
        edge_map_r = rotate(edge_map, angle)
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
    filenames = ['{base}-{el}{ext}'.format(base='-'.join(f.split('-')[:-1]),
                                           el=el,
                                           ext=os.path.splitext(f)[1])
                 for f in glob.glob(pattern)]
    path, base = os.path.split(fnmatch.filter(filenames, pattern)[0])

    # Write sinogram (absorption map)
    s_filename = os.path.join(path, 's_{base}'.format(base=base))
    # append r, c, f, a to -matrix suffix so sinograms read in as unique images
    if '-matrix' in s_filename:
        s_filename = s_filename.replace('-matrix', '-matrix-' + event_type[0])
    write_tiff32(s_filename, im)


def project(p, event_type, anglesfile):
    """This is the entry point for the sinogram project script.
    Performs the projection at all specified angles and writes the resulting
    sinograms out as tiffs.

    Parameters
    ----------
    p : Phantom2d object
    event_type : string
        One of 'afrc'.
    anglesfile : string
        Path to a textfile containing ordered list of projection
        angles in degrees.

    """
    assert event_type in 'afrc'
    event_type = {'a': 'absorption', 'f': 'fluoro', 'r': 'rayleigh',
                  'c': 'compton'}[event_type]

    anglelist = np.loadtxt(anglesfile)
    if event_type == 'fluoro':
        # fluorescence sinogram
        for el in p.el_maps:
            if el == 'matrix':
                continue
            im = project_sinogram(event_type, p, anglelist, el)
            write_sinogram(im, p, event_type, el)
    else:
        # event_type is 'absorption', 'rayleigh' or 'compton'
        # Absorption, Rayleigh or Compton scattering sinogram of matrix
        im = project_sinogram(event_type, p, anglelist)
        write_sinogram(im, p, event_type)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import phantom

    BASE = r'R:\Science\XFM\GaryRuben\projects\TMM'
    DATA_DIR = os.path.join(BASE, r'work\data')
    os.chdir(DATA_DIR)

    anglesfile = os.path.join(BASE, r'work\data\angles.txt')

    p = phantom.Phantom2d(filename='phantom1_100*.tiff', um_per_px=10.0,
                          energy=15)

    anglelist = np.loadtxt(anglesfile, dtype=int)
    el = 'Ar'
    sinogram = project_sinogram('absorption', p, anglelist, el, show_progress=True)
    # sinogram = project_sinogram('rayleigh', p, anglelist, el, show_progress=True)

    # np.save('sino'+el, sinogram)
    imshow(sinogram, extent=[0, 360, 0, 99], aspect=2, cmap='cubehelix')
    plt.xlabel('rotation angle (deg)')
    plt.ylabel('x')
    plt.show()

    for event_type in 'arcf':
        event_name = {'a': 'absorption', 'f': 'fluoro', 'r': 'rayleigh',
                      'c': 'compton'}[event_type]
        print('\n' + event_name)
        project(p, event_type, anglesfile)
