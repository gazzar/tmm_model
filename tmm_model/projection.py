#!/usr/bin/env python

# Copyright (c) 2014, Gary Ruben
# Released under the Modified BSD license
# See LICENSE


"""Projection class
Implements a class for 2D phantoms. The Golosio phantom is coded in terms of
geometry and composition data, with class methods supplied to instanciate it.
Alternatively, phantoms can be defined as a pair of files, one containing
geometry data as a greyscale bitmap and the other containing the composition
data structure defined in a yaml file.

"""

from __future__ import print_function
import sys, os
import numpy as np
from numpy import exp, pi
import scipy.constants as sc
from scipy.special import expm1
from skimage.transform import radon
from helpers import (write_tiff32, zero_outside_circle, zero_outside_mask,
                     rotate, imshow)
from data_helpers import brain_attenuation
import glob, fnmatch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xraylib as xrl

from maia import Maia


UM_PER_CM = 1e4
J_PER_KEV = 1e3 * sc.eV
deg_to_rad = lambda x: x / 180 * pi
rad_to_deg = lambda x: x * 180 / pi

brain = brain_attenuation()  # attenuation data object singleton
maia_d = Maia()  # Maia detector object singleton


def project_with_absorption(p, matrix_map, anglelist):
    """Generates the sinogram accounting for absorption by the element,
    or compound in the case of the matrix.
    Use the Mass Attenuation Coefficient data from NIST XAAMDI database
    http://physics.nist.gov/PhysRefData/XrayMassCoef/chap2.html

    Arguments:
    p - phantom object
        p.energy - incident beam photon energy (keV)
        p.um_per_px - length of one pixel of the map (um)
    matrix_map - square 2d float32 array of element or compound abundance
    anglelist - list of angles in degrees

    matrix_map (g cm^-3)
    ma units are cm^2 g^-1
    px_side (um)

    """
    sinogram = np.empty((matrix_map.shape[0], len(anglelist)), dtype=np.float32)
    for i, angle in enumerate(anglelist):
        im = rotate(matrix_map, -angle)
        sinogram[:, i] = (im.sum(axis=0) * brain.ma(p.energy) *
                          p.um_per_px / UM_PER_CM)
    return sinogram


def project_sinogram(event_type, p, anglelist, el=None, show_progress=False):
    """Generates the sinogram of the requested element accounting for
    absorption by the matrix defined by the matrix_map, and the geometry.

    Arguments:
    event_type - string, one of ['rayleigh', 'compton', 'fluoro']
    p - phantom object,
        p.energy - incident beam photon energy (keV).
        p.um_per_px - length of one pixel of the map (um).
    anglelist - ordered list of angles in degrees.
    el - string with name of element (e.g. 'Fe') when projecting an element's
        fluorescence.
    show_progress - boolean flag. Display progress via iff True.

    matrix_map (g/cm^3)
    ma (cm^2/g)
    px_side (um)

    """
    assert event_type in ['rayleigh', 'compton', 'fluoro']

    sinogram = np.empty((len(anglelist), p.cols))
    for i, angle in enumerate(anglelist):
        if show_progress:
            sys.stdout.write("\r{:.0%}".format(float(i) / len(anglelist)))
            sys.stdout.flush()
        i_map = illumination_map(p, angle, I0=1.0)
        e_map = emission_map(event_type, p, i_map, angle, el)
        sinogram[i] = e_map.sum(axis=0)
    return sinogram


def illumination_map(p, angle, I0=1.0):
    """Generates the image-sized map of intensity at each 2d pixel for a given
    angle accounting for absorption at the incident energy by the element,
    or compound in the case of the matrix.
    Use the Mass Attenuation Coefficient data from NIST XAAMDI database
    http://physics.nist.gov/PhysRefData/XrayMassCoef/chap2.html

    Arguments:
    p - phantom object
        p.energy - incident beam energy (keV)
        p.um_per_px - length of one pixel of the map (um)
    angle - angle in degrees
    I0 - incident intensity (default 1.0)

    matrix_map (g/cm3)
    ma (cm2/g)
    px_side (um)

    """
    matrix_map = zero_outside_circle(p.el_maps['matrix'])
    i_map = np.empty_like(matrix_map)

    im = rotate(matrix_map, -angle)
    ma_t = brain.ma(p.energy) * p.um_per_px / UM_PER_CM
    i_map[:, 0] = I0 * np.ones(i_map.shape[0])
    for i in range(im.shape[1] - 1):
        i_map[:, i + 1] = i_map[:, i] * exp(-im[:, i] * ma_t)
    return i_map


def scattering_ma(event_type, p, row, col):
    """Return the Rayleigh or Compton mass attenuation coefficients (cm2/g)
    for icru44 brain tissue with density described by the 'matrix' map of
    phantom p into the Maia detector element indexed by row, col.

    Arguments:
    event_type - string, one of ['rayleigh', 'compton']
    p - phantom instance (matrix plus elements)
    row, col - maia detector element indices

    Returns:
    The (rayleigh, compton) mass attenuation coefficient

    """
    assert event_type in ['rayleigh', 'compton']

    omega = maia_d.solid_angle(row, col)

    # Get spherical angles (polar theta & azimuthal phi) to detector element
    y, x = maia_d.yx(row, col)
    theta = pi / 2 + np.arctan(maia_d.d_mm / np.hypot(x, y))
    phi = np.arctan2(y, x)

    compound = brain.brain_icru44_composition  # elemental data for brain

    # Get the contribution to Rayleigh and Compton scattering from each
    # element in the matrix compound and sum these.
    ma = 0.0
    for el in compound:
        # Assuming propagation along z, coordinate system used by the DCSP_Rayl
        # and DCSP_Compt methods is shown here:
        # http://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/
        #        3D_Spherical.svg/200px-3D_Spherical.svg.png
        # i.e. spherical coordinates with polar angle theta, azimuthal angle phi
        Z = compound[el].Z

        # Mass attenuation coefficients from cross-sections. See
        # http://physics.nist.gov/PhysRefData/XrayMassCoef/chap2.html
        # Units of the following expression:
        # elemental fraction by weight * solid angle *
        #                                   differential mass attenuation coefft
        # unitless * sterad * cm2/g/sterad
        if event_type == 'rayleigh':
            f = xrl.DCSP_Rayl
        else:
            f = xrl.DCSP_Compt
        ma += (compound[el].fraction * omega * f(Z, p.energy, theta, phi))
    return ma


def fluoro_ma(p, el_Z, row, col):
    """Return the fluorescence mass attenuation coefficient (cm2/g)
    for the specified element and solid angle for the Maia detector
    element indexed by row, col. 

    Arguments:
    p - phantom instance (matrix plus elements)
    el_Z - Z of element el
    row, col - maia detector element indices

    Returns:
    fluorescence mass attenuation coefficient (cm2/g)

    """
    omega = maia_d.solid_angle(row, col)

    line = xrl.KA_LINE

    # CS_FluorLine_Kissel_Cascade is the XRF cross section Q_{i,YX} in Eq. (12)
    # of Schoonjans et al.
    # Units of the following expression:
    # solid angle * differential mass attenuation coefft
    # sterad/sterad * cm2/g
    fluoro = omega / 4 / pi * xrl.CS_FluorLine_Kissel_Cascade(el_Z, line,
                                                              p.energy)
    return fluoro


def compton_scattered_energy(energy_in, row, col):
    """Energy of the Compton photons scattered into the direction of the Maia
    detector element.

    Arguments:
    energy_in - incident beam photon energy (keV)
    row, col - maia detector element indices

    Returns:
    Energy of scattered photons (keV)

    """
    # Get polar scattering angle (theta) to detector element
    y, x = maia_d.yx(row, col)
    theta = pi / 2 + np.arctan(maia_d.d_mm / np.hypot(x, y))

    # energy_out = 1.0 / (1.0/energy_in +
    #                     (1 - np.cos(theta))*J_PER_KEV/sc.m_e/sc.c/sc.c)
    return xrl.ComptonEnergy(energy_in, theta)


def emission_map(event_type, p, i_map, angle, el=None):
    """Compute the maia-detector-shaped map of Rayleigh, Compton or K-edge
    fluorescence from the element map for a uniform incident intensity I0

    Arguments:
    event_type - string; one of ['rayleigh', 'compton', 'fluoro']
    p - phantom instance (matrix plus elements).
    i_map - 2d array of floats; map of incident intensity.
    angle - float; stage rotation angle (degrees).
    el - string; element to consider, e.g. 'Fe'.

    Returns:
    The accumulated flux in Maia detector row 7 for the requested edge

    """
    assert event_type in ['rayleigh', 'compton', 'fluoro']

    # imap contains the values of incident attenuated flux $F_\gamma$
    imap_r = rotate(i_map, -angle)

    # Get matrix map and the map for the requested element and rotate them to
    # the same angle as the intensity map:
    # These both need to be in registration
    matrix_map = zero_outside_circle(p.el_maps['matrix'])
    matrix_map_r = rotate(matrix_map, -angle)
    del matrix_map

    k_alpha_energy = -1
    if event_type == 'fluoro':
        edge_map = zero_outside_circle(p.el_maps[el])
        edge_map_r = rotate(edge_map, -angle)
        del edge_map

        # Get Z for the fluorescing element and check that its K_alpha is below
        # the incident energy.
        el_Z = xrl.SymbolToAtomicNumber(el)
        line = xrl.KA_LINE
        k_alpha_energy = xrl.LineEnergy(el_Z, line)
        assert k_alpha_energy < p.energy

    # 2d accumulator for results
    accumulator = np.empty((maia_d.shape[1], imap_r.shape[0]))

    # Iterate over maia detector elements in theta, i.e. maia columns
    # This should be parallelizable
    row = 7
    for col in range(maia_d.cols):
        # Get angle to rotate maps so that propagation toward detector plane
        # corresponds with direction to maia detector element
        delta_theta_yx_radian = maia_d.yx_angles_radian(row, col)
        delta_theta_x = rad_to_deg(delta_theta_yx_radian[1])

        # For every maia detector element, get the solid angle (parallelize?)
        # Orient the maps toward the maia element
        # TODO: check sign of delta: +ve or -ve?
        imap_rm = rotate(imap_r, -delta_theta_x)
        matrix_map_rm = rotate(matrix_map_r, -delta_theta_x)

        # Rotate the geometry so that the detector is on the bottom, so we can
        # integrate by stepping through the row indices.
        imap_rm = np.rot90(imap_rm)
        matrix_map_rm = np.rot90(matrix_map_rm)

        # mass attenuation coefft. (cm2/g)
        if event_type == 'fluoro':
            edge_map_rm = rotate(edge_map_r, -delta_theta_x)
            edge_map_rm = np.rot90(edge_map_rm)

            # Simulate fluorescence event:
            # Generate the initial fluorescence intensity
            # Start with the highest-energy edge or a mean factor accounting for
            # all edges first (move to other edges later?)
            mac = fluoro_ma(p, el_Z, row, col)

            # Scale for propagation over one voxel
            # *_mac_t = *_mac * p.um_per_px/UM_PER_CM
            #     (cm3/g) =   (cm2/g) * cm
            mac_t = mac * p.um_per_px / UM_PER_CM
            # Generate outgoing radiation.
            imap_rm *= -expm1(-edge_map_rm * mac_t)
            del edge_map_rm
        else:
            mac = scattering_ma(event_type, p, row, col)
            mac_t = mac * p.um_per_px / UM_PER_CM
            # Generate outgoing radiation.
            imap_rm *= -expm1(-matrix_map_rm * mac_t)

        # Now we've "evented," we use the mass attenuation coefficients of the
        # matrix for propagation with absorption out to the detector, but this
        # is at a new energy depending on the event type.
        energy = {
            'rayleigh': p.energy,
            'compton': compton_scattered_energy(p.energy, row, col),
            'fluoro': k_alpha_energy,
        }[event_type]
        mac_t = brain.ma(energy) * p.um_per_px / UM_PER_CM

        # Propagate all intensity to the detector, accumulating (+) and
        # absorbing [exp(-mu/rho rho t)] as we go.
        for i in range(imap_rm.shape[0] - 1):
            imap_rm[i + 1] += imap_rm[i] * exp(-matrix_map_rm[i] * mac_t)

        # Store the result for this detector element.
        accumulator[col] = imap_rm[-1]

    return accumulator


def write_sinogram(im, p, algorithm, el='matrix'):
    """Project and write sinogram for element map el
    Creates a filename by prepending s_ to the filename and appending r or c
    if writing the rayleigh or compton sinogram, respectively.

    Arguments:
    im - 2d float ndarray image
    p - phantom object
    algorithm - one of
        f - fluorescence
        r - rayleigh
        c - compton
    el - string; name of current element, e.g. 'Fe'.

    """
    # Get the filename that matches the glob pattern for this element
    # and prepend s_ to it
    pattern = p.filename
    filenames = ['{base}-{el}{ext}'
                     .format(base='-'.join(f.split('-')[:-1]),
                             el=el,
                             ext=os.path.splitext(f)[1])
                 for f in glob.glob(pattern)]
    path, base = os.path.split(fnmatch.filter(filenames, pattern)[0])

    # Write sinogram (absorption map)
    s_filename = os.path.join(path, 's_{base}'.format(base=base))
    # append r or c to -matrix suffix so sinograms read in as unique images
    if '-matrix' in s_filename:
        s_filename = s_filename.replace('-matrix', '-matrix' + algorithm)
    write_tiff32(s_filename, np.rot90(im, 3))  # rotate 90 deg cw


def project(p, algorithm, anglesfile):
    """This is the entry point for the sinogram project script.
    p - a phantom.Phantom2d instance.
    algorithm - one of
        f - fluorescence
        r - rayleigh
        c - compton
    anglesfile - path to a textfile containing ordered list of projection
        angles in degrees.

    """
    assert algorithm in 'frc'

    anglelist = np.loadtxt(anglesfile)
    event_type = {'f': 'fluoro', 'r': 'rayleigh', 'c': 'compton'}[algorithm]
    if algorithm == 'f':
        # fluorescence sinogram
        for el in p.el_maps:
            if algorithm == 'f' and el == 'matrix':
                continue
            im = project_sinogram(event_type, p, anglelist, el)
            write_sinogram(im, p, algorithm, el)
    else:
        # algorithm is 'r' or 'c'
        # Rayleigh or Compton scattering sinogram of matrix
        im = project_sinogram(event_type, p, anglelist)
        write_sinogram(im, p, algorithm)


if __name__ == '__main__':
    import phantom

    BASE = r'R:\Science\XFM\GaryRuben\git_repos\tmm_model'
    DATA_DIR = os.path.join(BASE, r'tmm_model\data')
    os.chdir(DATA_DIR)

    anglesfile = os.path.join(BASE, r'commands\angles.txt')

    '''
    # split golosio into elements+matrix
    p = phantom.Phantom2d(filename='golosio_100.png', matrix_elements='N O C')
    p.split_map(DATA_DIR)
    '''

    '''
    p = phantom.Phantom2d(filename='golosio*matrix.tiff', um_per_px=10.0,
                          energy=1)
    project(p, 'a', anglesfile)
    '''

    '''
    p = phantom.Phantom2d(filename='golosio*matrix.tiff', um_per_px=10.0,
                          energy=15)
    map_and_write(p, 15)
    '''

    '''
    p = phantom.Phantom2d(filename='golosio*.tiff', um_per_px=10.0, energy=15)

    angle = 0.0
    i_map = illumination_map(p, angle, I0=1.0)
    r_map = emission_map('rayleigh', p, i_map, angle)
    c_map = emission_map('compton', p, i_map, angle)

    plt.subplot(211)
    imshow(r_map)
    plt.subplot(212)
    imshow(c_map)
    plt.show()
    '''

    '''
    p = phantom.Phantom2d(filename='golosio*.tiff', um_per_px=10.0, energy=15)

    angle = 0.0
    i_map = illumination_map(p, angle, I0=1.0)
    el = 'Fe'
    e_map = fluoro_map(p, i_map, el, angle)

    imshow(e_map)
    plt.xlabel('angle')
    plt.show()
    '''

    p = phantom.Phantom2d(filename='phantom1_100*.tiff', um_per_px=10.0,
                          energy=15)

    anglelist = np.loadtxt(anglesfile, dtype=int)
    el = 'Fe'
    sinogram = project_fluoro(p, el, anglelist, show_progress=True)

    np.save('sinoFe', np.rot90(sinogram))
    imshow(np.rot90(sinogram), extent=[0,360,0,99], aspect=2, cmap='cubehelix')
    plt.xlabel('rotation angle (deg)')
    plt.ylabel('x')
    plt.show()

    print('fluoro')
    project(p, 'f', anglesfile)
    print('\nrayleigh')
    project(p, 'r', anglesfile)
    print('\ncompton')
    project(p, 'c', anglesfile)
