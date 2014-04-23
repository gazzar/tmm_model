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

import maia


UM_PER_CM = 1e4
J_PER_KEV = 1e3 * sc.eV
deg_to_rad = lambda x: x/180*pi
rad_to_deg = lambda x: x*180/pi

brain = brain_attenuation()  # attenuation data object singleton


def g_radon(im, anglelist):
    """My attempt at a radon transform. I'll assume the "circle=True"
    condition that skimage makes, i.e. I'm ignoring what goes on outside the
    circular area whose diameter is equal to the side of the square array.

    Arguments:
    im - square 2d float32 array
    anglelist - list of angles in degrees

    """
    sinogram = np.empty((im.shape[0], len(anglelist)), dtype=np.float32)
    for i, angle in enumerate(anglelist):
        im_r = rotate(im, -angle)
        sinogram[:, i] = im_r.sum(axis=0)

    return sinogram


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
    mu_on_rho units are cm^2 g^-1
    px_side (um)

    """
    sinogram = np.empty((matrix_map.shape[0], len(anglelist)), dtype=np.float32)
    for i, angle in enumerate(anglelist):
        im = rotate(matrix_map, -angle)
        sinogram[:, i] = (im.sum(axis=0) * brain.mu_on_rho(p.energy) *
                          p.um_per_px / UM_PER_CM)
    return sinogram


def project_fluoro(p, el, anglelist, show_progress=False):
    """Generates the sinogram of the requested element accounting for
    absorption by the matrix defined by the matrix_map, and the geometry.

    Arguments:
    p - phantom object
        p.energy - incident beam photon energy (keV)
        p.um_per_px - length of one pixel of the map (um)
    el - string with name of element, e.g. 'Fe'
    anglelist - ordered list of angles in degrees
    show_progress - boolean flag. Display progress via iff True

    matrix_map (g/cm^3)
    mu_on_rho (cm^2/g)
    px_side (um)

    """
    sinogram = np.empty((len(anglelist), p.cols))
    maia_d = maia.Maia()
    for i, angle in enumerate(anglelist):
        if show_progress:
            sys.stdout.write("\r{:.0%}".format(float(i)/len(anglelist)))
            sys.stdout.flush()
        i_map = absorption_map(p, angle, I0=1.0)
        e_map = fluoro_map(p, i_map, el, angle, maia_d)
        sinogram[i] = e_map.sum(axis=0)
    return sinogram


def absorption_map(p, angle, I0=1.0):
    """Generates the image-sized map of intensity at each 2d pixel for a given
    angle accounting for absorption by the element, or compound in the case of
    the matrix.
    Use the Mass Attenuation Coefficient data from NIST XAAMDI database
    http://physics.nist.gov/PhysRefData/XrayMassCoef/chap2.html

    Arguments:
    p - phantom object
        p.energy - incident beam energy (keV)
        p.um_per_px - length of one pixel of the map (um)
    angle - angle in degrees
    I0 - incident intensity (default 1.0)

    matrix_map (g/cm3)
    mu_on_rho (cm2/g)
    px_side (um)

    """
    matrix_map = zero_outside_circle(p.el_maps['matrix'])
    i_map = np.empty_like(matrix_map)

    im = rotate(matrix_map, -angle)
    mu_on_rho_t = brain.mu_on_rho(p.energy) * p.um_per_px / UM_PER_CM
    i_map[:, 0] = I0 * np.ones(i_map.shape[0])
    for i in range(im.shape[1] - 1):
        i_map[:, i + 1] = i_map[:, i] * exp(-im[:, i] * mu_on_rho_t)
    return i_map


def rayleigh_compton_mu_on_rho(p, maia_d, row, col):
    """Return the Rayleigh and Compton mass attenuation coefficients (cm2/g)
    for icru44 brain tissue with density described by the 'matrix' map of
    phantom p into the Maia detector element indexed by row, col. 
 
    Arguments:
    p - phantom instance (matrix plus elements)
    maia_d - maia detector object instance
    row, col - maia detector element indices

    Returns:
    A pair (rayleigh, compton) of mass attenuation coefficients

    """
    omega = maia_d.solid_angle(row, col)

    # Get spherical angles (polar theta & azimuthal phi) to detector element
    y, x = maia_d.yx(row, col)
    theta = pi / 2 + np.arctan(maia_d.d_mm / np.hypot(x, y))
    phi = np.arctan2(y, x)

    compound = brain.brain_icru44_composition  # elemental data for brain

    # Get the contribution to Rayleigh and Compton scattering from each
    # element in the matrix compound and sum these.
    rayleigh = 0.0
    compton = 0.0
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
        rayleigh += (compound[el].fraction * omega *
                     xrl.DCSP_Rayl(Z, p.energy, theta, phi))
        compton += (compound[el].fraction * omega *
                    xrl.DCSP_Compt(Z, p.energy, theta, phi))
    return rayleigh, compton


def fluoro_mass_atten(p, el_Z, maia_d, row, col):
    """Return the fluorescence mass attenuation coefficient (cm2/g)
    for the specified element and solid angle for the Maia detector
    element indexed by row, col. 

    Arguments:
    p - phantom instance (matrix plus elements)
    el_Z - Z of element el
    maia_d - maia detector object instance
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


def compton_scattered_energy(energy_in, maia_d, row, col):
    """Energy of the Compton photons scattered into the direction of the Maia
    detector element.

    Arguments:
    energy_in - incident beam photon energy (keV)
    maia_d - maia detector object instance
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


# noinspection PyNoneFunctionAssignment
def rayleigh_compton_map(p, imap, angle, maia_d):
    """Compute the Rayleigh and Compton scattering contributions into a row of
    Maia detector elements from the tissue matrix absorption map
    p.el_maps['matrix'] given an incident intensity map imap.

    Arguments:
    p - phantom instance (matrix plus elements)
    imap - map of incident intensity
    angle - stage rotation angle (degrees)
    maia_d - maia detector instance

    Returns:
    The accumulated flux in Maia detector row 7
    (rayleigh, compton)

    """

    # imap contains the values of incident attenuated flux $F_\gamma$
    imap_r = rotate(imap, -angle)

    # Get matrix map and rotate it to the same angle as the intensity map:
    # These need to be in registration
    matrix_map = zero_outside_circle(p.el_maps['matrix'])
    matrix_map_r = rotate(matrix_map, -angle)
    del matrix_map

    # 2d accumulators for results
    rayleigh = np.empty((maia_d.shape[1], imap_r.shape[0]))
    compton = np.empty((maia_d.shape[1], imap_r.shape[0]))

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

        # Make a copy of the intensity map for the Compton component.
        imap_cm = imap_rm.copy()

        # Simulate scattering event:
        # Get intensity scattered toward Maia element using scattering cross
        # sections provided by xraylib. These are in the form of mass
        # attenuation coefficients, so the next steps propagate through one
        # voxel to determine the scattered intensity.

        # Rayleigh and Compton mass attenuation coefficients (cm2/g)
        rayleigh_mu_on_rho, compton_mu_on_rho = \
            rayleigh_compton_mu_on_rho(p, maia_d, row, col)

        # Scale for propagation over one voxel
        # *_mu_on_rho_t = *_mu_on_rho * p.um_per_px/UM_PER_CM
        #     (cm3/g) =   (cm2/g) * cm
        rayleigh_mu_on_rho_t = rayleigh_mu_on_rho * p.um_per_px / UM_PER_CM
        compton_mu_on_rho_t = compton_mu_on_rho * p.um_per_px / UM_PER_CM

        # Scatter, updating Rayleigh and Compton intensity maps.
        imap_rm *= -expm1(-matrix_map_rm * rayleigh_mu_on_rho_t)
        imap_cm *= -expm1(-matrix_map_rm * compton_mu_on_rho_t)

        # Now we've scattered, we use the mass attenuation coefficients of the
        # matrix for propagation with absorption out to the detector.
        # Rayleigh scattering intensity is evaluated at the beam energy whereas
        # Compton is at the reduced energy determined by the scattering angle to
        # the Maia detector element.

        # Rayleigh:
        mu_on_rho_t = brain.mu_on_rho(p.energy) * p.um_per_px / UM_PER_CM

        # Compton:
        compton_energy = compton_scattered_energy(p.energy, maia_d, row, col)
        mu_on_rho_compt_t = brain.mu_on_rho(
            compton_energy) * p.um_per_px / UM_PER_CM

        # Propagate all intensity to the detector, accumulating (+) and
        # absorbing [exp(-mu/rho rho t)] as we go.
        for i in range(imap_rm.shape[0] - 1):
            imap_rm[i + 1] += imap_rm[i] * exp(-matrix_map_rm[i] * mu_on_rho_t)
            imap_cm[i + 1] += imap_cm[i] * exp(-matrix_map_rm[i] *
                                               mu_on_rho_compt_t)

        # Store the result for this detector element.
        rayleigh[col] = imap_rm[-1]
        compton[col] = imap_cm[-1]

    return rayleigh, compton


def fluoro_map(p, imap, el, angle, maia_d):
    """Compute the maia-detector-shaped map of K-edge fluorescence from the
    element map for a uniform incident intensity I0

    Arguments:
    p - phantom instance (matrix plus elements)
    imap - map of incident intensity
    el - element to consider, e.g. 'Fe' (string)
    angle - stage rotation angle (degrees)
    maia_d - maia detector instance

    Returns:
    The accumulated flux in Maia detector row 7 for the requested edge

    """
    # imap contains the values of incident attenuated flux $F_\gamma$
    imap_r = rotate(imap, -angle)

    # Get matrix map and the map for the requested element and rotate them to
    # the same angle as the intensity map:
    # These both need to be in registration
    matrix_map = zero_outside_circle(p.el_maps['matrix'])
    matrix_map_r = rotate(matrix_map, -angle)
    del matrix_map
    edge_map = zero_outside_circle(p.el_maps[el])
    edge_map_r = rotate(edge_map, -angle)
    del edge_map

    # Get Z for the fluorescing element and check that its K_alpha is below the
    # incident energy.
    el_Z = xrl.SymbolToAtomicNumber(el)
    line = xrl.KA_LINE
    k_alpha_energy = xrl.LineEnergy(el_Z, line)
    assert k_alpha_energy < p.energy

    # 2d accumulator for results
    edge = np.empty((maia_d.shape[1], imap_r.shape[0]))

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
        edge_map_rm = rotate(edge_map_r, -delta_theta_x)

        # Rotate the geometry so that the detector is on the bottom, so we can
        # integrate by stepping through the row indices.
        imap_rm = np.rot90(imap_rm)
        matrix_map_rm = np.rot90(matrix_map_rm)
        edge_map_rm = np.rot90(edge_map_rm)

        # Simulate fluorescence event:
        # Generate the initial fluorescence intensity
        # Start with the highest-energy edge or a mean factor accounting for all
        # edges first (move to other edges later?)

        # Fluorescence crosssection (cm2/g)
        fluoro_mu_on_rho = fluoro_mass_atten(p, el_Z, maia_d, row, col)

        # Maybe I should also get the crosssections for all the brain tissue
        # matrix elements here and track their K_alpha emissions along with the
        # element of interest - not for the moment.

        # Scale for propagation over one voxel
        # *_mu_on_rho_t = *_mu_on_rho * p.um_per_px/UM_PER_CM
        #     (cm3/g) =   (cm2/g) * cm
        fluoro_mu_on_rho_t = fluoro_mu_on_rho * p.um_per_px / UM_PER_CM

        # Fluoresce, updating the fluorescence intensity map.
        imap_rm *= -expm1(-edge_map_rm * fluoro_mu_on_rho_t)

        # Now we've fluoresced, we use the mass attenuation coefficients of the
        # matrix for propagation with absorption out to the detector, but this
        # is at the K_alpha energy of the fluorescing element.
        mu_on_rho_t = brain.mu_on_rho(k_alpha_energy) * p.um_per_px / UM_PER_CM

        # Propagate all intensity to the detector, accumulating (+) and
        # absorbing [exp(-mu/rho rho t)] as we go.
        for i in range(imap_rm.shape[0] - 1):
            imap_rm[i + 1] += imap_rm[i] * exp(-matrix_map_rm[i] * mu_on_rho_t)

        # Store the result for this detector element.
        edge[col] = imap_rm[-1]

    return edge


def project_and_write(p, el, algorithm, anglelist):
    """Project and write sinogram for element map el
    Prepends s_ to the filename.

    Arguments:
    p - phantom object
    el - name of current element, e.g. 'Fe'
    algorithm - one of 'r', 'g', 'a', 'f'
        r - conventional radon as implemented in scikits-image
        g - my attempt at a conventional radon transform
        a - my attempt at a radon transform with absorption through the matrix
        f - fluorescence
    anglelist - list of angles in degrees

    """
    assert algorithm in ['r', 'g', 'a', 'f']

    el_map0 = p.el_maps[el]
    im = zero_outside_circle(el_map0)
    if algorithm == 'r':
        # conventional Radon transform
        im = radon(im, anglelist, circle=True)

    if algorithm == 'g':
        # my conventional Radon transform
        im = g_radon(im, anglelist)

    if algorithm == 'a':
        # my Radon transform with absorption
        im = project_with_absorption(p, im, anglelist)

    if algorithm == 'f':
        # fluorescence sinogram
        im = project_fluoro(p, el, anglelist)

    if algorithm == 'c':
        # rayleigh and compton scattering sinograms
        im = rayleigh_compton_map()

    # Get the filename that matches the glob pattern for this element
    # and prepend s_ to it
    pattern = p.filename
    filenames = ['{}-{}{}'.format('-'.join(f.split('-')[:-1]), el, os.path
                                  .splitext(f)[1]) for f in glob.glob(pattern)]
    path, base = os.path.split(fnmatch.filter(filenames, pattern)[0])

    # Write sinogram (absorption map)
    s_filename = os.path.join(path, 's_' + base)
    write_tiff32(s_filename, np.rot90(im, 3))  # rotate 90 deg cw


def map_and_write(p, angle):
    """Project and write sinogram for element map el.
    Prepends m_ to the filename.

    Arguments:
    p - phantom object
    angle - angle in degrees

    """
    im = absorption_map(p, angle)

    # Get the filename that matches the glob pattern for this element
    # and prepend m_ to it
    pattern = p.filename
    el = 'matrix'
    filenames = ['{}-{}{}'.format('-'.join(f.split('-')[:-1]), el, os.path
                                  .splitext(f)[1]) for f in glob.glob(pattern)]
    path, base = os.path.split(fnmatch.filter(filenames, pattern)[0])

    # Write sinogram (absorption map)
    s_filename = os.path.join(path, 'm_' + base)
    write_tiff32(s_filename, im)  # rotate 90 deg cw


def project(p, algorithm, anglesfile):
    """
    """
    anglelist = np.loadtxt(anglesfile)
    for el in p.el_maps:
        if algorithm=='f' and el=='matrix':
            continue
        project_and_write(p, el, algorithm, anglelist)


if __name__ == '__main__':
    import phantom

    DATA_DIR = r'R:\Science\XFM\GaryRuben\git_repos\tmm_model\tmm_model\data'
    os.chdir(DATA_DIR)

    anglesfile = \
        r'R:\Science\XFM\GaryRuben\git_repos\tmm_model\commands\angles_small' \
        r'.txt'

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
    i_map = absorption_map(p, angle, I0=1.0)
    maia_d = maia.Maia()
    r_map, c_map = rayleigh_compton_map(p, i_map, angle, maia_d)

    plt.subplot(211)
    imshow(r_map)
    plt.subplot(212)
    imshow(c_map)
    plt.show()
    '''

    '''
    p = phantom.Phantom2d(filename='golosio*.tiff', um_per_px=10.0, energy=15)

    angle = 0.0
    i_map = absorption_map(p, angle, I0=1.0)
    maia_d = maia.Maia()
    el = 'Fe'
    e_map = fluoro_map(p, i_map, el, angle, maia_d)

    imshow(e_map)
    plt.xlabel('angle')
    plt.show()
    '''

    p = phantom.Phantom2d(filename='golosio*.tiff', um_per_px=10.0, energy=15)

    '''
    anglelist = np.loadtxt(anglesfile, dtype=int)
    el = 'Fe'
    sinogram = project_fluoro(p, el, anglelist, show_progress=True)

    np.save('sinoFe', np.rot90(sinogram))
    imshow(np.rot90(sinogram), extent=[0,360,0,99], aspect=2, cmap='coolwarm')
    plt.xlabel('rotation angle (deg)')
    plt.ylabel('x')
    plt.show()
    '''
    project(p, 'f', anglesfile)
