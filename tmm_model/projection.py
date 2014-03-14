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

import os
import numpy as np
from scipy.constants import physical_constants as pc
from skimage.transform import radon
#from skimage.io import imread, imsave
from helpers import (write_tiff32, zero_outside_circle, zero_outside_mask,
                     rotate, imshow)
from data_helpers import brain_attenuation
import glob, fnmatch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xraylib as xrl

#import xraylib as xrl
import maia


UM_PER_CM = 1e4
deg_to_rad = lambda x: x/180*np.pi
rad_to_deg = lambda x: x*180/np.pi

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
        p.energy - incident beam energy (keV)
        p.um_per_px - length of one pixel of the map (um)
    matrix_map - square 2d float32 array of element or compound abundance
    anglelist - list of angles in degrees

    matrix_map (g cm^-3)
    mu_on_rho units are cm^2 g^-1
    px_side (um)

    """
    sinogram = np.empty((matrix_map.shape[0], len(anglelist)), dtype=np.float32)
    b = brain_attenuation()
    for i, angle in enumerate(anglelist):
        im = rotate(matrix_map, -angle)
        sinogram[:, i] = (im.sum(axis=0) * b.mu_on_rho(p.energy) *
                          p.um_per_px/UM_PER_CM)
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

    b = brain_attenuation()
    im = rotate(matrix_map, -angle)
    mu_on_rho_t = b.mu_on_rho(p.energy) * p.um_per_px/UM_PER_CM
    i_map[:,0] = I0 * np.ones(i_map.shape[0])
    for i in range(im.shape[1]-1):
        i_map[:,i+1] = i_map[:,i] * np.exp(-im[:,i] * mu_on_rho_t)
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
    theta = np.pi/2 + np.arctan(maia_d.d_mm/np.hypot(x, y))
    phi = np.arctan2(y, x)

    b = brain_attenuation()                     # attenuation data object
    compound = b.brain_icru44_composition       # elemental data for brain
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
    # but this includes outside the sample, so either:
    # 1. Get the Rayleigh scattering crosssection for air and use this outside
    #    the sample, or
    # 2. For now, just use the sample as a mask to zero the intensity outside
    #    the sample.
    # Generate a label array using scipy.ndimage and mask the parts
    # corresponding to the label value in the 0,0 pixel of the label array
    zero_outside_mask(imap, mask=p.el_maps['matrix'])
    imap_r = rotate(imap, -angle)

    # Get matrix map and rotate it to the same angle as the intensity map:
    # These need to be in registration
    matrix_map = zero_outside_circle(p.el_maps['matrix'])
    matrix_map_r = rotate(matrix_map, -angle)

    rayleigh = np.zeros((maia_d.shape[1], imap_r.shape[0]))    # 2d accumulator
    compton = np.zeros((maia_d.shape[1], imap_r.shape[0]))     # 2d accumulator

    # Iterate over maia detector elements in theta, i.e. maia columns
    # This should be parallelizable
    row = 7
    for col in range(maia_d.cols):
        # Get angle to rotate maps so that propagation toward detector plane
        # corresponds with direction to maia detector element
        delta_theta_yx_radian = maia_d.yx_angles_radian(row, col)
        delta_theta_x = rad_to_deg(delta_theta_yx_radian[1])

        # For every maia detector element, get the solid angle (parallel here?)
        # First orient the intensity map toward the maia element
        # TODO: check sign of delta: +ve or -ve?
        imap_rm = rotate(imap_r, -delta_theta_x)
        matrix_map_rm = rotate(matrix_map_r, -delta_theta_x)

        # Rotate the geometry so that the detector is on the bottom, so we can
        # integrate by stepping through the row indices.
        imap_rm = np.rot90(imap_rm)
        matrix_map_rm = np.rot90(matrix_map_rm)

        # make a copy of the intensity map for the Compton component
        imap_cm = imap_rm.copy()

        # Get the contribution to Rayleigh and Compton scattering from each
        # element in the matrix compound and sum these.
        # Get the Rayleigh and Compton mass attenuation coefficients (cm2/g)
        rayleigh_mu_on_rho, compton_mu_on_rho = \
            rayleigh_compton_mu_on_rho(p, maia_d, row, col)
    
        # Now, accumulate all intensity to the detector, with absorption
        # mu_on_rho_t = mu_on_rho * p.um_per_px/UM_PER_CM
        #     (cm3/g) =   (cm2/g) * cm
        rayleigh_mu_on_rho_t = rayleigh_mu_on_rho * p.um_per_px/UM_PER_CM
        compton_mu_on_rho_t = compton_mu_on_rho * p.um_per_px/UM_PER_CM
        for i in range(imap_rm.shape[0]-1):
#             imap_rm[i+1] += imap_rm[i] * np.exp(-matrix_map_rm[i] *
#                                                  rayleigh_mu_on_rho_t)
#             imap_cm[i+1] += imap_cm[i] * np.exp(-matrix_map_rm[i] *
#                                                  compton_mu_on_rho_t)
            imap_rm[i+1] += imap_rm[i] * np.exp(-matrix_map_rm[i] *
                                                 rayleigh_mu_on_rho_t)
            imap_cm[i+1] += imap_cm[i] * np.exp(-matrix_map_rm[i] *
                                                 compton_mu_on_rho_t)

        rayleigh[col] = imap_rm[-1]
        compton[col] = imap_cm[-1]

    return rayleigh, compton

'''
def rayleigh_map(p, imap, angle, maia_d):
    """Compute the maia-detector-shaped map of Rayleigh scattering from the
    element map for a uniform incident intensity I0.

    Arguments:
    p - phantom instance (matrix plus elements)
    imap - map of incident intensity
    angle - stage rotation angle (degrees)
    maia_d - maia detector instance

    """
    # imap contains the values of incident attenuated flux $F_\gamma$
    # but this includes outside the sample, so either:
    # 1. Get the Rayleigh scattering crosssection for air and use this outside
    #    the sample, or
    # 2. For now, just use the sample as a mask to zero the intensity outside
    #    the sample.
    # Generate a label array using scipy.ndimage and mask the parts
    # corresponding to the label value in the 0,0 pixel of the label array
    zero_outside_mask(imap, mask=p.el_maps['matrix'])
    imap_r = rotate(imap, -angle)

    # Get matrix map and rotate it to the same angle as the intensity map:
    # These need to be in registration
    matrix_map = zero_outside_circle(p.el_maps['matrix'])
    matrix_map_r = rotate(matrix_map, -angle)

    rayleigh = np.zeros(maia_d.shape[1])    # 1d accumulator

    # Iterate over maia detector elements in theta, i.e. maia columns
    # This should be parallelizable
    row = 7
    for col in range(maia_d.cols):
        # Get angle to rotate maps so that propagation toward detector plane
        # corresponds with direction to maia detector element
        delta_theta_yx_radian = maia_d.yx_angles_radian(row, col)
        delta_theta_x = rad_to_deg(delta_theta_yx_radian[1])

        # For every maia detector element, get the solid angle (parallel here?)
        # First orient the intensity map toward the maia element
        # TODO: check sign of delta: +ve or -ve?
        imap_rm = rotate(imap_r, -delta_theta_x)
        matrix_map_rm = rotate(matrix_map_r, -delta_theta_x)

        # Rotate the geometry so that the detector is on the bottom, so we can
        # integrate by stepping through the row indices.
        imap_rm = np.rot90(imap_rm)
        matrix_map_rm = np.rot90(matrix_map_rm)

        # Get the contribution to Rayleigh scattering from each element in the
        # matrix compound and sum these.
        # Get the Rayleigh mass attenuation coefficient (cm2/g)
        rayleigh_mu_on_rho, _compton_mu_on_rho = \
            rayleigh_compton_mu_on_rho(p, maia_d, row, col)
    
        # Now, accumulate all intensity to the detector, with absorption
        # mu_on_rho_t = mu_on_rho * p.um_per_px/UM_PER_CM
        #     (cm3/g) =   (cm2/g) * cm
        mu_on_rho_t = rayleigh_mu_on_rho * p.um_per_px/UM_PER_CM
        for i in range(imap_rm.shape[0]-1):
            imap_rm[i+1] += imap_rm[i] * np.exp(-matrix_map_rm[i] * mu_on_rho_t)

        """
        plt.subplot(121)
        plt.imshow(matrix_map_rm, vmin=None, vmax=None,
                   interpolation='nearest', origin='upper', cmap='gray')
        plt.subplot(122)
        im = plt.imshow(imap_rm, vmax=2, # vmin=0.88, vmax=imap_r.max(),
                   interpolation='nearest', origin='upper', cmap='gray')
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        plt.colorbar(im, cax=cax)
        plt.show()
        """

        rayleigh[col] = imap_rm[-1].sum()

    return rayleigh
'''

def edge_map(p, i_map, el, angle, maia_d):
    """Compute the maia-detector-shaped map of K-edge scattering from the
    element map for a uniform incident intensity I0

    I will use the symbols from Miller [1] in this code.
    $\tau$ photoelectric cross-section
    $\rho$ concentration of element
    $\omega$ ($K$) line fluorescence yield
    $B$ branching ratio for ($K_{\alpha 1}$) line
    $\Omega$ detector solid angle

    [1] US Nuclear Regulatory Commission "Passive Nondestructive Assay of
        Nuclear Materials" by Doug Reilly, Norbert Enselin, Hastings Smith Jr.
        and Sarah Kreiner (1991). NuREG/cR-5550 LA-UR-90-732

    Arguments:
    p - phantom instance (matrix plus elements)
    i_map - absorption (intensity) map
    el - element to consider, e.g. 'Fe' (string)
    angle - degrees
    maia_d - maia detector instance

    """
    # i_map contains the values of incident attenuated flux $F_\gamma$
    rho = rotate(p.el_map[el], -angle)
    i_map_rotated = rotate(i_map, -angle)
    rayleigh_map = np.zeros(maia_d.shape)   # 1d accumulator

    # get angle to rotate i_map so that propagation toward detector plane
    # corresponds with direction to maia detector element
    row, col = (0, 0)   # Just do this for element (0, 0) for the moment
    delta = maia_d.maia_data.yx_angles_radian(row, col)
    omega = maia_d.maia_data.solid_angle(row, col)

    # for every maia detector element, get the solid angle (parallel loop here?)
    # First orient the intensity map toward the maia element
    #TODO: check sign of delta: +ve or -ve?
    rotated_map = rotate(rho, -delta)
    
    # Generate the initial fluorescence intensity
    # Start with the highest-energy edge or a mean factor accounting for all
    # edges first (move to other edges later?)
    # Use constant 1.0 for the \tau \omega B term.
    # However, this implies that an energy downconversion takes place to another
    # edge energy (K_\alpha, or a mean edge energy)
    # fluoro yield, Rayleigh and Compton cross-sections

    #edge =
    edge = 1

    return edge


def project_and_write(p, el, el_map0, algorithm, anglelist):
    """Project and write sinogram for element map el

    Arguments:
    p - phantom object
    el - name of current element, e.g. 'Fe'
    el_map0 - numpy float32 array with elemental abundance of el 
    algorithm - one of 'r', 'g', 'a'
        r - conventional radon as implemented in scikits-image
        g - my attempt at a conventional radon transform
        a - my attempt at a radon transform with absorption through the matrix
    anglelist - list of angles in degrees

    """
    assert algorithm in ['r', 'g', 'a']

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

    # Get the filename that matches the glob pattern for this element
    # and prepend s_ to it
    pattern = p.filename
    filenames = ['{}-{}{}'.format(
                   '-'.join(f.split('-')[:-1]), el, os.path.splitext(f)[1])
                 for f in glob.glob(pattern)]
    path, base = os.path.split(fnmatch.filter(filenames, pattern)[0])

    # Write sinogram (absorption map)
    s_filename = os.path.join(path, 's_'+base)
    write_tiff32(s_filename, np.rot90(im,3))            # rotate 90 deg cw


def map_and_write(p, angle):
    """Project and write sinogram for element map el

    Arguments:
    p - phantom object
    angle - angle in degrees

    """
    im = absorption_map(p, angle)

    # Get the filename that matches the glob pattern for this element
    # and prepend s_ to it
    pattern = p.filename
    el = 'matrix'
    filenames = ['{}-{}{}'.format(
                   '-'.join(f.split('-')[:-1]), el, os.path.splitext(f)[1])
                 for f in glob.glob(pattern)]
    path, base = os.path.split(fnmatch.filter(filenames, pattern)[0])

    # Write sinogram (absorption map)
    s_filename = os.path.join(path, 'm_'+base)
    write_tiff32(s_filename, im)            # rotate 90 deg cw


def project(p, algorithm, anglesfile):
    """
    """
    anglelist = np.loadtxt(anglesfile)
    for el in p.el_maps:
        el_map0 = p.el_maps[el]
        project_and_write(p, el, el_map0, algorithm, anglelist)


if __name__ == '__main__':
    import phantom

    DATA_DIR = r'R:\Science\XFM\GaryRuben\git_repos\tmm_model\tmm_model\data'
    os.chdir(DATA_DIR)

    anglesfile = r'R:\Science\XFM\GaryRuben\git_repos\tmm_model\commands\angles.txt'

    '''
    # split golosio into elements+matrix
    p = phantom.Phantom2d(filename='golosio_100.png', matrix_elements='N O C')
    p.split_map(DATA_DIR)
    '''

    '''
    p = phantom.Phantom2d(filename='golosio*matrix.tiff', um_per_px=10.0, energy=1)
    project(p, 'a', anglesfile)
    '''

    '''
    p = phantom.Phantom2d(filename='golosio*matrix.tiff', um_per_px=10.0, energy=15)
    map_and_write(p, 15)
    '''

    p = phantom.Phantom2d(filename='golosio*.tiff', um_per_px=10.0, energy=15)

#     print brain_attenuation.brain_icru44_composition['O'].fraction

    angle = 0.0
    i_map = absorption_map(p, angle, I0=1.0)
    maia_d = maia.Maia()
#     muR, muC = rayleigh_compton_mu_on_rho(p, maia_d, row=6, col=0)
    r_map, c_map = rayleigh_compton_map(p, i_map, angle, maia_d)

    plt.subplot(211)
    imshow(r_map)
    plt.subplot(212)
    imshow(c_map)
    plt.show()