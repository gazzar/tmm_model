#!/usr/bin/env python

# Copyright (c) 2014, Gary Ruben
# Released under the Modified BSD license
# See LICENSE

import os, sys

# Gain access to tmm_model when running locally or as an imported module
# See http://stackoverflow.com/questions/2943847
if __name__ == "__main__" and __package__ is None:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(1, parent_dir)
    import tmm_model
    __package__ = "tmm_model"

import textwrap
import argparse
import phantom
import projection
import version


description = textwrap.dedent("""\
    {filename} -
    A commandline tool to generate a sinogram from a tiff or tiffs.
    Examples:
    python {filename} -a angles.txt path/golosio*.tiff
    python {filename} -t f -e 12.0 -a angles.txt path/golosio*.tiff
    """.format(filename=__file__))

parser = argparse.ArgumentParser(
    version = version.__version__,
    description = description,
    )
parser.add_argument('filepattern', action='store',
                    help=textwrap.dedent('''e.g. a-*.tiff reads
                                          a-Ca.tiff, a-Zn.tiff, etc.'''))
parser.add_argument('-t', '--algorithm', action='store', default='r',
                    choices=['f','r','c'],
                    help=textwrap.dedent(
                    '''algorithm {f=fluoro, r=rayleigh(default), c=compton}'''))
parser.add_argument('-a', '--anglelist', action='store', default='angles.txt',
                    help=textwrap.dedent('''filename of textfile containing
                                          list of projection angles, e.g.
                                          angles.txt'''))
parser.add_argument('-s', '--scale', type=float, default=10.0,
                    help='scale (um/px) (default 10.0)')
parser.add_argument('-e', '--energy', type=float, default=15.0,
                    help='energy (keV) (default 15.0)')

args = vars(parser.parse_args())


filepattern = args['filepattern']
algorithm = args['algorithm']
anglelist = args['anglelist']
scale = args['scale']
energy = args['energy']

p = phantom.Phantom2d(filename=filepattern, um_per_px=scale, energy=energy)
projection.project(p, algorithm, anglelist)
