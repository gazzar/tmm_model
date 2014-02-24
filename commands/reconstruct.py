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
import reconstruction
import version


description = textwrap.dedent("""\
    {filename} -
    A commandline tool to reconstruct/backproject from a sinogram.
    Example use:
    python {filename} path/s_golosio*.tiff -a angles.txt
    """.format(filename=__file__))

parser = argparse.ArgumentParser(
    version = version.__version__,
    description = description,
    )
parser.add_argument('filepattern', action='store',
                    help=textwrap.dedent(
                    '''e.g. a-*.tiff reads a-Ca.tiff, a-Zn.tiff, etc.'''))
parser.add_argument('-m', '--method', action='store', default='f',
                    choices=['f','s'],
                    help=textwrap.dedent(
                    '''Algorithm; one of [f=fbp, s=SART] (default f)
                    fbp uses a standard ramp'''))
parser.add_argument('-n', type=int,
                    help=textwrap.dedent(
                    '''Number of iterations - ignored if not using SART'''))
parser.add_argument('-a', '--anglelist', action='store', default='angles.txt',
                    help=textwrap.dedent('''Filename of textfile containing
                                          list of projection angles, e.g.
                                          angles.txt'''))

args = vars(parser.parse_args())


filepattern = args['filepattern']
method = args['method']
anglelist = args['anglelist']
n = args['n']

p = phantom.Phantom2d(filename=filepattern)
reconstruction.reconstruct(p, method, anglelist)
