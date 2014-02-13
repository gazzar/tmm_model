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
    Example use:
    python {filename} path/golosio*.tiff -a angles.txt
    """.format(filename=__file__))

parser = argparse.ArgumentParser(
    version = version.__version__,
    description = description,
    )
parser.add_argument('filepattern', action='store',
                    help=textwrap.dedent('''e.g. a-*.tiff reads
                                          a-Ca.tiff, a-Zn.tiff, etc.'''))
parser.add_argument('--algorithm', action='store', default='r',
                    help=textwrap.dedent('algorithm (default r=radon)'))
parser.add_argument('-a', '--anglelist', action='store', default='angles.txt',
                    help=textwrap.dedent('''filename of textfile containing
                                          list of projection angles, e.g.
                                          angles.txt'''))

args = vars(parser.parse_args())


filepattern = args['filepattern']
algorithm = args['algorithm']
anglelist = args['anglelist']

p = phantom.Phantom2d(filename=filepattern)
projection.project(p, algorithm, anglelist)
