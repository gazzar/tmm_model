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
import version
import re


description = textwrap.dedent("""\
    pngtotiff -
    A commandline tool to convert a phantom defined as a .png and .yaml
    combination to corresponding tiffs.
    """)

parser = argparse.ArgumentParser(
    version = version.__version__,
    description = description,
    )
parser.add_argument('filepattern', action='store',
                    help=textwrap.dedent('''filepattern e.g. golosio_100 reads
                                          golosio_100.png and golosio.yaml and
                                          writes files golosio_100-x.tiff'''))
parser.add_argument('-m', '--matrix', action='store', default='',
                    help=textwrap.dedent('define the matrix elements\
                                          e.g. NOHC, SiN'))

args = vars(parser.parse_args())

# get the list of matrix elements as a space-delimited string
matrix_elements = args['matrix']
matrix_elements = re.sub(r'([a-z]*)([A-Z])',r'\1 \2', matrix_elements).strip()

filename = args['filepattern']
p = phantom.Phantom2d(filename=filename, matrix_elements=matrix_elements)
p.split_map(os.path.dirname(filename))
