#!/usr/bin/env python

import os, sys

# Gain access to acsemble when running locally or as an imported module
# See http://stackoverflow.com/questions/2943847
if __name__ == "__main__" and __package__ is None:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(1, parent_dir)
    import acsemble
    __package__ = "acsemble"

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
                                         golosio_100.png and writes files
                                         golosio_100-x.tiff'''))
parser.add_argument('yamlfile', action='store',
                    help='yamlfile e.g. golosio.yaml')

args = vars(parser.parse_args())

filename = args['filepattern']
yamlfile = args['yamlfile']
p = phantom.Phantom2d(filename=filename, yamlfile=yamlfile)
p.split_map(os.path.dirname(filename))
