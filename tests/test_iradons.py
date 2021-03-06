#!/usr/bin/python3

from __future__ import absolute_import, division, print_function
import six
import context
import unittest
import numpy as np
from acsemble import config
from acsemble import iradons

def setup_module():
    config.config_init()

if __name__ == "__main__" :
    import sys
    from numpy.testing import run_module_suite
    run_module_suite(argv=sys.argv)
