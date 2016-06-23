#!/usr/bin/python3

from __future__ import absolute_import, division, print_function
import six
import context
import xraylib as xrl
from acsemble import config
def setup_module():
    config.config_init()


def test_xraylib_installed():
    assert(type(xrl.AVOGNUM) is float)


if __name__ == "__main__" :
    import sys
    from numpy.testing import run_module_suite
    run_module_suite(argv=sys.argv)
