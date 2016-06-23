#!/usr/bin/python3

from __future__ import absolute_import, division, print_function
import six
import context
from nose.tools import assert_raises
from acsemble import config
def setup_module():
    config.config_init()

def test_config_version():
    assert config.yaml_config_version == 0.1

def test_config_update():
    assert_raises(AttributeError, lambda: config.new_key == 'new_value')
    new_item = {'new_key': 'new_value'}
    config.update_config(new_item)
    assert config.new_key == 'new_value'


if __name__ == "__main__" :
    import sys
    from numpy.testing import run_module_suite
    run_module_suite(argv=sys.argv)
