#!/usr/bin/python3

from __future__ import absolute_import, division, print_function
import six

from . import config

import logging
logger = logging.getLogger(__name__)
import click


@click.command()
@click.option('--config-path', '-c', default=None, help='Path to config file')
def main(config_path):
    if config_path is not None:
        config.update_from_file(config_path)
    print('element: {}'.format(config.element))
