#!/usr/bin/python3

from __future__ import absolute_import, division, print_function
import six
import logging
import click

from . import config

logger = logging.getLogger(__name__)


@click.command()
@click.option('--config-path', '-c', default=None, help='Path to config file')
def main(config_path):
    config.config_init(config_path)
    print('element: {}'.format(config.element))
