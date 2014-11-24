#!/usr/bin/env python

import os, sys

# Gain access to acsemble when running locally or as an imported module
# See http://stackoverflow.com/questions/2943847
if __name__ == "__main__" and __package__ is None:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(1, parent_dir)
    import acsemble
    __package__ = "acsemble"

import phantom
import projection
import config


config.parse()

p = phantom.Phantom2d(
    filename=config.filepattern,
    yamlfile=config.yamlfile,
    um_per_px=config.scale,
    energy=config.energy
)

projection.project(p, config.event_type)
