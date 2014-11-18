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
