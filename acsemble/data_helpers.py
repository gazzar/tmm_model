#!/usr/bin/python3

from __future__ import absolute_import, division, print_function
from . import config

import logging
logger = logging.getLogger(__name__)

import xraylib as xrl
import numpy as np


class MatrixProperties(object):
    """A class exposing the matrix properties
    Example:
    p = Phantom2d(filename)
    b = MatrixProperties(p)
    b.ma(13)

    """
    def __init__(self, p):
        self.compounds = p._read_composition(p.yamlfile)
        # Get the index of the only compound in the elemental-makeup dictionary.
        # This will correspond to the matrix compound.
        self.compound_ix = [k for k, v in self.compounds.items()
                            if len(v[1]) > 1][0]
        # Now, get the matrix compound density and elemental-makeup dictionary.
        self.density, self.cp = self.compounds[self.compound_ix]
        assert np.isclose(sum(self.cp.values()), 1.0)

    def ma(self, energy):
        return sum([self.cp[el] * xrl.CS_Total(
            xrl.SymbolToAtomicNumber(el), energy) for el in self.cp])
