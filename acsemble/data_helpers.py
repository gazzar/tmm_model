from __future__ import print_function
import xraylib as xrl


class MatrixProperties(object):
    """A class exposing the matrix properties
    Example:
    p = Phantom2d(filename)
    b = MatrixProperties(p)
    b.ma(13)

    """
    def __init__(self, p):
        self.compounds = p._read_composition(p.yamlfile)
        self.compound_ix = [k for k, v in self.compounds.items()
                            if len(v[1]) > 1][0]
        self.density, self.cp = self.compounds[self.compound_ix]

    def ma(self, energy):
        return sum([self.cp[el] * xrl.CS_Total(
            xrl.SymbolToAtomicNumber(el), energy) for el in self.cp])
