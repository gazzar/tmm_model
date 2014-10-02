import pandas as pd
import numpy as np
from scipy import interpolate
import xraylib as xrl
from collections import namedtuple
import os

'''
# Gain access to tmm_model when running locally or as an imported module
# See http://stackoverflow.com/questions/2943847
if __name__ == "__main__" and __package__ is None:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(1, parent_dir)
    import tmm_model
    __package__ = "tmm_model"
'''

'''
Note: Density of wet brain matter (C_1.4 H_1 O_7.1 N_0.2) is 1.04 g/cm^3

The ICRU-44 brain tissue model is based on the following elemental mix
http://physics.nist.gov/PhysRefData/XrayMassCoef/tab2.html
<Z/A>       I (eV)   Density (g/cm3)    Composition (Z: fraction by weight)
0.55239     73.9     1.040E+00          H  1: 0.107000
                                        C  6: 0.145000
                                        N  7: 0.022000
                                        O  8: 0.712000
                                        Na 11: 0.002000
                                        P  15: 0.004000
                                        S  16: 0.002000
                                        Cl 17: 0.003000
                                        K  19: 0.003000
'''

Element = namedtuple('Element', 'label Z fraction')

# class Compound(object):
#     """
#     A compound containing one or more elements.
#
#     """
#     def __init__(self, density):
#         self.density = density
#         self.elements

cp_brain_icru44_density = 1.04  # g/cm3
cp_brain_icru44 = [
    Element('H', 1 , 0.107000),
    Element('C' ,6 , 0.145000),
    Element('N' ,7 , 0.022000),
    Element('O' ,8 , 0.712000),
    Element('Na',11, 0.002000),
    Element('P' ,15, 0.004000),
    Element('S' ,16, 0.002000),
    Element('Cl',17, 0.003000),
    Element('K' ,19, 0.003000),
    ]


def reweight_compound(compound, weight):
    """
    Takes a compound (list of Elements) and returns a matching list, with the
    fractions multiplied by the weight.

    Parameters
    ----------
    compound : list of Element namedtuples
    weight : float
        The weight factor (between 0 and 1) used to weight the entire compound

    Returns
    -------
    list of Element namedtuples

    """
    assert 0.0 <= weight <= 1.0
    return [el._replace(fraction=el.fraction * weight) for el in compound]


class brain_properties(object):
    """A class wrapping the NIST ICRU_44 data for brain tissue
    Example:
    b = brain_data()
    b.ma(13)

    """
    def __init__(self):
        self.cp_brain_icru44_density = cp_brain_icru44_density

    def ma(self, energy):
        return sum([el.fraction * xrl.CS_Total(el.Z, energy) for el in
                    cp_brain_icru44])

    @property
    def density(self):
        return self.cp_brain_icru44_density

    def get_weighted_brain_compound(self, weight):
        return reweight_compound(cp_brain_icru44, weight)


if __name__ == "__main__":
    b = brain_properties()
    print b.ma(1)
    print b.ma(1.01)
    print b.ma(1.07)
    print 
    print b.ma(1.073)
    print b.ma(2.145)
    print 
    print b.ma(2.15)
    print b.ma(2.47)
    print 
    print b.ma(2.475)
    print b.ma(2.82)
    print 
    print b.ma(2.83)
    print b.ma(3.6)
    print 
    print b.ma(3.61)
    print b.ma(21)

    # print b.brain_table.mu_cm2_g.values

#     import matplotlib.pyplot as plt
# #     xs = b.brain_table.Energy_MeV.values
#     xs = np.logspace(-3, np.log10(b.brain_table.Energy_MeV.values[-1]-1), 10000)
#     ys = np.array([b.ma(x*1000) for x in xs])
#
#     plt.loglog(xs, ys, '.-')
#     plt.show()

    print b.cp_brain_icru44_density
    print b.get_weighted_brain_compound(1)
    print b.get_weighted_brain_compound(0.5)
