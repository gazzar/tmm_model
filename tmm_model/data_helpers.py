import pandas as pd
import numpy as np
from scipy import interpolate
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

BRAIN_DATA = os.path.join(os.path.dirname(__file__), 'data',
                          'ICRU_44_brain_grey_white_matter.txt')

class brain_attenuation(object):
    """A class wrapping the NIST ICRU_44 data for brain tissue
    Example:
    b = brain_data()
    b.mu_on_rho(13)

    Note: Mass of wet brain matter (C_1.4 H_1 O_7.1 N_0.2) is 1.04 g/cm^3 

    """
    def __init__(self):
        self.brain_table = brain_table = pd.read_fwf(BRAIN_DATA)
#        self.brain_table.mu_cm2_g[2] = 3.7e3
        # Identify the discontinuities in the lookup table data that
        # correspond to K edges.
        cuts = brain_table.edges.notnull().nonzero()[0]
        self.cuts = np.hstack(([0], cuts.flat, len(brain_table.edges)-1))
        # A list of index 2-tuples defining piecewise-continuous intervals
        self.intervals = [tuple(i) for i in np.hstack(
                                            ([0], np.dstack((cuts,cuts)).flat,
                                             len(brain_table.edges)-1)
                                            ).reshape(-1,2)]
        # Do piecewise Hermite Polynomial interpolation on each interval.
        # Do this in log-log coordinates. 
        self.funcs = [interpolate.PchipInterpolator(
                          np.log(brain_table.Energy_MeV.values[low:high]),
                          np.log(brain_table.mu_cm2_g.values[low:high]))
                      for low, high in self.intervals]
        self.energy_min = self.brain_table.Energy_MeV.values.min()
        self.energy_max = self.brain_table.Energy_MeV.values.max()

    def mu_on_rho(self, energy):
        """Return mu/rho (in cm^2/g)
        See http://stackoverflow.com/questions/20941973/python-pandas-interpolate-with-new-x-axis

        Arguments:
        energy - energy (keV)
        
        Returns:
        mu/rho (in cm^2/g)

        """
        energy_MeV = energy/1000.0
        # Don't extrapolate
        assert(self.energy_min <= energy_MeV <= self.energy_max)

        # ix is the index of the data entry immediately above energy_MeV
        # ix2 is the index of the piecewise-continuous interval. This is the
        # index of the correct interpolating function.
        ix = (energy_MeV < self.brain_table.Energy_MeV).argmax()
        ix2 = (ix < self.cuts).argmax() - 1

        return np.exp(self.funcs[ix2](np.log(energy_MeV)))


if __name__ == "__main__":
    b = brain_attenuation()
    print b.mu_on_rho(1)
    print b.mu_on_rho(1.01)
    print b.mu_on_rho(1.07)
    print 
    print b.mu_on_rho(1.073)
    print b.mu_on_rho(2.145)
    print 
    print b.mu_on_rho(2.15)
    print b.mu_on_rho(2.47)
    print 
    print b.mu_on_rho(2.475)
    print b.mu_on_rho(2.82)
    print 
    print b.mu_on_rho(2.83)
    print b.mu_on_rho(3.6)
    print 
    print b.mu_on_rho(3.61)
    print b.mu_on_rho(21)

    # print b.brain_table.mu_cm2_g.values

    import matplotlib.pyplot as plt
#     xs = b.brain_table.Energy_MeV.values
    xs = np.logspace(-3, np.log10(b.brain_table.Energy_MeV.values[-1]-1), 10000)
    ys = np.array([b.mu_on_rho(x*1000) for x in xs])

    plt.loglog(xs, ys, '.-')
    plt.show()
