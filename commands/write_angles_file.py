import numpy as np

a = np.arange(0, 360, 3)
np.savetxt('angles.txt', a, fmt='%1.3g')