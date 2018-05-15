import numpy as np
import silicon_params as si
from pot_for_ff import pot_for_ff


k0 = si.k0 * si.ab

kk = k0 * np.array([[1,  0,  0],
                    [-1, 0,  0],
                    [0,  1,  0],
                    [0, -1,  0],
                    [0,  0,  1],
                    [0,  0, -1]])

pot_for_ff(kk[0, :], kk[0, :], '1')
pot_for_ff(kk[1, :], kk[1, :], '2')
pot_for_ff(kk[2, :], kk[2, :], '3')
