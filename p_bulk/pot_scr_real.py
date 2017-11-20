import numpy as np
import silicon_params as si


def pot_scr_real(r):

    au = 5.29e-11
    r = r * si.ab / au
    
    A = 1.175
    alpha = 0.757
    betha = 0.322
    gamma = 2.0442
    
    epsilon = 1 + A * si.eps1 * np.exp(-alpha * r) + (1 - A) * si.eps1 * np.exp(-betha * r) - np.exp(-gamma * r)
    
    return epsilon
