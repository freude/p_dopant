import silicon_params as si


def pot_scr(kk):

    au = 5.29e-11 
    kk = kk / si.ab * au
    
    A = 1.175
    alpha = 0.757
    betha = 0.322
    gamma = 2.044
    
    epsilon = A * (kk**2) / (kk**2+alpha**2) +\
              (1 - A) * (kk**2) / (kk**2 + betha**2) +\
              1 / si.eps1 * (gamma**2) / (kk**2+gamma**2)

    epsilon = epsilon*si.eps1

    return epsilon