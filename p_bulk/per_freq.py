import math
import numpy as np
import silicon_params as si


def per_freq(wf):

    s = wf.shape
    lc = si.a_Si / si.ab
    dV = (lc / s[0]) ** 3

    ub = np.fft.ifftshift(np.fft.fftn(np.real(wf))) * dV

    s = ub.shape
    x = np.linspace(0, s[0], num=s[0], endpoint=False) - s[0] // 2
    X, Y, Z = np.meshgrid(x, x, x)
    G = np.zeros((s[0] * s[1] * s[2], 6))

    G[:, 0] = X.flatten() * 2 * math.pi / lc
    G[:, 1] = Y.flatten() * 2 * math.pi / lc
    G[:, 2] = Z.flatten() * 2 * math.pi / lc
    G[:, 3] = np.real(ub.flatten())
    G[:, 4] = np.imag(ub.flatten())
    G[:, 5] = (np.abs(ub.flatten()))**2

    return G[G[:, 5].argsort()[::-1], :]