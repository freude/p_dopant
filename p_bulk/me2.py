import numpy as np
import logging as log
from abi_read import abi_read
from per_freq import per_freq
from smoother_2 import smoother_2, find
from pot_mat import pot_mat
from scipy.integrate import simps
from coordsys import CoordSys


def me2(k1, k2, flag):
    """
    If 'me', the function computes the integral
    over a coupling filtered potential

    If 'pot', the function computes the filtered potential itself

    :param k1:
    :param k2:
    :param flag:
    :return:
    """

    if np.array_equal(k1, k2) and flag == 'mes':

        ME = 0    # matrix element is usually needed only for nondiagonal term

    else:
        # ----- compute Fourier harmonics of the periodic Bloch fucntion -----

        # if (exist(strcat(pwd, '/dis_scr/G.mat'), 'file') == 2)
            # G = load(strcat(pwd, '/dis_scr/G.mat'));
        # G = G.G;
        # else
        num_cells = 1
        T = 50
        wf = np.conj(abi_read(num_cells, T, k1)) * (abi_read(num_cells, T, k2))
        # wf = abi_read(num_cells, T, k2)
        G = per_freq(wf)
        # save(strcat(pwd, 'dis_scr/G.mat'), 'G');
        # end;

        if np.array_equal(k1, k2):
            num_cells = 70
            coorsys = CoordSys(num_cells, 3, 'au')
            coorsys.set_origin_cells(num_cells / 2 + 1)
        else:
            num_cells = 16
            coorsys = CoordSys(num_cells, 10, 'au')
            coorsys.set_origin_cells(num_cells / 2 + 1)

        V, _, _, _ = pot_mat(coorsys, k1, k2)
        x = coorsys.x()

        # ----------------------- apply filter function ----------------------

        log.info('    Apply filter function...')
        R_tr = 4
        V1sm = smoother_2(x, G, k1, k2, R_tr)
        log.info('    Done!')

        if flag == 'pot':

            jj1 = np.argmin(np.abs(x + 0.65 * R_tr))
            jj2 = np.argmin(np.abs(x - 0.65 * R_tr))

            # delta = V1sm(jj2, jj2, jj2) - V(jj2, jj2, jj2);
            # V = V + delta;

            V[jj1:jj2, jj1:jj2, jj1:jj2] = -V1sm[jj1:jj2, jj1:jj2, jj1:jj2]
            V1sm = -V

    # ----------------------------------------------------------------

    M2 = 1

    if (((k1[find(k1)] > 0) and (k2[find(k2)] < 0)) or
            ((k2[find(k2)] > 0) and (k1[find(k1)] < 0))) and\
            (find(k1) == find(k2)):
        M1 = 1
    else:
        M1 = 1

    X, Y, Z = np.meshgrid(x, x, x)
    kk = -0*(k2 - k1)

    ME = 0

    if flag == 'mes':
        ME = simps(simps(simps(V1sm * M1 * M2 * np.exp(1j * (kk[0] * X + kk[1] * Y + kk[2] * Z)), x), x), x)
    elif flag == 'pot':
        ME = (x, V1sm)
    else:
        ValueError("flag has wrong value")

    return ME

