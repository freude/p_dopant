import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from read_env import read_env1
from abi_read import abi_read
from coordsys import CoordSys


def compose_wf(path, kk, bands, eigen_val, eigen_vec):
    num_cells = 14
    T = 9

    coorsys = CoordSys(num_cells, T, 'au')
    coorsys.set_origin_cells(num_cells / 2 + 1)
    x = coorsys.x()
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    s = X.shape
    Nbands = len(bands)
    M1 = np.zeros((3, Nbands, s[0], s[1], s[2]))

    for jj1 in xrange(3):
        print(jj1)
        M1[jj1, :, :, :, :] = read_env1(X, Y, Z, bands, path, kk[2 * jj1], 0)

    wf1 = np.zeros((6, s[0], s[1], s[2]), dtype=np.complex)

    for j in xrange(6):
        wf1[j, :, :, :] = abi_read(num_cells, T, kk[j, :]) * \
                          np.exp(1j * (kk[j, 0] * X + kk[j, 1] * Y + kk[j, 2] * Z))

    num_bs = 12

    bas_fun = []
    bas_fun.append(x)

    for jj in xrange(num_bs):
        F = np.zeros((s[0], s[1], s[2]), dtype=np.complex)
        for j2 in xrange(Nbands):
            for j3 in xrange(6):

                if j3 == 0 or j3 == 1:
                    jjj = 0
                if j3 == 2 or j3 == 3:
                    jjj = 1
                if j3 == 4 or j3 == 5:
                    jjj = 2

                F = F + eigen_vec[j2 + Nbands * j3, jj] * \
                    np.squeeze(M1[jjj, j2, :, :, :]) * \
                    np.squeeze(wf1[j3, :, :, :])

        ME = simps(simps(simps(np.abs(F) ** 2, x), x), x)
        if (np.max(np.abs(np.real(F))) > np.max(np.abs(np.imag(F)))):
            bas_fun.append(np.real(F) / np.sqrt(ME))
        else:
            bas_fun.append(np.imag(F) / np.sqrt(ME))

    return bas_fun


if __name__ == "__main__":

    F = compose_wf(kk, bands, eigen_val, eigen_vec)
    plt.imshow(F[0])
    plt.draw()
    plt.savefig("out.png")
