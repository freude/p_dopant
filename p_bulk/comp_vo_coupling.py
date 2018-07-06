import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import silicon_params as si
from read_env1 import read_env1, k2flag, read_amp_from_buffer
from read_env import read_env_amp
from me2 import me2
from compose_wf import compose_wf


def comp_vo_coupling(atomic_coords, path, bands, interpolators_dict=None):

    pth = os.getcwd()
    Nbands = np.size(bands)
    Nbands = 5

    # bands = np.array([1, 3]) - 1
    # Nbands = np.size(bands)

    x = np.linspace(-6, 6, 100)

    # try:
    #     with open(pth1, "r") as f:
    #         amp_x = pickle.load(f)
    # except EnvironmentError:
    #     amp_x, _ = read_env1(x, bands, path, kk[1, :], 0, np.array(atomic_coords))
    #     pickle.dump(amp_x, open(pth1, "wb"))
    #
    # try:
    #     with open(pth2, "r") as f:
    #         amp_y = pickle.load(f)
    # except EnvironmentError:
    #     amp_y, _ = read_env1(x, bands, path, kk[3, :], 0, np.array(atomic_coords))
    #     pickle.dump(amp_y, open(pth2, "wb"))
    #
    # try:
    #     with open(pth3, "r") as f:
    #         amp_z = pickle.load(f)
    # except EnvironmentError:
    #     amp_z, _ = read_env1(x, bands, path, si.kk[5, :], 0, np.array(atomic_coords))
    #     pickle.dump(amp_z, open(pth3, "wb"))

    reset_buf = False

    if not interpolators_dict:
        _, pth = k2flag(si.kk[1, :])
        pth = os.path.join(path, pth, 'amp_' + str(np.size(bands)) + '.pkl')
        amp_x, _ = read_amp_from_buffer(pth, reset=reset_buf)(read_env1)(x,
                                                                         bands,
                                                                         path,
                                                                         si.kk[1, :],
                                                                         0,
                                                                         np.array(atomic_coords))

        _, pth = k2flag(si.kk[3, :])
        pth = os.path.join(path, pth, 'amp_' + str(np.size(bands)) + '.pkl')
        amp_y, _ = read_amp_from_buffer(pth, reset=reset_buf)(read_env1)(x,
                                                                         bands,
                                                                         path,
                                                                         si.kk[3, :],
                                                                         0,
                                                                         np.array(atomic_coords))

        _, pth = k2flag(si.kk[4, :])
        pth = os.path.join(path, pth, 'amp_' + str(np.size(bands)) + '.pkl')
        amp_z, _ = read_amp_from_buffer(pth, reset=reset_buf)(read_env1)(x,
                                                                         bands,
                                                                         path,
                                                                         si.kk[5, :],
                                                                         0,
                                                                         np.array(atomic_coords))

        energy_x = np.loadtxt(os.path.join(path, 'v0/EE_0.dat'))[bands]
        energy_y = np.loadtxt(os.path.join(path, 'v1/EE_0.dat'))[bands]
        energy_z = np.loadtxt(os.path.join(path, 'v2/EE_0.dat'))[bands]

    else:

        d = atomic_coords[1][2] - atomic_coords[0][2]

        amp_x = np.zeros((len(atomic_coords), Nbands))
        amp_y = np.zeros((len(atomic_coords), Nbands))
        amp_z = np.zeros((len(atomic_coords), Nbands))

        energy_x = np.zeros(Nbands)
        energy_y = np.zeros(Nbands)
        energy_z = np.zeros(Nbands)

        for j1 in  xrange(Nbands):

            energy_x[j1] = interpolators_dict['x']['E'][j1](d)
            energy_y[j1] = interpolators_dict['y']['E'][j1](d)
            energy_z[j1] = interpolators_dict['z']['E'][j1](d)

            for j2 in xrange(len(atomic_coords)):

                amp_x[j2, j1] = interpolators_dict['x']['amp'][np.ravel_multi_index((j2, j1), amp_x.shape)](d)
                amp_y[j2, j1] = interpolators_dict['y']['amp'][np.ravel_multi_index((j2, j1), amp_y.shape)](d)
                amp_z[j2, j1] = interpolators_dict['z']['amp'][np.ravel_multi_index((j2, j1), amp_z.shape)](d)

    amp = np.stack((amp_z, amp_y, amp_x))
    amp[np.abs(amp) < 0.01 * np.max(np.abs(amp))] = 0.0
    energy = [energy_z, energy_y, energy_x]

    # energy = [-0.87 - 0.24];
    # energy = [-0.87, -0.24, -0.1, -0.01]
    # energy = [-0.84 - 0.29 - 0.23];
    # energy = [-1.21348, -1.20538, -0.735405, -0.50513, -0.447229, -0.446852, -0.427727, -0.355853, -0.343691, -0.343115]

    # ---------------------------------------------------------------------

    ME_a = -0.99 / 40
    ME_b = -1.72 / 40
    M = np.zeros((6 * Nbands, 6 * Nbands), dtype=complex)

    # -----------------------------------------------------------------------------

    MEs = np.zeros((6, 6), dtype=complex)

    for j1 in xrange(6):
        for j2 in xrange(6):
            if j1 != j2:
                if np.sum(np.abs(si.kk[j1, :] + si.kk[j2, :])) == 0:
                    MEs[j1, j2] = ME_a
                else:
                    MEs[j1, j2] = ME_b
            else:
                MEs[j1, j2] = 0

    # -----------------------------------------------------------------------------

    for j1 in xrange(6):
        for j2 in xrange(6):
            for jj1 in xrange(Nbands):
                for jj2 in xrange(Nbands):
                    if j1 == j2 and jj1 != jj2:
                        M[jj1 + Nbands * j1, jj2 + Nbands * j2] = 0
                    elif j1 == j2 and jj1 == jj2:
                        M[jj1 + Nbands * j1, jj2 + Nbands * j2] = energy[int(j1/2)][jj1]
                    else:
                        for j, coords in enumerate(atomic_coords):
                            M[jj1 + Nbands * j1, jj2 + Nbands * j2] += MEs[j1, j2] * \
                                amp[int(j1 / 2), j, jj1] * amp[int(j2 / 2), j, jj2] * \
                                np.exp(1j * np.inner(si.kk[j1, :] - si.kk[j2, :], np.array(coords)))
                            # print(np.inner(kk[j2, :] - kk[j1, :],  np.array(coords)))
                            # print(j1, j2, jj1, jj2)

    # for j1 in xrange(6 * Nbands):
    #     for j2 in xrange(6 * Nbands):
    #         if j1 != j2 and j1 > j2:
    #             M[j1, j2] = np.conj(M[j2, j1])

    # ----------------------------------------------------------------------------
    # -------------Overlap matrix < k2, i | k1, j > for envelopes ----------------
    # ----------------------------------------------------------------------------

    a, EigVec = linalg.eig(M)
    a = np.real(a) * si.E_Har / si.q * 1000   # convert to meV
    ind = np.argsort(a)
    a1 = a[ind]

    print(a1)

    EigVec1 = EigVec[:, ind]

    F = compose_wf(path, si.kk, bands, a1, EigVec1)
    #np.save('F',np.abs(F[1][61, :, :]))
    plt.imshow(np.abs(F[1][61, :, :]))
    # plt.savefig("out.png")
    plt.show()

    return a1, amp


if __name__ == '__main__':

    atomic_coords = [[0, 0, -1.5], [0, 0, 1.5]]
    comp_vo_coupling(atomic_coords)
