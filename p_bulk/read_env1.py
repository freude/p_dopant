import os
import pickle
import numpy as np
from scipy.integrate import simps
from invdisttree import Invdisttree


def k2flag(k1):

    if k1[0] != 0:
        if k1[0] > 0:
            flag = 'x'
        else:
            flag = '-x'

        pth = 'v0/'

    if k1[1] != 0:
        if k1[1] > 0:
            flag = 'y'
        else:
            flag = '-y'

        pth = 'v1/'

    if k1[2] != 0:
        if k1[2] > 0:
            flag = 'z'
        else:
            flag = '-z'

        pth = 'v2/'

    return flag, pth


def read_amp_from_buffer(pth, reset=False):
    def wrap(func):
        def _wrap(*args, **kwargs):

            if reset:
                pth1 = pth + '1'
            else:
                pth1 = pth

            try:
                with open(pth1, "r") as f:
                    amp = pickle.load(f)
                    matrix = None
            except EnvironmentError:
                amp, matrix = func(*args, **kwargs)
                pickle.dump(amp, open(pth, "wb"))

            return amp, matrix

        return _wrap
    return wrap


def read_env1(x, bands, path, k1, indi, coord):

    _, pth = k2flag(k1)        # defines a subdirectory where to look for envelope functions
    s = coord.shape
    Nimp = s[0]                # determines number of impurities form the coordinate array
    Nbands = np.size(bands)    # determines number of bands

    p1 = np.loadtxt(os.path.join(path, pth, 'ff_' + str(indi) + '.dat'))                   # load row data

    # parse the raw data file
    a = []
    for j in xrange(p1.shape[0]):
        if (p1[j, 0] == 111) and (p1[j, 1] == 111) and (p1[j, 2] == 111):
            a.append(j)

    amp = np.zeros((Nimp, Nbands))

    for jjj in xrange(Nbands):
        F1 = p1[a[bands[jjj]] + 1:a[bands[jjj] + 1] - 1, 3]
        X = p1[a[bands[jjj]] + 1:a[bands[jjj] + 1] - 1, 0]
        Y = p1[a[bands[jjj]] + 1:a[bands[jjj] + 1] - 1, 1]
        Z = p1[a[bands[jjj]] + 1:a[bands[jjj] + 1] - 1, 2]

        # X1 = X[(np.abs(X) < x0) * (np.abs(Y) < y0) * (np.abs(Z) < z0)]
        # Y1 = Y[(np.abs(X) < x0) * (np.abs(Y) < y0) * (np.abs(Z) < z0)]
        # Z1 = Z[(np.abs(X) < x0) * (np.abs(Y) < y0) * (np.abs(Z) < z0)]
        # F1 = F1[(np.abs(X) < x0) * (np.abs(Y) < y0) * (np.abs(Z) < z0)]

        Fq = Invdisttree(np.vstack((X, Y, Z)).T, F1)
        amp[:, jjj] = Fq(np.vstack((coord[:, 0], coord[:, 1], coord[:, 2])).T, nnear=11, eps=0, p=1)

        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        M = Fq(np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T, nnear=11, eps=0, p=1).reshape(X.shape)
        np.nan_to_num(M, copy=False)

        # normalize
        ME = simps(simps(simps(np.abs(M)**2, x), x), x)
        print(ME)
        M = M / np.sqrt(ME)

        # compute amplitudes in certain points
        amp[:, jjj] = Fq(np.vstack((coord[:, 0], coord[:, 1], coord[:, 2])).T, nnear=11, eps=0, p=1) / np.sqrt(ME)

        if amp[0, jjj] < 0:
            amp[:, jjj] = -amp[:, jjj]

    return amp, M


def read_env_amp(bands, path, k1, indi, coord):

    _, pth = k2flag(k1)        # defines a subdirectory where to look for envelope functions
    s = coord.shape
    Nimp = s[0]                # determines number of impurities form the coordinate array
    Nbands = np.size(bands)    # determines number of bands

    p1 = np.loadtxt(os.path.join(path, pth, 'ff_', str(indi), '.dat'))                   # load row data

    # parse the raw data file
    a = []
    for j in xrange(p1.shape[0]):
        if (p1[j, 0] == '111') and (p1[j, 1] == '111') and (p1[j, 2] == '111'):
            a.append(j)

    amp = np.zeros((Nimp, Nbands))

    for jjj in xrange(Nbands):
        F1 = p1[a[bands[jjj]] + 1:a[bands[jjj] + 1] - 1, 3]
        X = p1[a[bands[jjj]] + 1:a[bands[jjj] + 1] - 1, 0]
        Y = p1[a[bands[jjj]] + 1:a[bands[jjj] + 1] - 1, 1]
        Z = p1[a[bands[jjj]] + 1:a[bands[jjj] + 1] - 1, 2]
        Fq = Invdisttree(np.vstack((X, Y, Z)).T, F1)
        amp[:, jjj] = Fq(np.vstack((coord[:, 0], coord[:, 1], coord[:, 2])).T, nnear=11, eps=0, p=1)

    return amp
