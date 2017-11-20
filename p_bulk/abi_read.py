import os
import numpy as np
import logging as log
from scipy.integrate import simps
from invdisttree import Invdisttree
import silicon_params as si
from read_cube import read_cube


def transform_to_uc(wf1, L):
    """
    The function converts a wave functions computed on a grid in a primitive cell to
    the wave functions specified on a user-defined grid in a unit cell.

    :param wf1:  3D real-space wave function computed by ABINIT for a primitive cell
    :param L:    number of points along each of dimenssions
    :return:     3D real-space wave function computed by ABINIT for a unit cell
    """

    a0 = 0.5431
    num_points = wf1.shape[0]       # number of points along each of dimenssions in wf1 array
    num_cells = 3                   # number of primitive cells needed to form unit cell

    xx = np.linspace(0.0, num_cells, num_points*num_cells, endpoint=False) - 1.0
    x1, y1, z1 = np.meshgrid(xx, xx, xx)

    wf = np.zeros((num_cells*num_points, num_cells*num_points, num_cells*num_points))

    for j1 in xrange(num_cells):
        for j2 in xrange(num_cells):
            for j3 in xrange(num_cells):

                wf[j1*num_points:((j1+1)*num_points),
                   j2*num_points:((j2+1)*num_points),
                   j3*num_points:((j3+1)*num_points)] = wf1

    x = (y1 + z1) * 0.5 * a0
    y = (x1 + z1) * 0.5 * a0
    z = (x1 + y1) * 0.5 * a0

    f = Invdisttree(np.vstack((x.flatten(), y.flatten(), z.flatten())).T, wf.flatten())

    lin = np.linspace(0, a0, L, endpoint=False)
    x1, y1, z1 = np.meshgrid(lin, lin, lin)

    wf = f(np.vstack((x1.flatten(), y1.flatten(), z1.flatten())).T, nnear=11, eps=0, p=1)

    return wf.reshape(x1.shape)


def read_wf(T, k1):

    # associate valley index with abinit wave-function

    if k1[0] != 0:
        if k1[0] > 0:
            indi = 5
        if k1[0] < 0:
            indi = 6

    if k1[1] != 0:
        if k1[1] > 0:
            indi = 3
        if k1[1] < 0:
            indi = 4

    if k1[2] != 0:
        if k1[2] > 0:
            indi = 1
        if k1[2] < 0:
            indi = 2

    # check if the unitcell function is already stored(both real and imaginary parts)

    pwd = os.path.dirname(os.path.abspath(__file__))

    if os.path.isfile(os.path.join(pwd, 'p_dopant_data/wfr_' + str(indi) + '_' + str(T) + '.npy')) and \
            os.path.isfile(os.path.join(pwd, 'p_dopant_data/wfi_' + str(indi) + '_' + str(T) + '.npy')):

        log.info("    Loading wave functions wfr and wfi from disk...."),

        wfr = np.load(os.path.join(pwd, 'p_dopant_data/wfr_' + str(indi) + '_' + str(T) + '.npy'))
        wfi = np.load(os.path.join(pwd, 'p_dopant_data/wfi_' + str(indi) + '_' + str(T) + '.npy'))
        wf = wfr + 1j * wfi

        log.info("    Done!")

    else:

        wf = np.loadtxt(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),
                                     'p_dopant_data/wf_k87_b5.npy'))
        # wf = read_cube('/home/mk/qe_si/results/silicon.wf_K001_B005.cube')
        # wfr = transform_to_uc(wf, 30)
        # wfi = np.zeros(np.shape(wfr))
        wfr = transform_to_uc(np.reshape(wf[:, 0], (20, 20, 20)), T)
        wfi = transform_to_uc(np.reshape(wf[:, 1], (20, 20, 20)), T)

        #wfr = transform_to_uc(wfr, T)
        #wfi = transform_to_uc(wfi, T)

        np.save(os.path.join(pwd, 'p_dopant_data/wfr_'+ str(indi) + '_' + str(T) + '.npy'), wfr)
        np.save(os.path.join(pwd, 'p_dopant_data/wfi_'+ str(indi) + '_' + str(T) + '.npy'), wfi)

        wf = wfr + 1j * wfi

    x = np.arange(0.0, si.a_Si / si.ab, (si.a_Si / si.ab) / T)
    me = simps(simps(simps(np.abs(wf)**2, x), x), x)

    return (1.0 / np.sqrt(me)) * wf
    # return wf


def abi_read(fac, T, valley):
    # the function reads the periodic functions computed by ABINIT
    # for fac number of unit cells

    wf1 = read_wf(T, valley)

    # if valley(find(valley))<0
    #     wf1=-wf1;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    # end;

    # compose a fac number of cells
    wf = np.zeros((fac*T, fac*T, fac*T), dtype=np.complex)

    for j1 in xrange(fac):
        for j2 in xrange(fac):
            for j3 in xrange(fac):
                wf[j1*T:(j1+1)*T, j2*T:(j2+1)*T, j3*T:(j3+1)*T] = wf1

    return wf
