import os
from argparse import ArgumentParser
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import silicon_params as si
from aux_functions import yaml_parser, make_id
from comp_vo_coupling import comp_vo_coupling
from read_env1 import read_env1, read_amp_from_buffer, k2flag


def make_dataset(path_to_data, atomic_coords, bands):

    field = 0
    x = np.linspace(-6, 6, 100)

    dataset = {
        'x':   {'amp': [], 'E': []},
        'y':   {'amp': [], 'E': []},
        'z':   {'amp': [], 'E': []}
    }

    for coords in atomic_coords:

        ids = make_id(field, coords / si.a_Si)
        path = os.path.join(os.path.join(path_to_data, ids))

        _, pth = k2flag(si.kk[1, :])
        pth = os.path.join(path, pth, 'amp_' + str(np.size(bands)) + '.pkl')
        amp_x, _ = read_amp_from_buffer(pth)(read_env1)(x,
                                                        bands,
                                                        path,
                                                        si.kk[1, :],
                                                        0,
                                                        np.array(atomic_coords))

        dataset['x']['amp'].append(amp_x)

        _, pth = k2flag(si.kk[3, :])
        pth = os.path.join(path, pth, 'amp_' + str(np.size(bands)) + '.pkl')
        amp_y, _ = read_amp_from_buffer(pth)(read_env1)(x,
                                                        bands,
                                                        path,
                                                        si.kk[3, :],
                                                        0,
                                                        np.array(atomic_coords))

        dataset['y']['amp'].append(amp_y)

        _, pth = k2flag(si.kk[4, :])
        pth = os.path.join(path, pth, 'amp_' + str(np.size(bands)) + '.pkl')
        amp_z, _ = read_amp_from_buffer(pth)(read_env1)(x,
                                                        bands,
                                                        path,
                                                        si.kk[5, :],
                                                        0,
                                                        np.array(atomic_coords))

        dataset['z']['amp'].append(amp_z)

        dataset['x']['E'].append(np.loadtxt(os.path.join(path, 'v0/EE_0.dat'))[bands])
        dataset['y']['E'].append(np.loadtxt(os.path.join(path, 'v1/EE_0.dat'))[bands])
        dataset['z']['E'].append(np.loadtxt(os.path.join(path, 'v2/EE_0.dat'))[bands])

    return dataset


def dataset2interps(atomic_coords, dataset):

    x = (atomic_coords[:, 1, 2] - atomic_coords[:, 0, 2]) / si.ab
    interps = {
        'x':   {'amp': [], 'E': []},
        'y':   {'amp': [], 'E': []},
        'z':   {'amp': [], 'E': []}
    }

    for key, value in dataset.iteritems():
        for key1, value1 in dataset[key].iteritems():
            for item in xrange(dataset[key][key1][0].size):
                y = [y1[np.unravel_index(item, dataset[key][key1][0].shape)] for y1 in dataset[key][key1]]
                interps[key][key1].append(interpolate.InterpolatedUnivariateSpline(x, y))

    return interps


def main(**kwargs):

    bands = np.array([1, 2, 3, 4, 5]) - 1

    # ----------------------- parse inputs ----------------------

    cnfg = yaml_parser(kwargs.get('path_to_config'))
    path_to_data = kwargs.get('path_to_data')
    verbosity = kwargs.get('verbosity')

    atomic_coords = cnfg['atomic_coords']
    fields = cnfg['field']

    atomic_coords = si.a_Si * np.array(atomic_coords)
    fields = np.array(fields)

    if len(atomic_coords.shape) == 2:
        atomic_coords = [atomic_coords]

    if len(fields.shape) == 1:
        fields = [fields]

    a = []
    b = []
    c = []
    ans = []
    ans1 = []

    # dataset = make_dataset(path_to_data, atomic_coords, bands)
    # interps = dataset2interps(atomic_coords, dataset)

    # define new dense grid of atomic coordinates
    num_of_points = 250
    aa = np.stack((np.zeros(num_of_points), np.zeros(num_of_points), np.linspace(0.1e-9, 4.8e-9, num_of_points)))
    # atomic_coords = np.swapaxes(np.swapaxes(np.stack((-aa, aa)), 1, 2), 0, 1)

    for coords in atomic_coords:
        for field in fields:

            ids = make_id(field, coords / si.a_Si)
            coords = (coords / si.ab).tolist()
            path = os.path.join(os.path.join(path_to_data, ids))

            # a.append(np.loadtxt(os.path.join(path, 'v0/EE_0.dat')))
            # b.append(np.loadtxt(os.path.join(path, 'v1/EE_0.dat')))
            # c.append(np.loadtxt(os.path.join(path, 'v2/EE_0.dat')))
            en, env = comp_vo_coupling(coords, path, bands, interpolators_dict=None)
            ans.append(en)
            ans1.append(env)

    # plt.plot(atomic_coords[:, 1, 2]/1e9, np.array(a)*si.E_Har/si.q*1000)
    # plt.plot(atomic_coords[:, 1, 2]/1e9, np.array(b)*si.E_Har/si.q*1000)
    # plt.plot(atomic_coords[:, 1, 2]/1e9, np.array(c)*si.E_Har/si.q*1000)
    # plt.plot(2 * atomic_coords[:, 1, 2] / 1e-9, ans)
    plt.plot(2 * atomic_coords[:, 1, 2] / si.a_Si / 2, ans)
    plt.show()

    print('hi')


if __name__ == '__main__':

    parser = ArgumentParser()

    # path to the data directory
    parser.add_argument("--path_to_data", type=str,
                        help="path to save/load the data",
                        default=os.path.join(os.path.dirname(__file__), 'p_dopant_data'))

    # path to the config file
    parser.add_argument("--path_to_config", type=str,
                        help="path to the config file",
                        default=os.path.join(os.path.dirname(__file__), 'config.yml'))

    parser.add_argument("-v", "--verbosity", type=int,
                        help="increase output verbosity",
                        default=1)

    args = vars(parser.parse_args())

    main(**args)

