"""
The module contains a set of auxiliary functions facilitating computations
"""
import os
import yaml
import numpy as np


def yaml_parser(input_data):
    """

    :param input_data:
    :return:
    """

    output = None

    if input_data.lower().endswith(('.yml', '.yaml')):
        with open(input_data, 'r') as stream:
            try:
                output = yaml.load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    else:
        try:
            output = yaml.load(input_data)
        except yaml.YAMLError as exc:
            print(exc)

    return output


def clean_cache(pth):

    cache_data = os.path.join(pth, 'v0', 'E.dat')
    with open(cache_data, 'r+') as f:
        f.truncate()
    cache_data = os.path.join(pth, 'v1', 'E.dat')
    with open(cache_data, 'r+') as f:
        f.truncate()
    cache_data = os.path.join(pth, 'v2', 'E.dat')
    with open(cache_data, 'r+') as f:
        f.truncate()


def make_id(field, atomic_coords):

    id = []

    if np.sum(np.abs(np.array(field))) > 0:
        for item in field:
            id.append("%.2f" % item)

    for item in np.array(atomic_coords).flatten():
        if item == 0.0:
            id.append("0")
        else:
            id.append("%.2f" % item)

    id = "_".join(id)
    id = id.replace('-', 'm')
    id = id.replace('.', 'd')
    id = "sys_" + id

    return id


def array2list(array):
    """
    A recursive algorithm to convert a multidimensional numpy array to
    a multidimensional list

    :param array: numpy.array
    :return:      list
    """

    ans = []

    if len(array.shape) == 1:
        ans = list(array)
    else:
        for item in array:
            ans.append(array2list(item))

    return ans


if __name__ == '__main__':

    a = np.ones((3, 3, 3))

    # b = array2list(a)
    b = a.tolist()

    print b
