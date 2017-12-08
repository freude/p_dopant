import os
import subprocess as sp
from argparse import ArgumentParser
import numpy as np
import silicon_params as si
from pot_for_ff import pot_for_ff


def make_mesh_generator_script(path):
    return """include "mesh_gen.idp"

verbosity=0;
real [int,int] BBB=[[-6, 6],[-6, 6],[-6, 6]]; // coordinates the cube
mesh3 Th=meshgen(BBB, 0, 0, 0, 1);            // coordinates of impurities
medit(1,Th);
savemesh(Th,"%s/mesh_sample.mesh"); // save mesh for further processing
""" % path


def main(**kwargs):
    """
    Main script that computes the set of envelope functions

    :param kwargs: input arguments parsed from the command line
    :return:
    """

    path_to_data = kwargs.get('path_to_data')
    path_to_config = kwargs.get('path_to_config')
    verbosity = kwargs.get('verbosity')

    # ---------------------- making mesh ------------------------

    if not os.path.isdir(path_to_data):
        raise EnvironmentError("Path to data does not exist")

    if not os.path.isfile(os.path.join(path_to_data, 'mesh_sample.mesh')):
        # file does not exist
        path_to_make_mesh = os.path.join(os.path.dirname(__file__),
                                         'make_mesh.edp')
        os.system('rm %s' % path_to_make_mesh)
        with open(path_to_make_mesh, 'w') as f:
            f.write(make_mesh_generator_script(path_to_data))

        sp.call(["FreeFem++", "make_mesh.edp"])

    # # ----------- computing periodic Bloch functions -----------
    #
    # flag = True
    #
    # for file_name in os.listdir(path_to_data):
    #     if file_name.startwith('wf'):
    #         flag = False
    #
    # if flag:
    #     os.system('python pot_ff.py')

    # -------------------- making potential ---------------------

    if not os.path.isfile(os.path.join(path_to_data, 'pot3.txt')):
        # file does not exist

        k0 = si.k0 * si.ab

        kk = k0 * np.array([[1,  0,  0],
                            [-1, 0,  0],
                            [0,  1,  0],
                            [0, -1,  0],
                            [0,  0,  1],
                            [0,  0, -1]])

        pot_for_ff(kk[0, :], kk[0, :], '1')
        pot_for_ff(kk[1, :], kk[1, :], '2')
        pot_for_ff(kk[2, :], kk[2, :], '3')

    # ---------------- computing envelope functions  ------------

    p1 = sp.Popen(["FreeFem++", "si_ham.edp",
                   "0", "0", "1.00", "1.00", "0.19", path_to_data])

    p2 = sp.Popen(["FreeFem++", "si_ham.edp",
                   "0", "0", "1.00", "0.19", "1.00", path_to_data])

    p3 = sp.Popen(["FreeFem++", "si_ham.edp",
                   "0", "0", "0.19", "1.00", "1.00", path_to_data])

    p1.communicate()
    p2.communicate()
    p3.communicate()


if __name__ == '__main__':

    parser = ArgumentParser()

    # path to the data directory
    parser.add_argument("--path_to_data", type=str,
                        help="path to save/load the data",
                        default=os.path.join(os.path.dirname(__file__), 'p_dopant_data'))

    # path to the config file
    parser.add_argument("--path_to_config", type=str,
                        help="path to the config file",
                        default=os.path.join(os.path.dirname(__file__), 'config.yaml'))

    parser.add_argument("-v", "--verbosity", type=int,
                        help="increase output verbosity",
                        default=1)

    args = vars(parser.parse_args())

    main(**args)
