import os
import subprocess as sp
from argparse import ArgumentParser
import ast
import numpy as np
import silicon_params as si
from pot_for_ff import pot_for_ff
from aux_functions import yaml_parser


def make_mesh_generator_script(path,
                               cube_coords='[[-7, 7], [-7, 7], [-7, 7]]',
                               num_elem_init='[5, 5, 5]',
                               atomic_coords='[[0, 0, 0]]',
                               mesh_adaptation=1,
                               verbosity=1):

    cube_coords = str(cube_coords)
    num_elem_init = str(num_elem_init)
    atomic_coords = str(atomic_coords)
    mesh_adaptation = int(mesh_adaptation)
    coords = ast.literal_eval(atomic_coords)

    return """include "mesh_gen.idp"
    
              verbosity={6!s};
              real [int,int] cubecoords={1!s};   // coordinates the cube
              real [int,int] atomiccoords={2!s}; // atomic coordinates
              int [int] numelem={4!s};       // initial number of elments
              mesh3 Th=meshgen(cubecoords, {3!s}, atomiccoords, numelem, {5!s});
              if (verbosity > 0) medit(1,Th);
              
              savemesh(Th,"{0}/mesh_sample.mesh"); // save mesh for further processing
              
           """.format(path,
                      cube_coords,
                      atomic_coords,
                      len(coords),
                      num_elem_init,
                      mesh_adaptation,
                      verbosity)


def main(**kwargs):
    """
    Main script that computes the set of envelope functions

    :param kwargs: input arguments parsed from the command line
    :return:
    """
    # ----------------------- parse inputs ----------------------

    cnfg = yaml_parser(kwargs.get('path_to_config'))
    path_to_data = kwargs.get('path_to_data')
    verbosity = kwargs.get('verbosity')

    if verbosity > 0:
        print("I am going to save data to {}".format(path_to_data))

    # ----------------- check whether path is exist -------------

    if not os.path.isdir(path_to_data):
        raise EnvironmentError("Path to data does not exist, ", path_to_data)

    # ---------------------- atomic coordinates ------------------------

    atomic_coords = [[0, 0, -1.5], [0, 0, 1.5]]

    # ---------------------- making mesh ------------------------

    if os.path.isfile(os.path.join(path_to_data, 'mesh_sample.mesh')):
        print("There is a mesh stored in the file: {}".format(os.path.join(path_to_data, 'mesh_sample.mesh')))
        user_input = raw_input("Do you want to generate a new mesh [y/N]:")
    else:
        user_input = 'y'

    if not os.path.isfile(os.path.join(path_to_data, 'mesh_sample.mesh')) or user_input.lower() == 'y':
        # file does not exist
        path_to_make_mesh = os.path.join(os.path.dirname(__file__), 'make_mesh.edp')
        os.system('rm %s' % path_to_make_mesh)

        with open(path_to_make_mesh, 'w') as f:
            mesh = make_mesh_generator_script(path_to_data,
                                              cube_coords=cnfg['cube_coords'],
                                              num_elem_init=cnfg['num_elem_init'],
                                              atomic_coords=atomic_coords,
                                              mesh_adaptation=cnfg['mesh_adaptation'],
                                              verbosity=1)
            f.write(mesh)

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

    if os.path.isfile(os.path.join(path_to_data, 'pot3.txt')):
        print("There is a potential stored in: {}".format(os.path.join(path_to_data, 'pot3.txt')))
        user_input = raw_input("Do you want to generate a new potential [y/N]:")
    else:
        user_input = 'y'

    if not os.path.isfile(os.path.join(path_to_data, 'pot3.txt')) or user_input.lower() == 'y':
        # file does not exist

        k0 = si.k0 * si.ab

        kk = k0 * np.array([[1,  0,  0],
                            [-1, 0,  0],
                            [0,  1,  0],
                            [0, -1,  0],
                            [0,  0,  1],
                            [0,  0, -1]])

        pot_for_ff(atomic_coords, kk[0, :], kk[0, :], '1')
        pot_for_ff(atomic_coords, kk[1, :], kk[1, :], '2')
        pot_for_ff(atomic_coords, kk[2, :], kk[2, :], '3')

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
                        default=os.path.join(os.path.dirname(__file__), 'config.yml'))

    parser.add_argument("-v", "--verbosity", type=int,
                        help="increase output verbosity",
                        default=1)

    args = vars(parser.parse_args())

    main(**args)
