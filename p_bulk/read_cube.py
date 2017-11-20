import numpy as np


def read_cube(file_path):
    with  (open(file_path, mode = 'r')) as f:

        comment1 = f.readline() #Save 1st comment
        comment2 = f.readline() #Save 2nd comment

        nOrigin = f.readline().split()        # Number of Atoms and Origin
        natoms = int(nOrigin[0])              # Number of Atoms
        origin = np.array([float(nOrigin[1]),
                           float(nOrigin[2]),
                           float(nOrigin[3])]) #Position of Origin

        nVoxel = f.readline().split() #Number of Voxels
        NX = int(nVoxel[0])
        X = np.array([float(nVoxel[1]),float(nVoxel[2]),float(nVoxel[3])])

        nVoxel = f.readline().split() #
        NY = int(nVoxel[0])
        Y = np.array([float(nVoxel[1]),float(nVoxel[2]),float(nVoxel[3])])

        nVoxel = f.readline().split() #
        NZ = int(nVoxel[0])
        Z = np.array([float(nVoxel[1]),float(nVoxel[2]),float(nVoxel[3])])

        atoms = []
        atomsXYZ = []

        for atom in range(natoms):
            line= f.readline().split()
            atoms.append(line[0])
            atomsXYZ.append(map(float,[line[2], line[3], line[4]]))

        data = np.zeros((NX, NY, NZ))
        i=0
        for s in f:
            for v in s.split():
                data[i//(NY*NZ), (i//NZ)%NY, i%NZ] = float(v)
                i+=1

        if i != NX*NY*NZ: raise NameError, "FSCK!"

    return data
