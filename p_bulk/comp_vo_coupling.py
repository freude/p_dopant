import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import silicon_params as si
from read_env1 import read_env1
from read_env import read_env_amp
from me2 import me2
from compose_wf import compose_wf

pth = os.getcwd()
k0 = si.k0 * si.ab

kk = k0 * np.array([[1,  0,  0],
                    [-1, 0,  0],
                    [0,  1,  0],
                    [0, -1,  0],
                    [0,  0,  1],
                    [0,  0, -1]])

bands = np.array([1, 3, 7, 10]) - 1
Nbands = np.size(bands)

# bands = np.array([1, 3]) - 1
# Nbands = np.size(bands)

x = np.linspace(-6, 6, 100)
path = os.path.join(pth, 'p_bulk/p_dopant_data/')
amp, _ = read_env1(x, bands, path,
                   kk[0, :],
                   0,
                   np.array([[0, 0, 0]]))

amp[np.abs(amp) < 0.01 * np.max(np.abs(amp))] = 0.0

amp = np.vstack((amp, amp, amp, amp, amp, amp))


print(amp)

amp = read_env_amp(bands, path,
                   kk[0, :],
                   0,
                   np.array([[0, 0, 0]]))

amp[np.abs(amp) < 0.01 * np.max(np.abs(amp))] = 0.0

amp = np.vstack((amp, amp, amp, amp, amp, amp))

print(amp)

#Energy = [-0.87 - 0.24];
Energy = [-0.87, -0.24, -0.1, -0.01]
# Energy = [-0.84 - 0.29 - 0.23];

# ---------------------------------------------------------------------
#
# if (exist(strcat(pwd, '/dis_scr/ME.mat'), 'file') == 2)
    # M = load(strcat(pwd, '/dis_scr/ME.mat'));
# ME_a = M.M(1);
# ME_b = M.M(2);
# else
# ME_a = me2(kk(1,:), kk(2,:), 'mes');
# ME_b = me2(kk(1,:), kk(3,:), 'mes');
# M = [real(ME_a) real(ME_b)];
# save(strcat(pwd, '/dis_scr/ME.mat'), 'M');
# end;
#
ME_a = me2(kk[0, :], kk[1, :], 'mes')
ME_b = me2(kk[0, :], kk[2, :], 'mes')
# ME_b = me2(kk(1,:), kk(4,:), 'mes')
# ME_b = me2(kk(1,:), kk(5,:), 'mes')
# ME_b = me2(kk(1,:), kk(6,:), 'mes')

# ME_a = me2(kk(2,:), kk(1,:), 'mes')
# ME_b = me2(kk(2,:), kk(3,:), 'mes')
# ME_b = me2(kk(2,:), kk(4,:), 'mes')
# ME_b = me2(kk(2,:), kk(5,:), 'mes')
# ME_b = me2(kk(2,:), kk(6,:), 'mes')

# ME_b = me2(kk(3,:), kk(1,:), 'mes')
# ME_b = me2(kk(3,:), kk(2,:), 'mes')
# ME_a = me2(kk(3,:), kk(4,:), 'mes')
# ME_b = me2(kk(3,:), kk(5,:), 'mes')
# ME_b = me2(kk(3,:), kk(6,:), 'mes')

# ME_b = me2(kk(4,:), kk(1,:), 'mes')
# ME_b = me2(kk(4,:), kk(2,:), 'mes')
# ME_a = me2(kk(4,:), kk(3,:), 'mes')
# ME_b = me2(kk(4,:), kk(5,:), 'mes')

# ME_b = me2(kk(5,:), kk(1,:), 'mes')
# ME_b = me2(kk(5,:), kk(2,:), 'mes')
# ME_b = me2(kk(5,:), kk(3,:), 'mes')
# ME_b = me2(kk(5,:), kk(4,:), 'mes')
# ME_a = me2(kk(5,:), kk(6,:), 'mes')

M = np.zeros((6 * Nbands, 6 * Nbands), dtype=complex)

# ---------------------------------------------------------------------

MEs = np.zeros((6, 6), dtype=complex)

for j1 in xrange(6):
    for j2 in xrange(6):
        if j1 != j2 and j1 < j2:

            if np.sum(np.abs(kk[j1, :] + kk[j2, :])) == 0:
                MEs[j1, j2] = ME_a
            else:
                MEs[j1, j2] = ME_b

        else:
            MEs[j1, j2] = 0

# ---------------------------------------------------------------------

for j1 in xrange(6):
    for j2 in xrange(6):
        for jj1 in xrange(Nbands):
            for jj2 in xrange(Nbands):
                if j1 == j2 and jj1 != jj2:
                    M[jj1 + Nbands * j1, jj2 + Nbands * j2] = 0
                elif j1 == j2 and jj1 == jj2:
                    M[jj1 + Nbands * j1, jj2 + Nbands * j2] = Energy[jj1]
                else:
                    M[jj1 + Nbands * j1, jj2 + Nbands * j2] = MEs[j1, j2] * amp[j1, jj1] * amp[j2, jj2]

for j1 in xrange(6 * Nbands):
    for j2 in xrange(6 * Nbands):
        if j1 != j2 and j1 > j2:
            M[j1, j2] = np.conj(M[j2, j1])

# ---------------------------------------------------------------------
# -------------Overlap matrix < k2, i | k1, j > for envelopes - ---------------
# ---------------------------------------------------------------------

# kin = [Energy(1) 0;...
# 0 Energy(2)];
#
# kin = [Energy(1) 0 0 0;...
# 0 Energy(2) 0 0;...
# 0 0 Energy(3) 0;...
# 0 0 0 Energy(4)];
# # -------------------------------------------
#
# D1 = M(1, 3);
# D2 = M(1, 4);
# D3 = M(2, 3);
# D4 = M(2, 4);
#
# D_1 = [D1 D2; D3 D4];
#
# D1 = M(1, 5);
# D2 = M(1, 6);
# D3 = M(2, 5);
# D4 = M(2, 6);
#
# D_2 = [D1 D2; D3 D4];
#
# D_1 = [D1 D2; D3 D4];
# D_2 = [D1 D2; D3 D4];
#
# D_1i = conj(D_1)';
# D_2i = conj(D_2)';
#
# Ma = [kin D_1   D_2   D_2   D_2   D_2;...
# D_1i kin D_2 D_2 D_2 D_2;...
# D_2i D_2i kin D_1 D_2 D_2;...
# D_2i D_2i D_1i kin D_2 D_2;...
# D_2i D_2i D_2i D_2i kin D_1;...
# D_2i D_2i D_2i D_2i D_1i kin];
#
# # Ma = [Energy(1) D1   D2   D2   D2   D2; ...
# # D1'  Energy(1) D2   D2   D2   D2;...
# # D2'  D2' Energy(1) D1 D2 D2;...
# # D2'  D2' D1'  Energy(1) D2   D2;...
# # D2'  D2' D2'  D2' Energy(1) D1;...
# # D2'  D2' D2'  D2' D1'  Energy(1)]; \ \

# ---------------------------------------------------------------------


a, EigVec = linalg.eig(M)
a = np.real(a) * 40
ind =np.argsort(a)
a1 = a[ind]

print(a1)

# EigVec1 = XX * EigVec;
# EigVec1 = EigVec1(:, ind);


EigVec1 = EigVec[:, ind]

F = compose_wf(path, kk, bands, a1, EigVec1)
np.save('F',np.abs(F[1][61, :, :]))
#plt.imshow(np.abs(F[1][61, :, :]))
#plt.savefig("out.png")
