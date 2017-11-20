# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from coordsys import CoordSys
from spec_func import *

# num_cells=4;
# T=80;

# num_cells = 50
# T = 6

num_cells = 100
T = 1


N_states = 7

#      n  l  m      E        p      sigma
qn = [[1, 0, 0, -2.56853, 1.12186, 0.24792],      #1s_sigma_g
      [2, 0, 0, -0.78975, 0.62208, 1.25053],      #2s_sigma_g
      [3, 0, 0, -0.37851, 0.430664, 2.25079],     #3s_sigma_g
      [2, 1, 0, -1.22415, 0.77449, 0.80764],      #2p_sigma_u
      [3, 1, 0, -0.49729, 0.493633, 1.83612],     #3p_sigma_u
      [4, 1, 0, -0.27065, 0.364171, 2.84435],     #4p_sigma_u
      [3, 2, 0, -0.45660, 0.473004, 1.95981],     #3d_sigma_g
      [4, 3, 0, -0.25158, 0.351105, 2.98741],     #4f_sigma_u
      [2, 1, 1, -0.91265, 0.66873, 0.09352],      #2p_pi_u
      [3, 2, 1, -0.44939, 0.469258, 0.98343]]     #3d_pi_g

coorsys = CoordSys(num_cells,T,'au')
coorsys.set_origin_cells(num_cells/2+1)
x=coorsys.x()
[X1,Y1,Z1] = np.meshgrid(x,x,x)
s=X1.shape

# Conversion cartesian coordinates to prolate spheroidal ones

Ra = [0, 0, -0.7]
Rb = [0, 0,  0.7]
R = np.sqrt((Rb[0]-Ra[0])**2+(Rb[1]-Ra[1])**2+(Rb[2]-Ra[2])**2)
ra = np.sqrt((X1-Ra[0])**2+(Y1-Ra[1])**2+(Z1-Ra[2])**2)
rb = np.sqrt((X1-Rb[0])**2+(Y1-Rb[1])**2+(Z1-Rb[2])**2)

lam = np.divide((ra+rb), (R))
mu = np.divide((ra-rb), (R))
phi = np.arctan2(Y1, X1)

bas_fun = [x]
# X1, Y1, Z1 = X1.flatten(), Y1.flatten(), Z1.flatten()

for jj in range(N_states):
    if qn[jj][2]==0:
       bas_fun.append(LargeLambda(qn[jj], jj, lam)*LargeM(qn[jj],jj,mu)*np.cos(qn[jj][2]*phi))
    else:
       bas_fun.append(LargeLambda(qn[jj], jj, lam)*LargeM(qn[jj],jj,mu)*np.sin(qn[jj][2]*phi))

    bas_fun[jj + 1] = np.nan_to_num(bas_fun[jj + 1])

    ME = np.trapz(np.trapz(np.trapz(np.abs(bas_fun[jj+1])**2, x, axis=2), x, axis=1), x, axis=0)
    print(ME)
    bas_fun[jj+1]=bas_fun[jj+1]/np.sqrt(ME)


a=plt.contour(bas_fun[5][:,51,:], 30); plt.colorbar()
plt.imshow((np.abs(bas_fun[5][:,51,:])))
plt.show()

from fci.gauss_fit import GFit

qn=4
# data = np.vstack((X1, Y1, Z1, bas_fun[qn].flatten())).T

wf = GFit(sn=10,
          qn=qn,
          num_fu=14,
          psave='./')

wf.cube = ([np.min(x), np.min(x), np.min(x)],
           [np.max(x), np.max(x), np.max(x)])

wf.nuclei_coords = np.array([Ra, Rb])
wf.set_init_conditions(method = 'nuclei', widths=1, amps=0.5)

wf.do_fit(bas_fun[5], x=X1, y=Y1, z=Z1)

g = wf.get_data_matrix(X1, Y1, Z1)
err=np.abs(bas_fun[5]-g)
a=plt.contour((err[:,51,:]), 30); plt.colorbar()
plt.imshow((err[:,51,:]))
plt.show()

print(np.max(err))
print(np.max(bas_fun[3]))

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(wf._gf[0::5], wf._gf[1::5], wf._gf[2::5])

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

a=plt.contour((g[:,51,:]), 30); plt.colorbar()
plt.imshow(g[:,51,:])
plt.show()

# # wf.save()
# print(wf._gf)
# # wf.draw_func(x,y,par='2d')
# g = wf.get_data(XX)
# # g1 = wf.show_gf(XXX)
# # AA = wf.get_value(XX.T)
#
# fig = plt.figure()
# # for j in range(0,wf._num_fu):
# #     plt.plot(XXX[0,:].T,g1[:,j])
#
# # plt.plot(xi[150, :].T, g.reshape(xi.shape)[150, :])
# # plt.plot(xi[150, :].T, AA.reshape(xi.shape)[150, :])
#
# # ax = fig.add_subplot(111, projection='3d')
# # ax.plot_surface(xi,yi,g1.reshape(xi.shape), cmap=cm.jet, linewidth=0.2)
#
# # plt.contour(xi, yi, -AA.reshape(xi.shape), colors='red')
# plt.contour(xi, yi, g.reshape(xi.shape), 30, colors='blue')

plt.hold(True)
plt.show()
