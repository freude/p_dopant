from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from scipy.special import binom
from  math import floor, factorial

def LargeM(qn, jj, x):

    f=[[   1.0769,   0.15634,   0.003407,   0.000031],   #1s_sigma_g
       [   1.0221,   0.04448,   0.000296,   0.000001],   #2s_sigma_g
       [   1.0105,   0.02094,   0.000067,        0.0],   #3s_sigma_g
       [   1.0368,   0.02474,   0.000168,   0.000001],   #2p_sigma_u
       [   1.0148,   0.00987,   0.000027,        0.0],   #3p_sigma_u
       [   1.0080,  0.005341,   0.000008,        0.0],   #4p_sigma_u
       [-0.004978,   0.99413,   0.005446,     0.0001],   #3d_sigma_g
       [-0.002114,   1.00131,   0.002177,   0.000002],   #4f_sigma_u
       [   1.0091,  0.006089,   0.000019,        0.0],   #2p_pi_u
       [   1.0068,  0.002717,   0.000003,        0.0]]   #3d_pi_g

    y = np.zeros(x.shape)
    m=qn[2]

    for j in range(len(f[jj])):
        if (qn[1] + qn[2])%2 == 0:
            s = 2*(j+1)-2
        else:
            s = 2*(j+1)-1
        # print(m+s, m)
        y=y+f[jj][j]*AssociatedLegendre(m+s, m, x)

    #y=y*((1.0-x**2)**(abs(m)/2))

    return y


def LargeLambda(qn,jj,x):

    g=[[1.0,   0.0105,   0.0004,   0.0,     0.0,    0.0,    0.0], # 1s_sigma_g
       [1.0,  -2.5506,  -0.0317,  -0.0013, -0.0002, 0.0,    0.0],  # 2s_sigma_g
       [1.0,  -4.0663,   3.6591,   0.0465,  0.0019, 0.0002, 0.0],  # 3s_sigma_g
       [1.0,   0.1788,   0.0006,   0.0,     0.0,    0.0,    0.0],  # 2p_sigma_u
       [1.0,  -1.5518,  -0.3581,  -0.0009, -0.0001, 0.0,    0.0],  # 3p_sigma_u
       [1.0,  -2.8631,   1.6243,   0.4816,  0.0011, 0.0001, 0.0],  # 4p_sigma_u
       [1.0,   2.2933,   0.3387,   0.0,     0.0,    0.0,    0.0],  # 3d_sigma_g
       [1.0,   6.9751,   5.4497,   0.4717,  0.0,    0.0,    0.0],  # 4f_sigma_u
       [1.0,   0.0224,  -0.0002,   0.0,     0.0,    0.0,    0.0],  # 2p_pi_u
       [1.0,   0.6179,  -0.0011,   0.0,     0.0,    0.0,    0.0]]  # 3d_pi_g

    y = np.zeros(x.shape)

    for j in range(len(g[jj])):
        y = y+g[jj][j]*np.power(np.divide(x-1, x+1), j)

    p = qn[4]
    sigma = qn[5]
    m = qn[2]

    y=np.exp(-p*x)*((x**2-1)**((m)/2))*((x+1)**sigma)*y
    return y

def AssociatedLegendre(l, m, x):
    Alm=np.zeros(x.shape)
    for r in range(int(floor(l/2-abs(m)/2))+1):
        Alm=Alm+((-1)**r)*binom(l-2*r,abs(m))*binom(l,r)*binom(2*l-2*r,l)*(x**(l-2*r-abs(m)))

    return ((-1)**m)*((1.0-x**2)**(abs(m)/2))*(factorial(abs(m))/(2**l)*Alm)