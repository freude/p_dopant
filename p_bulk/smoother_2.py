import math
import numpy as np
from read_env1 import k2flag
import silicon_params as si
from pot_scr import pot_scr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


def find(x):
    return np.where(x != 0)


def br_zone_valley(x2, y2, z2, valley, shift):
    """
    Computes the membership function for the point with coordinates x2, y2, z2
    to a certain sector of the Brillouin zone containing a single valley.
    The coordinate system of the Brillouin zone and some of its special points are shown below
    # -----------------------------------------------------
    # -----------------***************---------------------
    # ---------------*-----------------*-------------------
    # -------------*---------------------*-----------------
    # -----------*-------------------------*---------------
    # ---------*-----------------------------*-------------
    # ---------*-----------------------------*-------------
    # ---------*-----------------------------*-------------
    # ---------*--------------0(0,0,0)-*--*--*X(1,0,0)-----
    # ---------*-------------------*---------*-------------
    # ---------*----------------*-------*----*-------------
    # ---------*-------------------*---------*W(1,0.5,0)---
    # -----------*--------------------*----*---------------
    # -------------*---------------------*K(0.75,0.75,0)---
    # ---------------*-----------------*-------------------
    # -----------------***************---------------------
    # -----------------------------------------------------

    :param
    """

    #     tt=-math.pi/4;
    #
    #     # Rz=[np.cos(tt) -np.sin(tt)  0;
    #     #     np.sin(tt)  np.cos(tt)  0;
    #     #       0       0       1];
    #
    #     y1=y2*np.cos(tt)-z2*np.sin(tt);
    #     z1=y2*np.sin(tt)+z2*np.cos(tt);
    #     x1=x2;
    #
    #     tt=-math.pi/4;
    #
    #     x2=x1*np.cos(tt)-z1*np.sin(tt);
    #     z2=x1*np.sin(tt)+z1*np.cos(tt);
    #     y2=y1;
    #
    #
    #     x1=x2*np.cos(tt)-y2*np.sin(tt);
    #     y1=x2*np.sin(tt)+y2*np.cos(tt);
    #     z1=z2;

    x1 = x2
    y1 = y2
    z1 = z2

    ##
    if valley == 'x':
        tt = math.pi
        # Rx=[1     0      0;
        #     0 np.cos(tt) -np.sin(tt);
        #     0 np.sin(tt) np.cos(tt)];
        #
        # Ry=[np.cos(tt)  0   -np.sin(tt);
        #       0      1     0;
        #     np.sin(tt)  0   np.cos(tt)];
        #
        # Rz=[np.cos(tt) -np.sin(tt)  0;
        #     np.sin(tt)  np.cos(tt)  0;
        #       0       0       1];

        x = x1 * np.cos(tt) - y1 * np.sin(tt)
        y = x1 * np.sin(tt) + y1 * np.cos(tt)
        z = z1
        x = x - shift * si.k0 / si.ab

    if valley == 'y':
        tt = math.pi/2
        # Rx=[1     0      0;
        #     0 np.cos(tt) -np.sin(tt);
        #     0 np.sin(tt) np.cos(tt)];
        #
        # Ry=[np.cos(tt)  0   -np.sin(tt);
        #       0      1     0;
        #     np.sin(tt)  0   np.cos(tt)];
        #
        # Rz=[np.cos(tt) -np.sin(tt)  0;
        #     np.sin(tt)  np.cos(tt)  0;
        #       0       0       1];

        x = x1 * np.cos(tt) - y1 * np.sin(tt)
        y = x1 * np.sin(tt) + y1 * np.cos(tt)
        z = z1
        x = x - shift * si.k0 / si.ab

    if valley == '-y':
        tt = -math.pi/2
        # Rx=[1     0      0;
        #     0 np.cos(tt) -np.sin(tt);
        #     0 np.sin(tt) np.cos(tt)];
        #
        # Ry=[np.cos(tt)  0   -np.sin(tt);
        #       0      1     0;
        #     np.sin(tt)  0   np.cos(tt)];
        #
        # Rz=[np.cos(tt) -np.sin(tt)  0;
        #     np.sin(tt)  np.cos(tt)  0;
        #       0       0       1];

        x = x1 * np.cos(tt) - y1 * np.sin(tt)
        y = x1 * np.sin(tt) + y1 * np.cos(tt)
        z = z1
        x = x - shift * si.k0 / si.ab

    if valley == 'z':
        tt = math.pi/2
        # Rx=[1     0      0;
        #     0 np.cos(tt) -np.sin(tt);
        #     0 np.sin(tt) np.cos(tt)];
        #
        # Ry=[np.cos(tt)  0   -np.sin(tt);
        #       0      1     0;
        #     np.sin(tt)  0   np.cos(tt)];
        #
        # Rz=[np.cos(tt) -np.sin(tt)  0;
        #     np.sin(tt)  np.cos(tt)  0;
        #       0       0       1];

        x = x1 * np.cos(tt) - z1 * np.sin(tt)
        z = x1 * np.sin(tt) + z1 * np.cos(tt)
        y = y1
        x = x - shift * si.k0 / si.ab

    if valley == '-z':
        tt = -math.pi/2
        # Rx=[1     0      0;
        #     0 np.cos(tt) -np.sin(tt);
        #     0 np.sin(tt) np.cos(tt)];
        #
        # Ry=[np.cos(tt)  0   -np.sin(tt);
        #       0      1     0;
        #     np.sin(tt)  0   np.cos(tt)];
        #
        # Rz=[np.cos(tt) -np.sin(tt)  0;
        #     np.sin(tt)  np.cos(tt)  0;
        #       0       0       1];

        x = x1 * np.cos(tt) - z1 * np.sin(tt)
        z = x1 * np.sin(tt) + z1 * np.cos(tt)
        y = y1
        x = x - shift * si.k0 / si.ab

    if valley == '-x':
        x = x1
        y = y1
        z = z1
        x = x - shift * si.k0 / si.ab

    ###

    rho = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan(y / x)
    theta = np.arccos(z / rho)

    R1 = 2 * math.pi / 0.5431e-9
    R2 = 0.75 * 2 * math.pi / 0.5431e-9

    def s1(ph, th, rh): return (rh*np.cos(ph)*np.sin(th)-R1) < 0

    def s2(ph, th, rh):
        return (rh*np.cos(ph)*np.sin(th)/(2*R2)+rh*np.sin(ph)*np.sin(th)/(2*R2)+rh*np.cos(th)/(2*R2)-1) < 0

    def s3(ph, th, rh):
        return (rh*np.cos(ph)*np.sin(th)/(2*R2)-rh*np.sin(ph)*np.sin(th)/(2*R2)+rh*np.cos(th)/(2*R2)-1) < 0

    def s4(ph, th, rh):
        return (rh*np.cos(ph)*np.sin(th)/(2*R2)+rh*np.sin(ph)*np.sin(th)/(2*R2)-rh*np.cos(th)/(2*R2)-1) < 0

    def s5(ph, th, rh):
        return (rh*np.cos(ph)*np.sin(th)/(2*R2)-rh*np.sin(ph)*np.sin(th)/(2*R2)-rh*np.cos(th)/(2*R2)-1) < 0

    def s6(ph, th, rh): return (np.cos(ph)*np.sin(th)+np.cos(th)) > 0

    def s7(ph, th, rh): return (np.cos(ph)*np.sin(th)-np.cos(th)) > 0

    def s8(ph, th, rh): return (abs(ph)) < math.pi/4

    def s9(ph, th, rh): return x < 0

    bi = s1(phi, theta, rho) *\
        s2(phi, theta, rho) *\
        s3(phi, theta, rho) *\
        s4(phi, theta, rho) *\
        s5(phi, theta, rho) *\
        s6(phi, theta, rho) *\
        s7(phi, theta, rho) *\
        s8(phi, theta, rho) *\
        s9(phi, theta, rho)

    return bi


def smoother_2(x, G, k1, k2, R_tr, coords):

    if (((k1[find(k1)] > 0) and (k2[find(k2)] < 0)) or ((k2[find(k2)] > 0) and (k1[find(k1)] < 0))) and\
            (find(k1) == find(k2)):
        M1 = 1
    else:
        M1 = 1

    N = len(x)

    flag1, _ = k2flag(k1)
    flag2, _ = k2flag(k2)

    coord_stps = x[1] - x[0]
    kx = np.linspace(-math.pi / coord_stps, (math.pi - 2 * math.pi / N) / coord_stps, N)

    kX, kY, kZ = np.meshgrid(kx, kx, kx)

    bi = br_zone_valley(kX / si.ab, kY / si.ab, kZ / si.ab, flag1, 1)
    kk = (k1 - k2)
    # kk = -k1;

    V = np.zeros(kX.shape, dtype=complex)
    # R_tr = 7;

    for j in xrange(50):
        qq = np.sqrt((kX - M1 * kk[0] - G[j, 0])**2 + (kY - M1 * kk[1] - G[j, 1])**2 + (kZ - M1 * kk[2] - G[j, 2])**2)
        if flag1 == flag2:
            VV = 4. * math.pi / (qq**2) * (1 - np.cos(qq * R_tr))
            VV[np.isnan(VV)] = 4. * math.pi * 0.5 * R_tr ** 2
            V -= pot_scr(qq) * (G[j, 3] + 1j * G[j, 4]) * VV
        else:
            V -= pot_scr(qq) * (G[j, 3] + 1j * G[j, 4]) * 4 * math.pi / (qq**2)

    V_log = np.log(V)
    # plt.imshow(np.abs((bi * 0.3 * np.max(V_log) + V_log)[:, :, V.shape[2] / 2]))
    # plt.imshow(np.abs((V_log)[:, :, V.shape[2] / 2]))
    # plt.show()
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_trisurf(V_log[:, 0], V_log[:, 1], V_log[:, 2], cmap='Spectral', lw=1)
    # plt.show()
    # plt.imshow(np.abs((bi * 0.3 * np.max(V_log) + V_log)[:, :, V.shape[2] / 2]))
    # plt.show()

    aaa = np.zeros(kX.shape, dtype=complex)

    for item in coords:

        aaa += np.exp(1j * (kX * item[0] + kY * item[1] + kZ * item[2]))

    V = bi * V * aaa
    # bi = br_zone_valley(kX. / si.ab, kY. / si.ab, kZ. / si.ab, flag2, 1);
    # bi = fftshift(ifftn(ifftshift(bi))). * (abs(kx(2) - kx(1)) ^ 3). / ((2 * math.math.pi) ^ 3). * N ^ 3;
    # bi = trapz(x, trapz(x, trapz(x, (bi), 3), 2), 1);
    V1 = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(V))) * (np.abs(kx[1] - kx[0]) ** 3) / ((2.0 * math.pi) ** 3)

    return V1
