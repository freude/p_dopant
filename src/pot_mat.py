import numpy as np
import silicon_params as si
from pot_scr_real import pot_scr_real


def pot_mat(coord, k1, k2):

    a = 1e-7
    
    #V=(coord.coord_limits(2)-coord.coord_limits(1)+coord.coord_stps)^3;
    
    x = np.arange(coord.coord_limits[0], coord.coord_limits[1], coord.coord_stps)
    y = np.arange(coord.coord_limits[0], coord.coord_limits[1], coord.coord_stps)
    z = np.arange(coord.coord_limits[0], coord.coord_limits[1], coord.coord_stps)
    
    X, Y, Z = np.meshgrid(x, y, z)

    x1 = -3e-9 / si.ab
    y1 = 0
    z1 = 0
    
    x2 = 3e-9 / si.ab
    y2 = 0
    z2 = 0
    
    ix1 = np.argmin(np.abs(x-x1))
    iy1 = np.argmin(np.abs(y-y1))
    iz1 = np.argmin(np.abs(z-z1))
    
    ix2 = np.argmin(np.abs(x-x2))
    iy2 = np.argmin(np.abs(y-y2))
    iz2 = np.argmin(np.abs(z-z2))
    
    x1 = x[ix1] - 0.5 * coord.coord_stps
    y1 = y[iy1] - 0.5 * coord.coord_stps
    z1 = z[iz1] - 0.5 * coord.coord_stps
    
    x2 = x[ix2] - 0.5 * coord.coord_stps
    y2 = y[iy2] - 0.5 * coord.coord_stps
    z2 = z[iz2] - 0.5 * coord.coord_stps
    
    x1 = 0 - 0 * 0.5 * coord.coord_stps
    y1 = 0 - 0 * 0.5 * coord.coord_stps
    z1 = 0 - 0 * 0.5 * coord.coord_stps
    
    V = np.real(1.0 / np.sqrt((X-x1)**2+(Y-y1)**2+(Z-z1)**2+1j*a))*pot_scr_real(np.sqrt((X-x1)**2+(Y-y1)**2+(Z-z1)**2))
    
    #V(isinf(V))=0;
    
    V = V * np.exp(1j*((k1[0]-k2[0])*(X-x1)+(k1[1]-k2[1])*(Y-y1)+(k1[2]-k2[2])*(Z-z1)))

    return V, X, Y, Z