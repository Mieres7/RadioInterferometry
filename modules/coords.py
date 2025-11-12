"""
Transformaciones de coordenadas
Funciones que convierten entre sistemas de referencia (ECEF, ENU, ALT-AZ, XYZ, UVW, etc.).
"""

import numpy as np
from scipy.spatial.distance import pdist

def ecef_to_enu(ecef, array_center, phi=-33.45, lamb=-70.66, rad=True, is_array=True):
    if rad:
        phi, lamb = np.deg2rad(phi), np.deg2rad(lamb)

    cphi, clam = np.cos(phi), np.cos(lamb)
    sphi, slam = np.sin(phi), np.sin(lamb)

    R = np.array([[-slam,        clam,       0],
                  [-clam*sphi,   -slam*sphi, cphi],
                  [clam*cphi,    slam*cphi,  sphi]])

    dxyz = np.array(ecef) - np.array(array_center)
    enu = R @ dxyz.T if is_array else R @ dxyz
    return enu


def enu_to_altaz(enu, rad=True, is_array=True):
    if is_array:
        E, N, U = enu[0], enu[1], enu[2]
    else:
        E, N, U = float(enu[0]), float(enu[1]), float(enu[2])

    r = np.hypot(E, N)
    El = np.arctan2(U, r)
    A = np.arctan2(E, N)
  
    if rad:
        return El, A % (2*np.pi)
    else:
        return np.degrees(El), np.degrees(A) % 360.0


def hor_to_eq(enu, alt, az, phi=-33.45):
    phi = np.radians(phi)
    enu = np.asarray(enu)
    if enu.shape[-1] != 3:
        raise ValueError("El array ENU debe tener forma (..., 3)")

    E, N, U = enu[..., 0], enu[..., 1], enu[..., 2]
    b_norm = np.sqrt(E**2 + N**2 + U**2)

    X = b_norm * ( np.sin(alt)*np.cos(phi) - np.cos(alt)*np.sin(phi)*np.cos(az) )
    Y = b_norm * ( np.sin(az)*np.cos(alt) )
    Z = b_norm * ( np.sin(alt)*np.sin(phi) + np.cos(alt)*np.cos(phi)*np.cos(az) )

    return np.stack([X, Y, Z], axis=-1)


def eq_to_uvw(H_array, delta, r_eq):
    cd, sd = np.cos(delta), np.sin(delta)
    ch, sh = np.cos(H_array), np.sin(H_array)
    R = np.stack([
        np.stack([sh,        ch,        np.zeros_like(H_array)], axis=-1),
        np.stack([-sd*ch,    sd*sh,     np.full_like(H_array, cd)], axis=-1),
        np.stack([cd*ch,    -cd*sh,     np.full_like(H_array, sd)], axis=-1)
    ], axis=-2)
    uvw = np.einsum('hij,bj->bhi', R, r_eq)
    return uvw


def baselines(enu, include_conjugate=True):
    enu = np.asarray(enu)
    N = len(enu)
    diff = enu[:, None, :] - enu[None, :, :]

    if include_conjugate:
        mask = ~np.eye(N, dtype=bool)
        baselines_result = diff[mask]
    else:
        i_indices, j_indices = np.triu_indices(N, k=1)
        baselines_result = diff[i_indices, j_indices]

    return baselines_result


def max_basline(baselines):
    distances = pdist(baselines)
    return np.max(distances)
