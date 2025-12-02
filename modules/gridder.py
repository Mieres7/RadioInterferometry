from numba import cuda
import numpy as np
import cupy as cp
import math

from modules.coords import max_basline
from modules.noise import add_gaussian_noise
from modules.backend import get_backend

def grid_visibilities(V=None, uvw_lambda=None, du=None, dv=None, Npix=None, grid_config=None, mode='cupy'):
    """
    Grids complex visibilities onto a single (u, v) grid.
    """
    if grid_config is not None:
        V = grid_config.get('V', V)
        uvw_lambda = grid_config.get('uvw_lambda', uvw_lambda)
        du = grid_config.get('du', du)
        dv = grid_config.get('dv', dv)
        Npix = grid_config.get('Npix', Npix)

    if V is None or uvw_lambda is None or du is None or dv is None:
        raise ValueError("Faltan parámetros obligatorios (V, uvw, du, dv). "
                         "Deben pasarse como argumentos o dentro de grid_config.")

    mode = get_backend(mode)
    print(f'Backend a utilizar: {mode}')

    if mode == 'numba':
        V = cp.asarray(V)
        uvw_lambda = cp.asarray(uvw_lambda)
        vg, wg = grid_visibilities_cuda(V, uvw_lambda, du, dv, Npix=Npix)
        return vg, wg
    elif mode == 'cupy':
        sys = cp
        V_in = sys.asarray(V)
        uvw_in = sys.asarray(uvw_lambda)
    elif mode == 'numpy':
        sys = np
        V_in = V
        uvw_in = uvw_lambda

    u_coords = uvw_in[..., 0]
    v_coords = uvw_in[..., 1]

    VG = sys.zeros((Npix, Npix), dtype=sys.complex64)
    WG = sys.zeros((Npix, Npix), dtype=sys.float32)
    
    u_all = u_coords.ravel()
    v_all = v_coords.ravel()
    V_all = V_in.ravel() 

    omega_all = sys.ones_like(V_all, dtype=sys.float32)  # Pesos = 1

    i = sys.rint(u_all / du).astype(int) + Npix // 2
    j = sys.rint(v_all / dv).astype(int) + Npix // 2

    mask = (i >= 0) & (i < Npix) & (j >= 0) & (j < Npix)

    ii, jj = i[mask], j[mask]
    V_mask = V_all[mask]
    omega_mask = omega_all[mask]
    
    values_to_add = omega_mask * V_mask

    if mode == 'cupy':
        cp.add.at(VG.real, (jj, ii), values_to_add.real)
        cp.add.at(VG.imag, (jj, ii), values_to_add.imag)
        cp.add.at(WG, (jj, ii), omega_mask)
    else:
        sys.add.at(VG, (jj, ii), values_to_add)
        sys.add.at(WG, (jj, ii), omega_mask)
        
    valid_cells = WG > 0
    VG[valid_cells] /= WG[valid_cells]

    return VG, WG



@cuda.jit
def kernel_gridding_flat(u_flat, v_flat, vis_real, vis_imag, weights, 
                         grid_real, grid_imag, grid_wgt, 
                         du, dv, N):
    """
    Kernel que recibe arrays 1D perfectamente alineados.
    """
    tid = cuda.grid(1)
    
    # Verificamos límites del array aplanado
    if tid < u_flat.shape[0]:
        
        # 1. Recuperar coordenadas y datos del índice lineal
        u = u_flat[tid]
        v = v_flat[tid]
        vr = vis_real[tid]
        vi = vis_imag[tid]
        w = weights[tid]
        
        idx_u = int(math.floor(u / du) + N / 2)
        idx_v = int(math.floor(v / dv) + N / 2)
        
        # 3. Acumular si cae dentro de la grilla
        if 0 <= idx_u < N and 0 <= idx_v < N:
            # Convención (v, u) -> (fila, columna)
            cuda.atomic.add(grid_real, (idx_v, idx_u), vr * w)
            cuda.atomic.add(grid_imag, (idx_v, idx_u), vi * w)
            cuda.atomic.add(grid_wgt,  (idx_v, idx_u), w)

@cuda.jit
def kernel_normalize(grid_real, grid_imag, grid_wgt):
    x, y = cuda.grid(2)
    N = grid_real.shape[0]
    
    if x < N and y < N:
        w = grid_wgt[x, y]
        if w > 0:
            grid_real[x, y] /= w
            grid_imag[x, y] /= w


def grid_visibilities_cuda(V, uvw, du, dv, Npix=512):
    """
    Igual que antes, pero ahora usa CuPy en vez de NumPy.
    """

    # --- Flatten con CuPy ---
    vis_real_flat = cp.ascontiguousarray(V.real.reshape(-1).astype(cp.float32))
    vis_imag_flat = cp.ascontiguousarray(V.imag.reshape(-1).astype(cp.float32))

    u_flat = cp.ascontiguousarray(uvw[..., 0].reshape(-1).astype(cp.float32))
    v_flat = cp.ascontiguousarray(uvw[..., 1].reshape(-1).astype(cp.float32))

    weights_flat = cp.ones(vis_real_flat.size, dtype=cp.float32)

    # --- Convertir CuPy -> Numba (sin copia) ---
    d_u     = cuda.as_cuda_array(u_flat)
    d_v     = cuda.as_cuda_array(v_flat)
    d_vis_r = cuda.as_cuda_array(vis_real_flat)
    d_vis_i = cuda.as_cuda_array(vis_imag_flat)
    d_wgt   = cuda.as_cuda_array(weights_flat)

    # --- Crear grillas en CuPy ---
    grid_r = cp.zeros((Npix, Npix), dtype=cp.float32)
    grid_i = cp.zeros((Npix, Npix), dtype=cp.float32)
    grid_w = cp.zeros((Npix, Npix), dtype=cp.float32)

    # Convertirlas a Numba
    d_grid_r = cuda.as_cuda_array(grid_r)
    d_grid_i = cuda.as_cuda_array(grid_i)
    d_grid_w = cuda.as_cuda_array(grid_w)

    # --- Ejecutar kernels ---
    threads = 256
    blocks = (vis_real_flat.size + threads - 1) // threads

    kernel_gridding_flat[blocks, threads](
        d_u, d_v, d_vis_r, d_vis_i, d_wgt,
        d_grid_r, d_grid_i, d_grid_w,
        float(du), float(dv), int(Npix)
    )
    cuda.synchronize()

    # Normalización
    t2d = (16, 16)
    b_x = (Npix + t2d[0] - 1) // t2d[0]
    b_y = (Npix + t2d[1] - 1) // t2d[1]

    kernel_normalize[(b_x, b_y), t2d](d_grid_r, d_grid_i, d_grid_w)
    cuda.synchronize()

    # --- Retornar como CuPy ---
    VG = grid_r + 1j * grid_i
    WG = grid_w

    return VG, WG



def get_grid_config(V, uvw_lambda, N, baselines, oversampling_factor, frequencies, add_noise=False):
    # Max distance between baselines
    Dmax = max_basline(baselines)

    c = 299792458.0 
    freq = np.min(frequencies)
    min_wavelenghgt = c / freq
    dx = dy = (min_wavelenghgt / Dmax) / oversampling_factor

    du = 1.0 / (N * dx)
    dv = du

    if add_noise:
        V = add_gaussian_noise(V)

    grid_config = {
        'V': V,
        'uvw_lambda': uvw_lambda,
        'du': du,
        'dv': dv,
        'Dmax': Dmax,
        'frequencies': frequencies,
        'oversampling_factor': oversampling_factor,
        'baselines': baselines,
        'Npix': N,
        'pixel_size_image': dx
    }

    return grid_config