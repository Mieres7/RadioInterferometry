"""
Cálculos de visibilidades, frecuencias y grillas
"""

import numpy as np
import math
import cupy as cp
from numpy.fft import fftshift, ifftshift, ifft2
from numba import cuda


def uvw_to_lambda(uvw, freq_hz):
    c = 299792458.0
    lam = c / freq_hz
    return uvw / lam, lam


def uvw_to_lambda_range(uvw, freqs_hz):
    c = 299792458.0
    freqs_hz = np.asarray(freqs_hz)
    lams = c / freqs_hz
    uvw_lambda = uvw[:, :, np.newaxis, :] / lams[np.newaxis, np.newaxis, :, np.newaxis]
    return uvw_lambda, lams


def direction_cosines(ra_rad, dec_rad, ra0_rad, dec0_rad):
    dalpha = ra_rad - ra0_rad
    cosd, sind = np.cos(dec_rad), np.sin(dec_rad)
    cosd0, sind0 = np.cos(dec0_rad), np.sin(dec0_rad)
    l = cosd * np.sin(dalpha)
    m = sind * cosd0 - cosd * sind0 * np.cos(dalpha)
    n = sind * sind0 + cosd * cosd0 * np.cos(dalpha)
    return l, m, n


def visibilities_from_sources(uvw_lambda, sources, ra0_deg, dec0_deg, sigma_pb=0.05):
    u, v, w = uvw_lambda[..., 0], uvw_lambda[..., 1], uvw_lambda[..., 2]
    ra0, dec0 = np.deg2rad(ra0_deg), np.deg2rad(dec0_deg)

    ras = np.array([src['ra_deg'] for src in sources])
    decs = np.array([src['dec_deg'] for src in sources])
    S0s = np.array([src.get('S0', 1.0) for src in sources])

    ras_rad, decs_rad = np.deg2rad(ras), np.deg2rad(decs)
    l_src, m_src, n_src = direction_cosines(ras_rad, decs_rad, ra0, dec0)
    A_src = np.exp(-(l_src**2 + m_src**2) / (2 * sigma_pb**2))

    V_total = np.zeros(u.shape, dtype=np.complex64)
    for ls, ms, ns, As, Ss in zip(l_src, m_src, n_src, A_src, S0s):
        phase = 2j * np.pi * (u * ls + v * ms + w * (ns - 1.0))
        V_total += As * Ss / ns * np.exp(phase)

    omega = np.ones_like(V_total, dtype=float)
    return V_total, omega, l_src, m_src, n_src


def generate_random_sources(ra0_deg, dec0_deg, N=50, max_offset_deg=1.0, flux_range=(0.1, 2.0), seed=None):
    rng = np.random.default_rng(seed)
    ras = ra0_deg + rng.uniform(-max_offset_deg, max_offset_deg, N)
    decs = dec0_deg + rng.uniform(-max_offset_deg, max_offset_deg, N)
    fluxes = rng.uniform(flux_range[0], flux_range[1], N)
    return [{"ra_deg": ra, "dec_deg": dec, "S0": S} for ra, dec, S in zip(ras, decs, fluxes)]


def grid_visibilities(V, uvw_lambda, du, dv, Npix=256, use_gpu=True):
    """
    Grids complex visibilities onto a single (u, v) grid
    """

    if use_gpu:
        sys = cp
        V_in = sys.asarray(V)
        uvw_in = sys.asarray(uvw_lambda)
        print("Gridding in GPU (CuPy Vectorized)")
    else:
        sys = np
        V_in = V
        uvw_in = uvw_lambda
        print("Gridding in CPU (NumPy Vectorized)")

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

    if use_gpu:
        cp.add.at(VG.real, (jj, ii), values_to_add.real)
        cp.add.at(VG.imag, (jj, ii), values_to_add.imag)
        cp.add.at(WG, (jj, ii), omega_mask)
    else:
        sys.add.at(VG, (jj, ii), values_to_add)
        sys.add.at(WG, (jj, ii), omega_mask)
        
    valid_cells = WG > 0
    VG[valid_cells] /= WG[valid_cells]

    return VG, WG

# 1. KERNEL SIMPLE (Sin cambios, sirve igual)
@cuda.jit
def _grid_kernel(u, v, V_real, V_imag, du, dv, VG_real, VG_imag, WG, Npix):
    idx = cuda.grid(1)
    n = u.size
    if idx >= n:
        return

    # Calcular coordenadas de grilla
    i = int(math.floor(u[idx] / du + 0.5)) + Npix // 2
    j = int(math.floor(v[idx] / dv + 0.5)) + Npix // 2

    if i < 0 or i >= Npix or j < 0 or j >= Npix:
        return

    # Peso unitario implícito
    w = 1.0 
    
    cuda.atomic.add(VG_real, (j, i), V_real[idx])
    cuda.atomic.add(VG_imag, (j, i), V_imag[idx])
    cuda.atomic.add(WG, (j, i), w)

# 2. KERNEL CON CONVOLUCIÓN (Sin cambios)
@cuda.jit
def _grid_kernel_KB(u, v, V_real, V_imag, du, dv, kernel, VG_real, VG_imag, WG, Npix):
    idx = cuda.grid(1)
    n = u.size
    if idx >= n:
        return

    center_i = int(math.floor(u[idx] / du + 0.5)) + Npix // 2
    center_j = int(math.floor(v[idx] / dv + 0.5)) + Npix // 2

    if center_i < 0 or center_i >= Npix or center_j < 0 or center_j >= Npix:
        return

    kernel_x, kernel_y = kernel.shape
    half_x = kernel_x // 2
    half_y = kernel_y // 2

    vis_real = V_real[idx]
    vis_imag = V_imag[idx]
    w_in = 1.0

    for k_i in range(kernel_x):
        for k_j in range(kernel_y):
            i = center_i + (k_i - half_x)
            j = center_j + (k_j - half_y)

            if 0 <= i < Npix and 0 <= j < Npix:
                weight = w_in * kernel[k_i, k_j]
                cuda.atomic.add(VG_real, (j, i), weight * vis_real)
                cuda.atomic.add(VG_imag, (j, i), weight * vis_imag)
                cuda.atomic.add(WG, (j, i), weight)

def grid_visibilities_cuda(V, uvw_lambda, du, dv, Npix=256, threads_per_block=1024, conv_kernel=None):
    """
    Grids complex visibilities onto a single (u, v) grid using numba cuda
    
    Salida: 
       VG: (Npix, Npix) complex128
       WG: (Npix, Npix) float64
    """
    
    # Extraemos todas las coordenadas u y v de todas las frecuencias y las hacemos 1D
    u_all = np.ascontiguousarray(uvw_lambda[..., 0].ravel())
    v_all = np.ascontiguousarray(uvw_lambda[..., 1].ravel())
    
    # Extraemos todas las visibilidades complejas y las hacemos 1D
    V_all = np.ascontiguousarray(V.ravel())
    
    # Verificación de integridad
    assert u_all.size == V_all.size, "Error: Dimensiones de UVW y V no coinciden al aplanar."
    
    n_total_points = u_all.size

    # 2. Mover datos a GPU
    d_u = cuda.to_device(u_all)
    d_v = cuda.to_device(v_all)
    d_V_real = cuda.to_device(np.ascontiguousarray(V_all.real))
    d_V_imag = cuda.to_device(np.ascontiguousarray(V_all.imag))
    
    # 3. Reservar memoria para UNA sola imagen acumulada (2D)
    d_VG_real = cuda.device_array((Npix, Npix), dtype=np.float64)
    d_VG_imag = cuda.device_array((Npix, Npix), dtype=np.float64)
    d_WG = cuda.device_array((Npix, Npix), dtype=np.float64)

    # Inicializar a cero
    d_VG_real[:] = 0
    d_VG_imag[:] = 0
    d_WG[:] = 0

    # 4. Lanzar Kernel
    blocks_per_grid = (n_total_points + threads_per_block - 1) // threads_per_block

    if conv_kernel is not None:
        d_kernel = cuda.to_device(np.ascontiguousarray(conv_kernel))
        _grid_kernel_KB[blocks_per_grid, threads_per_block](
            d_u, d_v, d_V_real, d_V_imag, 
            du, dv, d_kernel, d_VG_real, d_VG_imag, d_WG, Npix
        )
    else:
        _grid_kernel[blocks_per_grid, threads_per_block](
            d_u, d_v, d_V_real, d_V_imag, 
            du, dv, d_VG_real, d_VG_imag, d_WG, Npix
        )

    # 5. Copiar resultados de vuelta a CPU
    VG_real = d_VG_real.copy_to_host()
    VG_imag = d_VG_imag.copy_to_host()
    WG = d_WG.copy_to_host() # Ahora es 2D (Npix, Npix)

    # 6. Normalización Final
    VG = np.zeros((Npix, Npix), dtype=np.complex128)
    
    # Evitar división por cero
    mask = WG > 0
    VG[mask] = (VG_real[mask] + 1j * VG_imag[mask]) / WG[mask]
    
    # Retornamos las grillas 2D
    return VG, WG


def to_fourier(visibilities):
    return fftshift(ifft2(ifftshift(visibilities)))
