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

import math
from numba import cuda
import numpy as np
import cupy as cp

# OPTIMIZACIÓN 1: Usamos fastmath=True para acelerar cálculos matemáticos
@cuda.jit(fastmath=True)
def _grid_kernel(u, v, V_real, V_imag, inv_du, inv_dv, VG_real, VG_imag, WG, Npix):
    idx = cuda.grid(1)
    if idx >= u.size:
        return

    # OPTIMIZACIÓN 2: Multiplicación en vez de división
    # i = int(u / du + 0.5) se convierte en u * inv_du
    u_val = u[idx]
    v_val = v[idx]
    
    i = int(u_val * inv_du + 0.5) + Npix // 2
    j = int(v_val * inv_dv + 0.5) + Npix // 2

    if i < 0 or i >= Npix or j < 0 or j >= Npix:
        return

    w = 1.0 
    
    # OPTIMIZACIÓN 3: Aseguramos que los arrays de destino sean float32 (ver función host)
    cuda.atomic.add(VG_real, (j, i), V_real[idx])
    cuda.atomic.add(VG_imag, (j, i), V_imag[idx])
    cuda.atomic.add(WG, (j, i), w)

@cuda.jit(fastmath=True)
def _grid_kernel_KB(u, v, V_real, V_imag, inv_du, inv_dv, kernel, VG_real, VG_imag, WG, Npix):
    idx = cuda.grid(1)
    if idx >= u.size:
        return

    u_val = u[idx]
    v_val = v[idx]

    # Multiplicación inversa
    center_i = int(u_val * inv_du + 0.5) + Npix // 2
    center_j = int(v_val * inv_dv + 0.5) + Npix // 2

    # Early exit para evitar cálculos innecesarios si el centro está muy lejos
    # (Considerando el radio del kernel para ser seguros)
    kernel_x, kernel_y = kernel.shape
    half_x = kernel_x // 2
    half_y = kernel_y // 2
    
    if center_i < -half_x or center_i >= Npix + half_x or \
       center_j < -half_y or center_j >= Npix + half_y:
        return

    vis_real = V_real[idx]
    vis_imag = V_imag[idx]
    
    # Cacheamos el kernel en registros si es posible (acceso repetido)
    # iterar sobre el kernel pequeño
    for k_i in range(kernel_x):
        for k_j in range(kernel_y):
            i = center_i + (k_i - half_x)
            j = center_j + (k_j - half_y)

            if 0 <= i < Npix and 0 <= j < Npix:
                # kernel[...] acceso es rápido si está en cache L1
                weight = kernel[k_i, k_j] # Asumiendo peso entrada = 1.0
                
                cuda.atomic.add(VG_real, (j, i), weight * vis_real)
                cuda.atomic.add(VG_imag, (j, i), weight * vis_imag)
                cuda.atomic.add(WG, (j, i), weight)

def grid_visibilities_cuda(V, uvw_lambda, du, dv, Npix=256, threads_per_block=1024, conv_kernel=None):
    """
    Versión optimizada usando float32 y pre-cálculo de inversas.
    """
    
    # 1. Aplanado y Conversión a FLOAT32 (Crítico para velocidad en GPUs gamers)
    u_all = np.ascontiguousarray(uvw_lambda[..., 0].ravel(), dtype=np.float32)
    v_all = np.ascontiguousarray(uvw_lambda[..., 1].ravel(), dtype=np.float32)
    
    # Separamos real e imag y convertimos a float32
    V_flat = V.ravel()
    V_real = np.ascontiguousarray(V_flat.real, dtype=np.float32)
    V_imag = np.ascontiguousarray(V_flat.imag, dtype=np.float32)
    
    n_total_points = u_all.size

    # 2. Mover datos a GPU
    d_u = cuda.to_device(u_all)
    d_v = cuda.to_device(v_all)
    d_V_real = cuda.to_device(V_real)
    d_V_imag = cuda.to_device(V_imag)
    
    # Pre-calcular inversas para evitar divisiones en el kernel
    inv_du = np.float32(1.0 / du)
    inv_dv = np.float32(1.0 / dv)

    # 3. Reservar memoria FLOAT32 (El cambio más importante para atomics)
    d_VG_real = cuda.device_array((Npix, Npix), dtype=np.float32)
    d_VG_imag = cuda.device_array((Npix, Npix), dtype=np.float32)
    d_WG = cuda.device_array((Npix, Npix), dtype=np.float32)

    # Inicializar a cero
    d_VG_real[:] = 0
    d_VG_imag[:] = 0
    d_WG[:] = 0

    blocks_per_grid = (n_total_points + threads_per_block - 1) // threads_per_block

    if conv_kernel is not None:
        # Asegurar que el kernel también sea float32
        d_kernel = cuda.to_device(np.ascontiguousarray(conv_kernel, dtype=np.float32))
        _grid_kernel_KB[blocks_per_grid, threads_per_block](
            d_u, d_v, d_V_real, d_V_imag, 
            inv_du, inv_dv, d_kernel, d_VG_real, d_VG_imag, d_WG, Npix
        )
    else:
        _grid_kernel[blocks_per_grid, threads_per_block](
            d_u, d_v, d_V_real, d_V_imag, 
            inv_du, inv_dv, d_VG_real, d_VG_imag, d_WG, Npix
        )

    # 4. Copiar y reconvertir
    VG_real = d_VG_real.copy_to_host()
    VG_imag = d_VG_imag.copy_to_host()
    WG = d_WG.copy_to_host()

    # Reconstrucción en CPU (puede hacerse en complex64 para ahorrar memoria)
    VG = np.zeros((Npix, Npix), dtype=np.complex64)
    mask = WG > 0
    VG[mask] = (VG_real[mask] + 1j * VG_imag[mask]) / WG[mask]
    
    return VG, WG


def to_fourier(visibilities):
    return fftshift(ifft2(ifftshift(visibilities)))
