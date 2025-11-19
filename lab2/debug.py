# %reload_ext autoreload
# %autoreload 2

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# imports
import numpy as np
# modules
from modules.simulation import visibilities_simulation

# simulate visibilities no grid
VLA_L_4 = {
  "latitude": 34.078749,
  "longitude": -107.617728,
  "file_route": "../antenna_arrays/alma.cycle10.8.cfg",
  "catalog_source": "Sirius",
  "utc_start": "2024-10-21T00:00:00",
  "utc_end": "2024-10-21T06:00:00",
  "step_min": 5,
  "n_freqs": 4,
  "interferometer": {
    "name": "VLA",
    "band_name": "L" },
  "n_sources": 100,
  "max_offset_deg": 1.0,
  "flux_range": [0.1, 2.0],
  "seed": 42
  }

V, uvw_lambda, frequencies, baselines_enu = visibilities_simulation(VLA_L_4)


print('V:', V.shape)
print('uvwlambda:', uvw_lambda.shape)

from modules.coords import max_basline
from modules.interferometry import grid_visibilities

# Resolucion
N = 512
# Distancion maxima entre baselines
Dmax = max_basline(baselines_enu)
oversampling_factor = 3

c = 299792458.0 
freq = np.min(frequencies)
min_wavelenghgt = c / freq
dx = dy = (min_wavelenghgt / Dmax) / oversampling_factor

imgs = []
dus = []

du = 1.0 / (N * dx)
dv = du



import cupy as cp
import numpy as np
from numba import cuda
import math

# =============================================================================
# 1. KERNEL BLINDADO (Anti-NaN y Anti-Segfault)
# =============================================================================
@cuda.jit
def _grid_kernel_final(u, v, V_real, V_imag, omega, du, dv, VG_real, VG_imag, WG, Npix, Nvis):
    idx = cuda.grid(1)
    
    if idx < Nvis:
        # Lectura de coordenadas
        uu = u[idx]
        vv = v[idx]
        
        # CHECK 1: Si el dato es NaN o Infinito, abortamos este hilo suavemente
        # (Evita que int(NaN) genere un segfault)
        if math.isnan(uu) or math.isnan(vv) or math.isinf(uu) or math.isinf(vv):
            return 

        # Cálculo de índices
        # Sumamos 0.5f para redondeo correcto en float32
        i = int(math.floor(uu / du + 0.5)) + Npix // 2
        j = int(math.floor(vv / dv + 0.5)) + Npix // 2

        # CHECK 2: Límites estrictos
        if (i >= 0) and (i < Npix) and (j >= 0) and (j < Npix):
            
            # CHECK 3: Índice plano seguro
            grid_idx = j * Npix + i
            
            # Última barrera de seguridad
            if grid_idx >= 0 and grid_idx < (Npix * Npix):
                w = omega[idx]
                vr = V_real[idx]
                vi = V_imag[idx]
                
                # CHECK 4: Visibilidad corrupta (NaNs en la señal)
                if not (math.isnan(vr) or math.isnan(vi)):
                    cuda.atomic.add(VG_real, grid_idx, w * vr)
                    cuda.atomic.add(VG_imag, grid_idx, w * vi)
                    cuda.atomic.add(WG, grid_idx, w)

@cuda.jit
def _normalize_kernel_final(VG_real, VG_imag, WG, Npix):
    x, y = cuda.grid(2)
    if x < Npix and y < Npix:
        idx = y * Npix + x
        w = WG[idx]
        if w > 0:
            VG_real[idx] /= w
            VG_imag[idx] /= w

# Reemplaza la función original por esta versión de prueba segura:
def grid_visibilities_cuda_final_safe(V, uvw, du, dv, Npix=256, threads_per_block=256):
    import math
    from numba import cuda

    print(f"DEBUG SAFE: du={du}, dv={dv}")
    if du == 0 or math.isnan(du):
        raise ValueError("¡Fatal! 'du' es 0 o NaN.")

    # Convertir en tipos float32/complex64 en host (NumPy) para to_device
    V_np = np.asarray(V).astype(np.complex64)            # host
    uvw_np = np.asarray(uvw).astype(np.float32)         # host

    # Aplanar en host (NumPy)
    u_flat = np.ascontiguousarray(uvw_np[..., 0].ravel())
    v_flat = np.ascontiguousarray(uvw_np[..., 1].ravel())

    V_flat = np.ascontiguousarray(V_np.ravel())
    vis_real = np.ascontiguousarray(V_flat.real.astype(np.float32))
    vis_imag = np.ascontiguousarray(V_flat.imag.astype(np.float32))

    Nvis = u_flat.size
    print(f"DEBUG SAFE: Nvis host = {Nvis}, V_flat size = {vis_real.size}")
    if Nvis != vis_real.size:
        raise ValueError(f"Mismatch fatal host: u={Nvis}, V={vis_real.size}")

    # --- Transfer con numba.cuda (seguro) ---
    u_nb = cuda.to_device(u_flat)
    v_nb = cuda.to_device(v_flat)
    vr_nb = cuda.to_device(vis_real)
    vi_nb = cuda.to_device(vis_imag)

    omega = np.ones(Nvis, dtype=np.float32)
    omega_nb = cuda.to_device(omega)

    VG_real_host = np.zeros(Npix * Npix, dtype=np.float32)
    VG_imag_host = np.zeros(Npix * Npix, dtype=np.float32)
    WG_host = np.zeros(Npix * Npix, dtype=np.float32)

    VG_real_nb = cuda.to_device(VG_real_host)
    VG_imag_nb = cuda.to_device(VG_imag_host)
    WG_nb = cuda.to_device(WG_host)

    # Lanzamiento del kernel
    blocks = (Nvis + threads_per_block - 1) // threads_per_block
    print(f"DEBUG SAFE: Lanzando kernel con blocks={blocks}, tpb={threads_per_block}")

    _grid_kernel_final[blocks, threads_per_block](
        u_nb, v_nb, vr_nb, vi_nb, omega_nb, 
        float(du), float(dv),
        VG_real_nb, VG_imag_nb, WG_nb, int(Npix), int(Nvis)
    )
    cuda.synchronize()

    # Normalización
    t2d = (16, 16)
    bx = (Npix + 15) // 16
    by = (Npix + 15) // 16
    _normalize_kernel_final[(bx, by), t2d](
        VG_real_nb, VG_imag_nb, WG_nb, int(Npix)
    )
    cuda.synchronize()

    # Copiar de vuelta a host
    VG_real_nb.copy_to_host(VG_real_host)
    VG_imag_nb.copy_to_host(VG_imag_host)
    WG_nb.copy_to_host(WG_host)

    VG = VG_real_host.reshape((Npix, Npix)) + 1j * VG_imag_host.reshape((Npix, Npix))
    WG_2d = WG_host.reshape((Npix, Npix))
    return VG, WG_2d

# Ejecución de prueba
print("Iniciando Gridding seguro...")
VG_n, WG_n = grid_visibilities_cuda_final_safe(V, uvw_lambda, du, dv, N)
print(f"Listo. Shape resultante: {VG_n.shape}")