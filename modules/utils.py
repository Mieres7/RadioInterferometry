"""
Utilidades Generales
"""

import numpy as np
import cupy as cp
import pandas as pd
import time
import matplotlib.pyplot as plt
from numba import cuda

from modules.coords import ecef_to_enu
from modules.gridder import grid_visibilities

def degree_to_time(theta, is_rad=False):
    if is_rad:
        theta = np.rad2deg(theta)
    h = int(theta / 15)
    m = int(((theta / 15) - h) * 60)
    s = ((((theta / 15) - h) * 60) - m) * 60
    return h, m, s


def read_cfg_to_enu(filename, array_center=None ,phi=-33.44, lamb=-70.76, rad=True):
  '''
  Read file and return antenna config on ENU coords
  '''
  with open(filename, "r") as f:
    lines = f.readlines()

  coordsys = None 
  for line in lines:
        if line.startswith("# coordsys"):
            coordsys = line.split("=")[1].strip()
            break
  
  antennas = []
  for line in lines:
      if line.startswith("#") or not line.strip():
          continue
      parts = line.split()
      x, y, z = map(float, parts[:3])
      antennas.append([x, y, z])
  antennas = np.array(antennas)

  if coordsys == "LOC (local tangent plane)": return antennas.T
  elif coordsys == "XYZ":
     array_center = array_center if array_center is not None else antennas.mean(axis=0)
     enu_antennas = ecef_to_enu(antennas, array_center, phi, lamb, rad)
     return np.array(enu_antennas)
  else:
    raise ValueError(f"coordsys desconocido: {coordsys}")

def compare_arrays(arr1, arr2, rtol=1e-5, atol=1e-8, name="Arreglos"):
    """
    Compara dos arreglos (NumPy o CuPy) para ver si son equivalentes.
    
    Parámetros:
    - arr1, arr2: Arreglos a comparar (pueden ser numpy.ndarray o cupy.ndarray).
    - rtol: Tolerancia relativa (por defecto 1e-5 para float32).
    - atol: Tolerancia absoluta.
    - name: Nombre para identificar la comparación en los prints.
    
    Retorna:
    - True si son iguales (dentro de la tolerancia), False si no.
    """
    
    # 1. Normalizar a CPU (NumPy)
    # Si es CuPy, usamos .get() o cp.asnumpy(). Si es NumPy, lo dejamos igual.
    a1 = cp.asnumpy(arr1) if hasattr(arr1, 'device') else arr1
    a2 = cp.asnumpy(arr2) if hasattr(arr2, 'device') else arr2
    
    # 2. Verificar formas (Shapes)
    if a1.shape != a2.shape:
        print(f"❌ {name}: ERROR DE DIMENSIÓN.")
        print(f"   Shape 1: {a1.shape} | Shape 2: {a2.shape}")
        return False

    # 3. Comparación con tolerancia (np.allclose)
    # equal_nan=True considera que NaN == NaN es verdadero (útil si hay datos faltantes)
    are_close = np.allclose(a1, a2, rtol=rtol, atol=atol, equal_nan=True)
    
    if are_close:
        print(f"✅ {name}: ÉXITO. Los arreglos coinciden.")
        return True
    else:
        # 4. Diagnóstico de error si fallan
        diff = np.abs(a1 - a2)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"❌ {name}: FALLÓ. Diferencias encontradas.")
        print(f"   Máxima diferencia absoluta: {max_diff:.2e}")
        print(f"   Diferencia media: {mean_diff:.2e}")
        print(f"   Tolerancia usada (rtol): {rtol}")
        return False
    

def benchmark_gridding(V, uvw, du, dv, grid_sizes=[256, 512, 1024, 2048]):
    """
    Ejecuta un benchmark comparativo entre CPU, CuPy y Numba,
    incluyendo speedups relativos entre métodos.
    """
    results = []
    
    # Pre-calentamiento / JIT warm-up para Numba
    print("Calentando kernels de Numba...")
    try:
        grid_visibilities(V[:100], uvw[:100], du, dv, Npix=256, mode='numba')
    except Exception as e:
        print(f"Warning: Warm-up falló ({e}), la primera medición puede ser más lenta.")

    for N in grid_sizes:
        print(f"--- Midiendo para N = {N}x{N} ---")
        
        # ======================
        # 1. CPU
        # ======================
        start = time.perf_counter()
        _, _ = grid_visibilities(V, uvw, du, dv, Npix=N, mode='numpy')
        end = time.perf_counter()
        time_cpu = end - start
        
        # ======================
        # 2. CuPy (GPU)
        # ======================
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        _, _ = grid_visibilities(V, uvw, du, dv, Npix=N, mode='cupy')
        cp.cuda.Stream.null.synchronize()
        end = time.perf_counter()
        time_cupy = end - start
        
        # ======================
        # 3. Numba CUDA (GPU)
        # ======================
        cuda.synchronize()
        start = time.perf_counter()
        _, _ = grid_visibilities(V, uvw, du, dv, Npix=N,mode='numba')
        cuda.synchronize()
        end = time.perf_counter()
        time_numba = end - start
        
        # ======================
        # Guarda resultados
        # ======================
        results.append({
            "Grid Size": N,
            "CPU (s)": time_cpu,
            "CuPy (s)": time_cupy,
            "Numba (s)": time_numba,

            # Speedups básicos
            "Speedup Numba vs CPU": time_cpu / time_numba if time_numba > 0 else 0,
            "Speedup CuPy vs CPU":  time_cpu / time_cupy if time_cupy > 0 else 0,
            "Speedup Numba vs CuPy": time_cupy / time_numba if time_numba > 0 else 0
        })
        
        print(f"   CPU: {time_cpu:.4f}s | CuPy: {time_cupy:.4f}s | Numba: {time_numba:.4f}s")

    return pd.DataFrame(results)

def calculate_quality_metrics(I_restored, dirty_residuals):
    """
    Calcula MAD STD y PSNR según las ecuaciones 18 y 19 del laboratorio.
    
    Parámetros:
    -----------
    I_restored : cupy.ndarray
        La imagen final restaurada (I_model * beam + residuals).
    dirty_residuals : cupy.ndarray
        La imagen 'dirty' de los residuos (Adjoint(V_obs - V_pred)).
        
    Retorna:
    --------
    mad_std : float
        Estimación robusta de la desviación estándar del ruido.
    psnr : float
        Peak Signal-to-Noise Ratio (lineal, según Eq. 19).
    """
    
    # Asegurarnos de usar solo la parte real
    # El ruido y la intensidad física son cantidades reales
    I_restored_real = I_restored.real
    residuals_real = dirty_residuals.real
    
    # --- 1. MAD STD (Ecuación 18) ---
    # Calcular la mediana de los residuos
    median_resid = cp.median(residuals_real)
    
    # Calcular la desviación absoluta respecto a la mediana
    abs_deviation = cp.abs(residuals_real - median_resid)
    
    # Calcular la Mediana de la Desviación Absoluta (MAD)
    mad = cp.median(abs_deviation)
    
    # Convertir a estimación de STD (Factor 1.4826 para distribución normal)
    mad_std = 1.4826 * mad
    
    # --- 2. PSNR (Ecuación 19) ---
    # El PDF define PSNR como la relación directa entre el Máximo y el Ruido
    peak_signal = cp.max(I_restored_real)
    
    # Evitar división por cero
    if mad_std == 0:
        psnr = float('inf')
    else:
        psnr = peak_signal / mad_std
        
    return mad_std, psnr