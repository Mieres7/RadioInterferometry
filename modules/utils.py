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
from modules.interferometry import grid_visibilities, grid_visibilities_cuda

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
    Ejecuta el benchmark comparativo entre CPU, CuPy y Numba.
    Retorna un DataFrame con los resultados.
    """
    results = []
    
    # Pre-calentamiento para Numba (compilación JIT)
    print("Calentando kernels de Numba...")
    try:
        # Ejecución dummy pequeña
        grid_visibilities_cuda(V[:100], uvw[:100], du, dv, Npix=256)
    except Exception as e:
        print(f"Warning: Warm-up falló ({e}), la primera medición puede ser lenta.")

    for N in grid_sizes:
        print(f"--- Midiendo para N = {N}x{N} ---")
        
        # 1. CPU (NumPy)
        # Nota: Si N es muy grande, la CPU puede tardar mucho. Puedes limitar N para CPU.
        if N <= 1024: 
            start = time.perf_counter()
            _, _ = grid_visibilities(V, uvw, du, dv, Npix=N, use_gpu=False)
            end = time.perf_counter()
            time_cpu = end - start
        else:
            time_cpu = np.nan # Omitir para ahorrar tiempo
            
        # 2. GPU (CuPy Vectorizado)
        # Sincronizamos antes de empezar
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        _, _ = grid_visibilities(V, uvw, du, dv, Npix=N, use_gpu=True)
        # Sincronizamos antes de terminar para asegurar que la GPU terminó
        cp.cuda.Stream.null.synchronize()
        end = time.perf_counter()
        time_cupy = end - start
        
        # 3. GPU (Numba Kernels)
        # Nota: Tu función grid_visibilities_cuda ya tiene las transferencias
        # incluidas, así que es una comparación justa "end-to-end".
        cuda.synchronize()
        start = time.perf_counter()
        _, _ = grid_visibilities_cuda(V, uvw, du, dv, Npix=N)
        cuda.synchronize()
        end = time.perf_counter()
        time_numba = end - start
        
        # Guardar resultados
        results.append({
            "Grid Size": N,
            "CPU (s)": time_cpu,
            "CuPy (s)": time_cupy,
            "Numba (s)": time_numba,
            "Speedup Numba vs CPU": time_cpu / time_numba if time_cpu > 0 else 0
        })
        
        print(f"   CPU: {time_cpu:.4f}s | CuPy: {time_cupy:.4f}s | Numba: {time_numba:.4f}s")

    df = pd.DataFrame(results)
    return df

def plot_benchmark_results(df):
    """
    Grafica los tiempos de ejecución y el Speedup.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico 1: Tiempos de Ejecución (Escala Logarítmica)
    # Usamos log porque la diferencia entre CPU y GPU suele ser enorme
    df.plot(x="Grid Size", y=["CPU (s)", "CuPy (s)", "Numba (s)"], 
            kind="bar", ax=axes[0], logy=True)
    axes[0].set_title("Tiempo de Ejecución (Escala Log)")
    axes[0].set_ylabel("Tiempo (segundos)")
    axes[0].grid(True, which="both", linestyle="--", alpha=0.5)
    
    # Gráfico 2: Speedup (Factor de Aceleración)
    # Calculamos Speedup vs CuPy también para ver cuál GPU gana
    df["Speedup Numba vs CuPy"] = df["CuPy (s)"] / df["Numba (s)"]
    
    df.plot(x="Grid Size", y=["Speedup Numba vs CuPy"], 
            kind="bar", ax=axes[1], color="orange")
    axes[1].set_title("Speedup: Numba Kernels vs CuPy Vectorizado")
    axes[1].set_ylabel("Factor de Aceleración (X veces más rápido)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.show()

# --- EJECUCIÓN DEL TEST ---
# Asegúrate de usar tus variables reales (V, uvw_lambda, etc.)
# df_results = benchmark_gridding(V_con_ruido, uvw_lambda, du, dv, grid_sizes=[256, 512, 1024, 2048])
# print(df_results)
# plot_benchmark_results(df_results)