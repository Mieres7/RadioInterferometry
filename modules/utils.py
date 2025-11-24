"""
Utilidades Generales
"""

import numpy as np
import cupy as cp

from modules.coords import ecef_to_enu

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