from numba import cuda
import cupy as cp

# Selección manual opcional
USE_CUPY = True      # fuerza CuPy
USE_NUMBA = True    # o fuerza Numba
# Si ambos son False → modo auto


def get_backend(requested_mode="auto"):
    """
    Devuelve: 'cupy', 'numba', 'numpy'
    según disponibilidad real.
    """
    # --- Forzar numpy ---
    if requested_mode == "numpy":
        return "numpy"

    # --- Forzar cupy ---
    if requested_mode == "cupy":
        try:
            _ = cp.array([1])
            return "cupy"
        except Exception:
            return "numpy"

    # --- Forzar numba ---
    if requested_mode == "numba":
        try:
            if cuda.is_available():
                return "numba"
        except:
            pass
        return "numpy"

    # --- Modo automático ---
    # 1) cupy
    try:
        _ = cp.array([1])
        return "cupy"
    except:
        pass

    # 2) numba
    try:
        if cuda.is_available():
            cuda.detect()
            return "numba"
    except:
        pass

    # 3) fallback
    return "numpy"




# ==================================================
# SELECCIÓN REAL DEL BACKEND
# ==================================================

if USE_CUPY:
    BACKEND = get_backend("cupy")
elif USE_NUMBA:
    BACKEND = get_backend("numba")
else:
    BACKEND = get_backend("auto")


# ==================================================
# IMPORT DINÁMICO: xp y fftconvolve de acuerdo al backend
# ==================================================

if BACKEND == "cupy":
    import cupy as xp
    from cupyx.scipy.signal import fftconvolve

elif BACKEND == "numpy":
    import numpy as xp
    from scipy.signal import fftconvolve

elif BACKEND == "numba":
    import numpy as xp          # arrays CPU
    from scipy.signal import fftconvolve
else:
    import numpy as xp
    from scipy.signal import fftconvolve


# ==================================================
# Funciones auxiliares del backend
# ==================================================
def asnumpy(x):
    """
    Convierte un array del backend a numpy.ndarray.
    - Si BACKEND == "cupy": usa x.get()
    - Si BACKEND == "numpy" o "numba": devuelve x sin cambios
    - Maneja escalares, listas, tuplas
    """
    # Caso: backend cupy
    if BACKEND == "cupy":
        # convertir cupy → numpy
        try:
            return x.get()
        except AttributeError:
            # ya es numpy o un escalar
            return x

    # Caso: numpy o numba
    return x

__all__ = [
    "xp",
    "fftconvolve",
    "asnumpy",
    "BACKEND",
]