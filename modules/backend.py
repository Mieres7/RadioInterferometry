from numba import cuda
import cupy as cp


def get_backend(requested_mode="auto"):
    """
    Devuelve: 'cupy', 'numba', 'numpy'
    según disponibilidad real (no solo import).
    Evita fallos por CUDA driver en Cupy o Numba.
    """
    # --------------- Forzar backend ---------------
    if requested_mode == "numpy":
        return "numpy"

    if requested_mode == "cupy":
        try:
            _ = cp.array([1])  # fuerza a inicializar CUDA
            return "cupy"
        except Exception as e:
            print(f"[WARN] Cupy solicitado pero no disponible: {e}")
            return "numpy"

    if requested_mode == "numba":
        try:
            # verifica que CUDA esté disponible y funcional
            if cuda.is_available():
                # probar que *realmente* funciona
                # cuda.detect()
                return "numba"
            else:
                raise RuntimeError("Numba CUDA no disponible")
        except Exception as e:
            print(f"[WARN] Numba CUDA solicitado pero no disponible: {e}")
            return "numpy"

    # --------------- Modo automático ---------------
    # 1) CUPY
    try:
        _ = cp.array([1])
        return "cupy"
    except Exception:
        pass

    # 2) NUMBA CUDA
    try:
        
        if cuda.is_available():
            cuda.detect()  # importante: detecta si el driver sirve
            return "numba"
    except Exception:
        pass

    # 3) fallback
    return "numpy"