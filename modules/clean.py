import numpy as np

def clean(
    Idirty,
    gain=0.1,
    threshold=0.05,
    max_iter=1000
):
    """
    Implementación básica del algoritmo Högbom CLEAN.
    
    Parameters
    ----------
    Idirty : ndarray (N, N)
        Dirty image.
    gain : float
        CLEAN gain (0.1 - 0.3 típico).
    threshold : float
        Fracción del pico máximo inicial para parar.
    max_iter : int
        Número máximo de iteraciones.
        
    Returns
    -------
    Iclean : ndarray
        Modelo CLEAN (componentes puntuales).
    R : ndarray
        Imagen residual final.
    """
    
    Iclean = np.zeros_like(Idirty)
    R = Idirty.copy()

    peak_dirty = np.max(np.abs(Idirty))

    for it in range(max_iter):
        # encontrar pico
        idx = np.unravel_index(np.argmax(np.abs(R)), R.shape)
        peak = R[idx]

        # condición de parada
        if np.abs(peak) < threshold * peak_dirty:
            print(f"CLEAN converge en iteración {it}")
            break

        # actualizar modelo y residual
        Iclean[idx] += gain * peak
        R[idx] -= gain * peak

    return Iclean, R
