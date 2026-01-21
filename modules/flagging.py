import numpy as np
from scipy.signal import convolve

def mad_threshold(x, k=5.0, axis=None):
    """
    Calcula umbral estadístico robusto: median + k * MAD
    """
    med = np.median(x, axis=axis)
    mad = np.median(np.abs(x - med), axis=axis)
    return med + k * (mad + 1e-12)

def sliding_sum_freq(amp, w):
    """
    Suma amplitudes en ventanas de frecuencia.

    Parameters
    ----------
    amp : ndarray, shape (B, T, F)
        Amplitudes |V|
    w : int
        Tamaño de ventana

    Returns
    -------
    S : ndarray, shape (B, T, F-w+1)
        Sumas por ventana
    """
    return np.lib.stride_tricks.sliding_window_view(
        amp, window_shape=w, axis=2
    ).sum(axis=-1)


def expand_flags(exceed, w, F):
    """
    Expande flags de ventanas (B, T, F-w+1) a flags por canal (B, T, F)
    usando broadcasting seguro.

    Parameters
    ----------
    exceed : bool ndarray, shape (B, T, F-w+1)
        Ventanas que superan el umbral
    w : int
        Tamaño de la ventana
    F : int
        Número total de canales

    Returns
    -------
    flags : bool ndarray, shape (B, T, F)
    """
    B, T, Fw = exceed.shape
    flags = np.zeros((B, T, F), dtype=bool)

    for i in range(w):
        flags[..., i:i + Fw] |= exceed

    return flags

def sumthreshold_frequency(
    V,
    window_sizes=(1, 2, 4, 8),
    k=5.0
):
    """
    SumThreshold en frecuencia (optimizado por broadcasting).

    Parameters
    ----------
    V : complex ndarray, shape (B, T, F)
    window_sizes : iterable of int
    k : float

    Returns
    -------
    flags : bool ndarray, shape (B, T, F)
    """
    amp = np.abs(V)
    B, T, F = amp.shape
    flags = np.zeros((B, T, F), dtype=bool)

    for w in window_sizes:
        if w > F:
            continue

        S = sliding_sum_freq(amp, w)
        threshold = mad_threshold(S, k=k)
        exceed = S > threshold

        flags |= expand_flags(exceed, w, F)

    return flags

# -------------------------

def statistical_threshold(
    V,
    k=5.0
):
    """
    Flagging por umbral estadístico robusto en frecuencia.

    Parameters
    ----------
    V : complex ndarray, shape (B, T, F)
        Visibilidades complejas
    k : float
        Factor MAD

    Returns
    -------
    flags : bool ndarray, shape (B, T, F)
        True = RFI
    """
    amp = np.abs(V)
    med = np.median(amp, axis=2, keepdims=True)
    mad = np.median(np.abs(amp - med), axis=2, keepdims=True)

    threshold = med + k * (mad + 1e-12)

    flags = amp > threshold

    return flags


# ------------

def amplitude_variation_flagging(vis_3d, u, v, n_bins=20, k=4.0):
    """
    Implementa flagging por variación de amplitud segmentado por bins de distancia UV.
    vis_3d, u, v: Arreglos de forma (baselines, times, channels).
    """
    # 1. Calcular distancia UV y amplitudes
    dist_uv = np.sqrt(u**2 + v**2)
    amplitudes = np.abs(vis_3d)
    
    # 2. Aplanar los arreglos para procesar todos los puntos simultáneamente
    dist_flat = dist_uv.ravel()
    amp_flat = amplitudes.ravel()
    
    # 3. Crear los bins de distancia UV
    bins = np.linspace(dist_flat.min(), dist_flat.max(), n_bins + 1)
    # Asignar cada punto a un bin (0 a n_bins-1)
    bin_indices = np.digitize(dist_flat, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # 4. Inicializar la máscara de salida
    flag_mask_flat = np.zeros_like(amp_flat, dtype=bool)
    
    # 5. Calcular umbrales por bin
    for i in range(n_bins):
        # Máscara 1D para los puntos que pertenecen al bin i
        idx_in_bin = (bin_indices == i)
        
        if np.any(idx_in_bin):
            data_bin = amp_flat[idx_in_bin]
            
            # Estadísticos locales del bin
            med_bin = np.median(data_bin)
            mad_bin = np.median(np.abs(data_bin - med_bin))
            
            # Umbral estadístico: med + k*MAD
            threshold = med_bin + k * mad_bin
            
            # Marcar outliers dentro del bin
            flag_mask_flat[idx_in_bin] = data_bin > threshold
            
    # 6. Reestructurar la máscara a la forma original (B, T, C)
    return flag_mask_flat.reshape(vis_3d.shape)


# --------------

def phase_coherence_flagging(vis_3d, w, threshold_coh=0.3):
    """
    Implementa el flagging por coherencia de fase
    vis_3d: Cubo de visibilidades complejas.
    w: Tamaño de la ventana para calcular la coherencia.
    threshold_coh: Umbral theta_coh
    """
    phases_unit = vis_3d / np.abs(vis_3d)
    
    kernel = np.ones((1, 1, w))
    sum_phases = convolve(phases_unit, kernel, mode='same')
    
    coherence = np.abs(sum_phases / w)
    
    mask = coherence < threshold_coh
    
    return mask


# --------- Metrics -----------

def calculate_rfi_metrics(predicted_mask, true_mask):
    """
    Calcula las métricas de evaluación para la detección de RFI[cite: 427, 439].
    """
    # Forzar a booleano para evitar el TypeError
    pred = np.asarray(predicted_mask).astype(bool)
    true = np.asarray(true_mask).astype(bool)
    
    # 1. Calcular componentes básicos (TP, FP, FN) [cite: 429, 430, 431]
    tp = np.sum(pred & true)      # Verdaderos Positivos: RFI detectada correctamente [cite: 429]
    fp = np.sum(pred & ~true)     # Falsos Positivos: Datos limpios marcados como RFI [cite: 430]
    fn = np.sum(~pred & true)     # Falsos Negativos: RFI no detectada [cite: 431]
    
    # 2. Calcular métricas finales [cite: 428]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
    return {
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "TP": tp,
        "FP": fp,
        "FN": fn
    }