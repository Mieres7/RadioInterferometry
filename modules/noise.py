import numpy as np
import cupy as cp

def add_gaussian_noise(V, snr_db=20, seed=None):
    """
    Add gaussian noise to visibilities
    """
    xp = cp.get_array_module(V)
    
    if seed is not None:
        if xp == cp:
            cp.random.seed(seed)
        else:
            np.random.seed(seed)
            
    
    signal_power = xp.mean(xp.abs(V)**2)
    
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    
    sigma = xp.sqrt(noise_power / 2.0)
    
    # noise
    noise_real = xp.random.normal(loc=0.0, scale=sigma, size=V.shape)
    noise_imag = xp.random.normal(loc=0.0, scale=sigma, size=V.shape)
    
    V_noisy = V + (noise_real + 1j * noise_imag)
    
    print(f"Ruido agregado | SNR: {snr_db}dB | Sigma: {sigma:.2e}")
    return V_noisy

def add_thermal_noise(V, T_sys, delta_nu, tau):
    """
    Añade ruido térmico a las visibilidades según la Ec. 17.
    
    Parámetros:
    V: Array de visibilidades complejas.
    T_sys: Temperatura del sistema en Kelvin (K)[cite: 213].
    delta_nu: Ancho de banda del canal en Hertz (Hz)[cite: 215].
    tau: Tiempo de integración en segundos (s).
    """
    # Constante de Boltzmann [cite: 212]
    kb = 1.38e-23 
    
    # Cálculo de la desviación estándar del ruido (RMS) en Janskys (Jy)
    # sigma = (2 * kb * Tsys) / sqrt(delta_nu * tau) 
    sigma_raw = (2 * kb * T_sys) / np.sqrt(delta_nu * tau)
    sigma_thermal = sigma_raw / 1e-26

    # El ruido térmico es gausiano y complejo. 
    # Se genera para la parte real e imaginaria por separado.
    noise_real = np.random.normal(0, sigma_thermal / np.sqrt(2), V.shape)
    noise_imag = np.random.normal(0, sigma_thermal / np.sqrt(2), V.shape)
    
    V_noisy = V + (noise_real + 1j * noise_imag)
    
    return V_noisy, sigma_thermal


# def narrowband_rfi(
#     V,
#     frequencies,
#     amplitude,
#     f_rfi=None,
#     k_rfi=None,
#     phase_mode="random",
#     seed=None
# ):
#     """
#     Inject narrowband RFI:
#     V_RFI = A_RFI * exp(i phi) * delta(f - f_RFI)

#     Either f_rfi (Hz) or k_rfi (channel index) must be provided.
#     """

#     rng = np.random.default_rng(seed)
#     frequencies = np.asarray(frequencies)

#     N_baselines, N_times, N_freqs = V.shape

#     # --- sanity checks ---
#     if f_rfi is not None and k_rfi is not None:
#         raise ValueError("Provide only one of f_rfi or k_rfi")

#     if f_rfi is None and k_rfi is None:
#         raise ValueError("Either f_rfi or k_rfi must be provided")

#     # --- normalize inputs to arrays ---
#     if k_rfi is not None:
#         k_list = np.atleast_1d(k_rfi).astype(int)
#         if np.any((k_list < 0) | (k_list >= N_freqs)):
#             raise ValueError(f"k_rfi must be in [0, {N_freqs-1}]")
#         f_list = frequencies[k_list]
#     else:
#         f_list = np.atleast_1d(f_rfi)
#         k_list = np.array([
#             np.argmin(np.abs(frequencies - f)) for f in f_list
#         ])

#     n_rfi = len(k_list)

#     # --- amplitude handling ---
#     amplitude = np.asarray(amplitude)

#     # Case 1: random amplitude range (amin, amax)
#     if amplitude.ndim == 1 and amplitude.size == 2:
#         amin, amax = amplitude
#         if amin < 0 or amax < 0 or amax <= amin:
#             raise ValueError("amplitude range must be (amin, amax) with 0 <= amin < amax")

#         # one random amplitude per RFI channel
#         amplitude = rng.uniform(amin, amax, size=n_rfi)

#     # Case 2: scalar amplitude
#     elif amplitude.size == 1:
#         amplitude = np.repeat(amplitude.item(), n_rfi)

#     # Case 3: one amplitude per RFI channel
#     elif amplitude.size == n_rfi:
#         pass

#     else:
#         raise ValueError(
#             "amplitude must be scalar, (amin, amax), or same length as RFI channels"
#         )


#     # --- phase ---
#     if phase_mode == "random":
#         phase = rng.uniform(
#             0, 2*np.pi,
#             size=(N_baselines, N_times, n_rfi)
#         )
#     elif phase_mode == "constant":
#         phase = np.zeros((N_baselines, N_times, n_rfi))
#     else:
#         raise ValueError("phase_mode must be 'random' or 'constant'")

#     rfi_signal = amplitude[None, None, :] * np.exp(1j * phase)

#     V_rfi = np.zeros_like(V)
#     for i, k in enumerate(k_list):
#         V_rfi[:, :, k] += rfi_signal[:, :, i]

#     return V_rfi, k_list, f_list


def narrowband_rfi(
    V,
    frequencies,
    amplitude,
    f_rfi=None,
    k_rfi=None,
    phase_mode="random",
    duty_cycle=1.0,
    seed=None
):
    """
    Inject narrowband RFI:
    V_RFI = A_RFI * exp(i phi) * delta(f - f_RFI)

    duty_cycle: fraction of time integrations affected by RFI
    """

    rng = np.random.default_rng(seed)
    frequencies = np.asarray(frequencies)

    N_baselines, N_times, N_freqs = V.shape

    if not (0.0 <= duty_cycle <= 1.0):
        raise ValueError("duty_cycle must be between 0 and 1")

    # --- sanity checks ---
    if f_rfi is not None and k_rfi is not None:
        raise ValueError("Provide only one of f_rfi or k_rfi")

    if f_rfi is None and k_rfi is None:
        raise ValueError("Either f_rfi or k_rfi must be provided")

    # --- normalize inputs to arrays ---
    if k_rfi is not None:
        k_list = np.atleast_1d(k_rfi).astype(int)
        if np.any((k_list < 0) | (k_list >= N_freqs)):
            raise ValueError(f"k_rfi must be in [0, {N_freqs-1}]")
        f_list = frequencies[k_list]
    else:
        f_list = np.atleast_1d(f_rfi)
        k_list = np.array([
            np.argmin(np.abs(frequencies - f)) for f in f_list
        ])

    n_rfi = len(k_list)

    # --- amplitude handling ---
    amplitude = np.asarray(amplitude)

    if amplitude.ndim == 1 and amplitude.size == 2:
        amin, amax = amplitude
        if amin < 0 or amax <= amin:
            raise ValueError("amplitude range must be (amin, amax)")
        amplitude = rng.uniform(amin, amax, size=n_rfi)

    elif amplitude.size == 1:
        amplitude = np.repeat(amplitude.item(), n_rfi)

    elif amplitude.size == n_rfi:
        pass

    else:
        raise ValueError("Invalid amplitude specification")

    # --- phase ---
    if phase_mode == "random":
        phase = rng.uniform(
            0, 2*np.pi,
            size=(N_baselines, N_times, n_rfi)
        )
    elif phase_mode == "constant":
        phase = np.zeros((N_baselines, N_times, n_rfi))
    else:
        raise ValueError("phase_mode must be 'random' or 'constant'")

    rfi_signal = amplitude[None, None, :] * np.exp(1j * phase)

    # --- duty cycle temporal ---
    if duty_cycle < 1.0:
        time_mask = rng.random(N_times) < duty_cycle
        rfi_signal[:, ~time_mask, :] = 0.0

    V_rfi = np.zeros_like(V)
    for i, k in enumerate(k_list):
        V_rfi[:, :, k] += rfi_signal[:, :, i]

    return V_rfi, k_list, f_list



# def broadband_rfi(
#     V,
#     frequencies,
#     amplitude,
#     f_range=None,
#     k_range=None,
#     phase_mode="random",
#     seed=None
# ):
#     """
#     Inyecta RFI de banda ancha:

#     V_RFI = A_RFI * exp(i phi) * Pi(f_start, f_end)

#     - amplitude puede ser:
#         * escalar
#         * array de largo N_canales
#         * rango (Amin, Amax) -> amplitud aleatoria independiente por canal

#     Se debe proporcionar exactamente uno: f_range (Hz) o k_range (índices).
#     """

#     rng = np.random.default_rng(seed)
#     frequencies = np.asarray(frequencies)

#     N_baselines, N_times, N_freqs = V.shape

#     # --- Verificación de entradas ---
#     if (f_range is not None) == (k_range is not None):
#         raise ValueError("Debe proporcionar exactamente uno: f_range o k_range")

#     # --- Selección de canales (definida SIEMPRE) ---
#     if k_range is not None:
#         k_start, k_end = k_range
#         k_start = max(0, int(k_start))
#         k_end = min(N_freqs - 1, int(k_end))
#         k_indices = np.arange(k_start, k_end + 1, dtype=int)
#     else:
#         f_start, f_end = f_range
#         k_indices = np.where(
#             (frequencies >= f_start) & (frequencies <= f_end)
#         )[0]

#     # --- Salida temprana si no hay canales ---
#     if k_indices.size == 0:
#         return np.zeros_like(V), k_indices, frequencies[k_indices]

#     n_chan = k_indices.size

#     # --- Manejo de Amplitud ---
#     amplitude = np.asarray(amplitude)

#     # Rango aleatorio (Amin, Amax)
#     if amplitude.ndim == 1 and amplitude.size == 2:
#         amin, amax = amplitude
#         if amin < 0 or amax <= amin:
#             raise ValueError("amplitude range debe cumplir 0 <= Amin < Amax")
#         amplitude = rng.uniform(amin, amax, size=n_chan)

#     # Escalar
#     elif amplitude.size == 1:
#         amplitude = np.full(n_chan, amplitude.item())

#     # Una amplitud por canal
#     elif amplitude.size == n_chan:
#         pass

#     else:
#         raise ValueError(
#             "amplitude debe ser escalar, (Amin, Amax) o array de largo N_canales"
#         )

#     amplitude = amplitude[None, None, :]  # (1, 1, n_chan)

#     # --- Manejo de Fase ---
#     if phase_mode == "random":
#         phase = rng.uniform(
#             0, 2 * np.pi,
#             size=(N_baselines, N_times, n_chan)
#         )
#     elif phase_mode == "constant":
#         phase = np.zeros((N_baselines, N_times, n_chan))
#     else:
#         raise ValueError("phase_mode debe ser 'random' o 'constant'")

#     # --- Señal RFI ---
#     rfi_signal = amplitude * np.exp(1j * phase)

#     V_rfi = np.zeros_like(V)
#     V_rfi[:, :, k_indices] = rfi_signal

#     return V_rfi, k_indices, frequencies[k_indices]

def broadband_rfi(
    V,
    frequencies,
    amplitude,
    f_range=None,
    k_range=None,
    phase_mode="random",
    duty_cycle=1.0,
    seed=None
):
    """
    Inyecta RFI de banda ancha:

    V_RFI = A_RFI * exp(i phi) * Pi(f_start, f_end)

    - amplitude puede ser:
        * escalar
        * array de largo N_canales
        * rango (Amin, Amax) -> amplitud aleatoria independiente por canal

    duty_cycle ∈ [0, 1]: fracción de tiempos afectados por RFI

    Se debe proporcionar exactamente uno: f_range (Hz) o k_range (índices).
    """

    rng = np.random.default_rng(seed)
    frequencies = np.asarray(frequencies)

    N_baselines, N_times, N_freqs = V.shape

    # --- sanity check duty cycle ---
    if not (0.0 <= duty_cycle <= 1.0):
        raise ValueError("duty_cycle debe estar entre 0 y 1")

    # --- Verificación de entradas ---
    if (f_range is not None) == (k_range is not None):
        raise ValueError("Debe proporcionar exactamente uno: f_range o k_range")

    # --- Selección de canales ---
    if k_range is not None:
        k_start, k_end = k_range
        k_start = max(0, int(k_start))
        k_end = min(N_freqs - 1, int(k_end))
        k_indices = np.arange(k_start, k_end + 1, dtype=int)
    else:
        f_start, f_end = f_range
        k_indices = np.where(
            (frequencies >= f_start) & (frequencies <= f_end)
        )[0]

    if k_indices.size == 0:
        return np.zeros_like(V), k_indices, frequencies[k_indices]

    n_chan = k_indices.size

    # --- Manejo de Amplitud ---
    amplitude = np.asarray(amplitude)

    if amplitude.ndim == 1 and amplitude.size == 2:
        amin, amax = amplitude
        if amin < 0 or amax <= amin:
            raise ValueError("amplitude range debe cumplir 0 <= Amin < Amax")
        amplitude = rng.uniform(amin, amax, size=n_chan)

    elif amplitude.size == 1:
        amplitude = np.full(n_chan, amplitude.item())

    elif amplitude.size == n_chan:
        pass

    else:
        raise ValueError(
            "amplitude debe ser escalar, (Amin, Amax) o array de largo N_canales"
        )

    amplitude = amplitude[None, None, :]  # (1, 1, n_chan)

    # --- Manejo de Fase ---
    if phase_mode == "random":
        phase = rng.uniform(
            0, 2 * np.pi,
            size=(N_baselines, N_times, n_chan)
        )
    elif phase_mode == "constant":
        phase = np.zeros((N_baselines, N_times, n_chan))
    else:
        raise ValueError("phase_mode debe ser 'random' o 'constant'")

    # --- Señal RFI ---
    rfi_signal = amplitude * np.exp(1j * phase)

    # --- duty cycle temporal ---
    if duty_cycle < 1.0:
        time_mask = rng.random(N_times) < duty_cycle  # (N_times,)
        rfi_signal[:, ~time_mask, :] = 0.0

    V_rfi = np.zeros_like(V)
    V_rfi[:, :, k_indices] = rfi_signal

    return V_rfi, k_indices, frequencies[k_indices]


def transient_rfi(
    V,
    times,
    amplitude,
    t_rfi=None,
    n_step=None,
    duration_steps=1,
    phase_mode="random",
    seed=None
):
    """
    Inyecta múltiples RFI transitorios.

    V_RFI(t) = A_RFI * exp(i phi) * delta(t - t_RFI)

    Parámetros:
    - t_rfi  : float o iterable de floats (tiempos)
    - n_step : int o iterable de ints (índices)
    """

    rng = np.random.default_rng(seed)
    times = np.asarray(times)
    N_baselines, N_times, N_freqs = V.shape

    # --- Verificación de entradas ---
    if (t_rfi is not None and n_step is not None) or (t_rfi is None and n_step is None):
        raise ValueError("Debe proporcionar exactamente uno: t_rfi o n_step")

    # --- Normalizar a listas ---
    if t_rfi is not None:
        t_rfi_list = np.atleast_1d(t_rfi)
        idx_list = [
            np.argmin(np.abs(times - t)) for t in t_rfi_list
        ]
    else:
        idx_list = np.atleast_1d(n_step)

    V_rfi = np.zeros_like(V)
    all_t_indices = []

    # --- Aplicar cada transiente ---
    for idx_start in idx_list:
        if idx_start < 0 or idx_start >= N_times:
            continue

        t_indices = np.arange(
            idx_start,
            min(N_times, idx_start + duration_steps)
        )

        if len(t_indices) == 0:
            continue

        all_t_indices.append(t_indices)

        # --- Fase ---
        if phase_mode == "random":
            phase = rng.uniform(
                0, 2 * np.pi,
                size=(N_baselines, len(t_indices), N_freqs)
            )
        elif phase_mode == "constant":
            phase = np.zeros((N_baselines, len(t_indices), N_freqs))
        else:
            raise ValueError("phase_mode debe ser 'random' o 'constant'")

        rfi_signal = amplitude * np.exp(1j * phase)

        V_rfi[:, t_indices, :] += rfi_signal

    if len(all_t_indices) == 0:
        return V_rfi, np.array([], dtype=int)

    return V_rfi, np.unique(np.concatenate(all_t_indices))


def correlated_rfi(
    V,
    uvw_lambda,
    amplitude,
    correlation_length=100.0,
    phase_mode="random",
    seed=None
):
    """
    Inyecta RFI Correlacionada Espacialmente:
    V_RFI(u,v) = A_RFI * exp(i phi) * C(u,v) [cite: 285]
    """
    rng = np.random.default_rng(seed)
    # V shape: (2016, 601, 150) -> (baselines, times, freqs)
    # uvw_lambda shape: (2016, 601, 150, 3)
    N_baselines, N_times, N_freqs = V.shape
    
    # Extraemos u y v (u=0, v=1 en la última dimensión)
    u = uvw_lambda[..., 0] # (2016, 601, 150)
    v = uvw_lambda[..., 1] # (2016, 601, 150)
    
    # Calcular la distancia UV en longitudes de onda
    uv_distance = np.sqrt(u**2 + v**2)
    
    # Función de correlación espacial C(u,v) [cite: 286]
    # La correlación es alta para baselines cercanos (u,v pequeños) 
    C_uv = np.exp(-(uv_distance**2) / (2 * correlation_length**2))
    
    # --- Manejo de Fase ---
    if phase_mode == "random":
        phase = rng.uniform(0, 2*np.pi, size=(N_baselines, N_times, N_freqs))
    elif phase_mode == "constant":
        phase = np.zeros((N_baselines, N_times, N_freqs))
    else:
        raise ValueError("phase_mode debe ser 'random' o 'constant'")

    # --- Generación de la señal V_RFI ---
    # Ahora C_uv ya tiene el mismo shape (2016, 601, 150) que la fase y amplitud
    V_rfi = amplitude * np.exp(1j * phase) * C_uv
    
    return V_rfi, C_uv