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