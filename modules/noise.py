# En modules/simulation.py

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