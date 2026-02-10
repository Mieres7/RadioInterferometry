import numpy as np

def A(u, v, l, m):
    exponent = -2j * np.pi * (np.outer(u, l) + np.outer(v, m))
    return np.exp(exponent)

def compute_model_visibilities(A, image_vector):
    """Calcula V_model = A * I"""
    return A @ image_vector 

def compute_chi_square(v_obs, v_model, sigma):
    """Calcula la función objetivo"""
    return np.sum(np.abs(v_obs - v_model)**2 / sigma**2) 

def compute_residual(A_H, v_obs, v_model):
    """Calcula el residual """
    return A_H @ (v_obs - v_model) 

import numpy as np

def nonlinear_conjugate_gradient(A, v_obs, N, iterations=20, tol=1e-6, reg_lambda=0.1):
    """
    Implementación 100% fiel al Algorithm 1 de la guía. 
    """
    # --- Inicialización --- [cite: 512]
    A_H = A.conj().T # Matriz adjunta [cite: 328, 485]
    I_k = np.zeros(A.shape[1], dtype=complex) # I_0 [cite: 512]
    
    # r0 = A^H * (V_obs - A * I0) [cite: 513]
    # Como I0 es cero, r0 = A^H * V_obs
    r_k = A_H @ (v_obs - (A @ I_k)) 
    p_k = r_k.copy() # p_0 = r_0 [cite: 514]

    # --- Iteración principal --- [cite: 516]
    for k in range(iterations):
        # 1. Verificar convergencia: ||r_k|| > epsilon [cite: 516]
        res_norm = np.linalg.norm(r_k)
        if res_norm <= tol:
            print(f"Convergencia alcanzada en iteración {k}")
            break

        # 2. Calcular alpha_k [cite: 516]
        # alpha_k = (r_k^H * r_k) / (p_k^H * A^H * A * p_k)
        Ap = A @ p_k  
        
        # 2. Calcular alpha_k incluyendo lambda [cite: 516, 536]
        alpha_k = np.vdot(r_k, r_k) / (np.vdot(Ap, Ap) + reg_lambda * np.vdot(p_k, p_k))

        # 3. Actualizar Imagen
        I_k = I_k + alpha_k * p_k
        I_k = np.real(I_k).astype(complex) # Forzar física real

        # 4. Actualizar Residual con el término de regularización 
        r_k_next = r_k - alpha_k * (A_H @ Ap + reg_lambda * p_k)

        # 5. Calcular beta_k
        beta_k = np.vdot(r_k_next, r_k_next) / np.vdot(r_k, r_k)

        # 6. Actualizar dirección
        p_k = r_k_next + beta_k * p_k
        r_k = r_k_next
        
    return np.real(I_k).reshape((N, N))