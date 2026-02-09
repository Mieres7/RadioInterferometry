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

def nonlinear_conjugate_gradient(A, v_obs, N, iterations=20, tol=1e-6):
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
    
    print(f"Iniciando Algoritmo 1 (Identico a la guia)...")

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
        alpha_k = np.vdot(r_k, r_k) / np.vdot(Ap, Ap) # El denominador es equivalente a p^H A^H A p

        # 3. Actualizar Imagen: I_{k+1} = I_k + alpha_k * p_k [cite: 516]
        I_k = I_k + alpha_k * p_k

        # 4. Actualizar Residual: r_{k+1} = r_k - alpha_k * A^H * A * p_k 
        # Esta es la parte que NO usa la imagen I directamente en la fórmula
        r_k_next = r_k - alpha_k * (A_H @ Ap)

        # 5. Calcular beta_k [cite: 519]
        # beta_k = (r_{k+1}^H * r_{k+1}) / (r_k^H * r_k)
        beta_k = np.vdot(r_k_next, r_k_next) / np.vdot(r_k, r_k)

        # 6. Actualizar dirección: p_{k+1} = r_{k+1} + beta_k * p_k [cite: 520]
        p_k = r_k_next + beta_k * p_k
        r_k = r_k_next
        
        # (Opcional) Monitoreo de la función objetivo [cite: 501]
        if k % 5 == 0:
            print(f"Iteración {k}: Norm Residual = {res_norm:.4e}")

    # --- Resultado --- [cite: 521, 522]
    return I_k.reshape((N, N))