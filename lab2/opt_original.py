import cupy as cp
import numpy as np
from modules.interferometry import adjoint_op, forward_op
from modules.backend import get_backend

'''
Regularizations
'''
def l1(image):
    '''
    L1 Regularization
    '''
    # Cost Function
    cost = cp.sum(cp.abs(image)) 
    # Gradient
    grad = cp.sign(image) 

    return cost, grad

def tsv(image):
    '''
    Total Squared Variation Regularization
    '''
    # Cost Functions
    diff_x = cp.roll(image, -1, axis=1) - image 
    diff_y = cp.roll(image, -1, axis=0) - image 
    
    cost = cp.sum(diff_x**2 + diff_y**2)
    
    # Gradient
    im1_x = cp.roll(image, 1, axis=1)
    im1_y = cp.roll(image, 1, axis=0)
    
    ip1_x = cp.roll(image, -1, axis=1)
    ip1_y = cp.roll(image, -1, axis=0)
    
    lap_x = (ip1_x - image) - (image - im1_x)
    lap_y = (ip1_y - image) - (image - im1_y)
    
    grad = -2.0 * (lap_x + lap_y)
    
    return cost, grad
    
def entropy(image):

    epsilon = 1

    log_term = cp.log(image + epsilon)
    
    # Cost function
    cost = cp.sum(image * log_term - image)
    # Gradient
    grad = log_term
    
    return cost, grad

'''
Objective function
'''
def obj_function(image, V_obs, weights, reg_lambda, reg_func):

    backend = get_backend('cupy')
    if backend == 'cupy':
        sys = cp
    else:
        sys = np

    V_pred = forward_op(Image=image, gridded=True)
    residual = V_pred - V_obs

    # cost
    f_cost = 0.5 * sys.sum(weights * sys.abs(residual)**2)
    # gradient
    grad = adjoint_op(weights * residual)
    grad = grad.real

    reg_cost, reg_grad = reg_func(image)

    total_cost = f_cost + reg_lambda * reg_cost
    total_grad = grad + reg_lambda * reg_grad

    return total_cost.real, total_grad
    

'''
Line Search
'''
def armijo_line_search(image, direction, grad, current_cost, V_obs, weights, reg_lambda, reg_func,
                       alpha=1.0, rho=0.5, c1=1e-4):
    
    grad_dot_dir = cp.sum(grad * direction)
    
    if grad_dot_dir > 0:
        return 1e-5

    while True:
        # 1. Try new image
        image_new = image + alpha * direction
        
        # 2. Obtain new cost
        cost_new, _ = obj_function(image_new, V_obs, weights, reg_lambda, reg_func)
        
        # 3. Armijo's condition
        if cost_new <= current_cost + c1 * alpha * grad_dot_dir:
            return alpha
            
        alpha *= rho
        
        if alpha < 1e-10: 
            return 1e-10

    

'''
LBFGS algorithm
'''
def lbfgs_optimize(image_init, V_obs, weights, reg_lambda, reg_func, 
                   max_iter=100, m=10, tol=1e-5):
    """
    Algoritmo LBFGS principal.
    m: memoria (número de pasos pasados a guardar) 
    """
    x = image_init.copy()
    
    # Historial para LBFGS (listas de cupy arrays)
    s_history = [] # s_k = x_{k+1} - x_k
    y_history = [] # y_k = g_{k+1} - g_k
    rho_history = []
    
    # 1. Evaluación inicial
    cost, grad = obj_function(x, V_obs, weights, reg_lambda, reg_func)
    
    print(f"Iter 0 | Cost: {cost:.6e}")
    
    for k in range(max_iter):
    

        # --- LBFGS Two-Loop Recursion ---
        q = grad.copy() # q comienza siendo el gradiente actual
        
        # El algoritmo trabaja mejor con vectores planos para los productos punto
        # pero las operaciones elemento a elemento funcionan igual en 2D.
        
        alphas = []
        
        # LOOP 1 (Hacia atrás)
        limit = len(s_history)
        for i in range(limit - 1, -1, -1):
            s = s_history[i]
            y = y_history[i]
            rho = rho_history[i]
            
            alpha_i = rho * cp.sum(s * q)
            alphas.append(alpha_i)
            
            q -= alpha_i * y
            
        # Escalado inicial de H_0 (importante para convergencia)
        if limit > 0:
            s_last = s_history[-1]
            y_last = y_history[-1]
            gamma = cp.sum(s_last * y_last) / cp.sum(y_last * y_last)
            r = gamma * q
        else:
            r = q
            
        # LOOP 2 (Hacia adelante)
        # Nota: alphas se llenó en orden inverso, así que iteramos hacia adelante
        for i in range(limit):
            s = s_history[i]
            y = y_history[i]
            rho = rho_history[i]
            alpha_i = alphas[limit - 1 - i]  # Recuperar en orden correcto (alphas está invertido)
            
            beta = rho * cp.sum(y * r)
            r += s * (alpha_i - beta)
            
        # La dirección de descenso es el negativo de la aproximación H*g
        direction = -r
        
        # --- Line Search ---
        # Usamos backtracking para encontrar el tamaño de paso
        step_size = armijo_line_search(x, direction, grad, cost, 
                                       V_obs, weights, reg_lambda, reg_func)
        

        # g_norm = cp.linalg.norm(grad)
        # print(f"Iter {k+1} | Cost: {cost:.6e} | Grad Norm: {g_norm:.6e} | Step: {step_size:.4e}")

        # --- Actualización ---
        x_new = x + step_size * direction
        
        # Calcular nuevo costo y gradiente
        cost_new, grad_new = obj_function(x_new, V_obs, weights, reg_lambda, reg_func)
        
        # --- Guardar en memoria (s y y) ---
        s_k = x_new - x
        y_k = grad_new - grad
        
        # Cálculo de rho para el siguiente paso (rho = 1 / y^T s)
        sy_dot = cp.sum(y_k * s_k)
        
        # Verificación de seguridad (curvatura)
        if sy_dot > 1e-10:
            if len(s_history) >= m:
                s_history.pop(0)
                y_history.pop(0)
                rho_history.pop(0)
            
            s_history.append(s_k)
            y_history.append(y_k)
            rho_history.append(1.0 / sy_dot)
            
        # Actualizar variables para siguiente iter
        x = x_new
        grad = grad_new
        cost = cost_new
        
        # Reporte y Convergencia
        if k % 10 == 0:
            print(f"Iter {k+1} | Cost: {cost:.6e} | Step: {step_size:.4e}")
            
        if cp.linalg.norm(grad) < tol:
            print("Convergencia alcanzada.")
            break
            
    return x

