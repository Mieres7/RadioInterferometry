import cupy as cp
import numpy as np
from modules.interferometry import adjoint_op, forward_op
from modules.backend import get_backend

import time
import numpy as np

try:
    import cupy as cp
    xp = cp
    USING_CUPY = True
except:
    xp = np
    USING_CUPY = False

Array = xp.ndarray

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

    return cost.real, grad.real
    

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
    
    return cost.real, grad.real
    
def entropy(image):

    epsilon = 1

    log_term = cp.log(image + epsilon)
    
    # Cost function
    cost = cp.sum(image * log_term - image)
    # Gradient
    grad = log_term
    
    return cost.real, grad.real

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
    
# -------------------------------------
# LBFGS Memory
# -------------------------------------
class LBFGSState:
    def __init__(self, m: int):
        self.m = m
        self.s_list = []
        self.y_list = []
        self.rho_list = []

    def add_pair(self, s, y, min_sy=1e-12):
        sy = _dot(s, y)
        if sy <= min_sy:
            return False
        
        rho = 1.0 / sy

        if len(self.s_list) == self.m:
            self.s_list.pop(0)
            self.y_list.pop(0)
            self.rho_list.pop(0)

        self.s_list.append(s)
        self.y_list.append(y)
        self.rho_list.append(rho)
        return True


# -------------------------------------
# Helpers
# -------------------------------------
def _as_xp(arr):
    if USING_CUPY and isinstance(arr, np.ndarray):
        return cp.asarray(arr)
    return arr
def _norm(x):
    return float(xp.linalg.norm(x.ravel()))

def _dot(a, b):
    return float(xp.vdot(a.ravel(), b.ravel()))


'''
Line Search
'''
def armijo_line_search(image, direction, grad, current_cost, 
                       args, alpha0=1.0, rho=0.5, c1=1e-4,
                       alpha_min=1e-12, max_iter=50):
    
    g_dot_d = _dot(grad, direction)
    if g_dot_d >= 0:
        return 0.0, current_cost, "non_descent"

    alpha = alpha0

    for _ in range(max_iter):
        new_img = image + alpha * direction
        cost_new, _ = obj_function(new_img, *args)

        if cost_new <= current_cost + c1 * alpha * g_dot_d:
            return float(alpha), float(cost_new), "ok"

        alpha *= rho
        if alpha < alpha_min:
            break

    return float(alpha), float(cost_new), "min_alpha"

def lbfgs_two_loop(grad, state, eps=1e-16):
    if len(state.s_list) == 0:
        return -grad

    q = grad.copy()
    alpha = []

    # backward loop
    for s, y, rho in zip(reversed(state.s_list), reversed(state.y_list), reversed(state.rho_list)):
        a = rho * _dot(s, q)
        alpha.append(a)
        q = q - a * y

    # scaling
    s_last = state.s_list[-1]
    y_last = state.y_list[-1]
    gamma = _dot(s_last, y_last) / (_dot(y_last, y_last) + eps)
    r = gamma * q

    # forward loop
    for s, y, rho, a in zip(state.s_list, state.y_list, state.rho_list, reversed(alpha)):
        beta = rho * _dot(y, r)
        r = r + s * (a - beta)

    return -r
    

'''
LBFGS algorithm
'''
def lbfgs_optimize(
    x0,
    args,
    m=10,
    max_iter=100,
    gtol=1e-6,
    ftol=1e-12,
    verbose=True
):
    global xp, USING_CUPY

    x = _as_xp(x0)

    state = LBFGSState(m=m)

    cost, grad = obj_function(x, *args)
    cost = float(cost)
    grad = _as_xp(grad)

    cost_history = [cost]
    t_start = time.time()

    if verbose:
        print(f"[LBFGS] it=0 cost={cost:.6e} ||g||={_norm(grad):.3e}")

    # -----------------------
    # MAIN LOOP
    # -----------------------
    for k in range(1, max_iter+1):

        gnorm = _norm(grad)
        if gnorm <= gtol:
            if verbose:
                print(f"[LBFGS] Converged (grad) at iter {k}")
            break

        direction = lbfgs_two_loop(grad, state)

        # direction must be descent
        if _dot(grad, direction) >= 0:
            direction = -grad

        # line search
        alpha, cost_new, status = armijo_line_search(
            x, direction, grad, cost, args
        )

        # fallback if needed
        if status in ["non_descent", "min_alpha"]:
            direction = -grad
            alpha = 1e-3
            x_next = x + alpha * direction
            cost_new, grad_new = obj_function(x_next, *args)
        else:
            x_next = x + alpha * direction
            cost_new, grad_new = obj_function(x_next, *args)

        cost_new = float(cost_new)
        grad_new = _as_xp(grad_new)

        # update memory
        s = x_next - x
        y = grad_new - grad
        state.add_pair(s, y)

        # update step
        x = x_next
        cost_prev = cost
        cost = cost_new
        grad = grad_new
        cost_history.append(cost)

        if verbose:
            print(f"[LBFGS] it={k} cost={cost:.6e} ||g||={_norm(grad):.3e} alpha={alpha:.2e}")

        # stopping by function decrease
        denom = abs(cost) + abs(cost_prev) + 1e-16
        rel_change = 2 * abs(cost - cost_prev) / denom
        if rel_change <= ftol:
            if verbose:
                print(f"[LBFGS] Converged (Δf small) at iter {k}")
            break

    total_time = time.time() - t_start

    info = {
        "cost_history": cost_history,
        "niter": len(cost_history)-1,
        "time": total_time
    }

    return x, info


