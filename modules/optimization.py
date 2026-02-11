import cupy as cp
import numpy as np
import time
from modules.interferometry import adjoint_op, forward_op
from modules.backend import get_backend, xp
import matplotlib.pyplot as plt

#Array = np.ndarray

'''
Regularizations
'''
def l1(image):
    '''
    L1 Regularization
    '''
    # Cost Function
    cost = xp.sum(xp.abs(image)) 
    # Gradient
    grad = xp.sign(image) 

    return cost.real, grad.real
    

def tsv(image):
    '''
    Total Squared Variation Regularization
    '''
    # Cost Functions
    diff_x = xp.roll(image, -1, axis=1) - image 
    diff_y = xp.roll(image, -1, axis=0) - image 
    
    cost = xp.sum(diff_x**2 + diff_y**2)
    
    # Gradient
    im1_x = xp.roll(image, 1, axis=1)
    im1_y = xp.roll(image, 1, axis=0)
    
    ip1_x = xp.roll(image, -1, axis=1)
    ip1_y = xp.roll(image, -1, axis=0)
    
    lap_x = (ip1_x - image) - (image - im1_x)
    lap_y = (ip1_y - image) - (image - im1_y)
    
    grad = -2.0 * (lap_x + lap_y)
    
    return cost.real, grad.real
    
def entropy(image):

    epsilon = 1e-12

    img = xp.clip(image, epsilon, None)

    log_term = xp.log(img)
    
    # Cost function
    cost = xp.sum(img * log_term - img) 
    # Gradient
    grad = log_term
    
    return cost.real, grad.real

'''
Objective function
'''
def obj_function(image, V_obs, weights, reg_lambda, reg_func):

    V_pred = forward_op(Image=image, gridded=True)
    residual = V_pred - V_obs

    # cost
    f_cost = 0.5 * xp.sum(weights * xp.abs(residual)**2)
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
    if xp is cp and isinstance(arr, np.ndarray):
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
    verbose=True,
    # ------------------------
    # Parámetros de Armijo
    # ------------------------
    armijo_alpha0=1.0,
    armijo_rho=0.5,
    armijo_c1=1e-4,
    armijo_alpha_min=1e-12,
    armijo_max_iter=50
):

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

        # ensure descent direction
        if _dot(grad, direction) >= 0:
            direction = -grad

        # -----------------------
        # LINE SEARCH (Armijo)
        # -----------------------
        alpha, cost_new, status = armijo_line_search(
            image=x,
            direction=direction,
            grad=grad,
            current_cost=cost,
            args=args,
            alpha0=armijo_alpha0,
            rho=armijo_rho,
            c1=armijo_c1,
            alpha_min=armijo_alpha_min,
            max_iter=armijo_max_iter
        )

        # fallback
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
            print(
                f"[LBFGS] it={k} cost={cost:.6e} "
                f"||g||={_norm(grad):.3e} alpha={alpha:.2e}"
            )

        # stopping by Δf
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


# --------- Non-Linear Conjugate Gradient ---------

def ncg_optimize(
    x0,
    args,
    max_iter=100,
    gtol=1e-6,
    ftol=1e-12,
    verbose=True,
    # Armijo params
    armijo_alpha0=1.0,
    armijo_rho=0.5,
    armijo_c1=1e-4,
    armijo_alpha_min=1e-12,
    armijo_max_iter=50,
    restart=True
):
    """
    Nonlinear Conjugate Gradient (Fletcher–Reeves variant)
    """
    x = _as_xp(x0)

    cost, grad = obj_function(x, *args)
    cost = float(cost)
    grad = _as_xp(grad)

    # Stats
    cost_history = [cost]
    gradnorm_history = [_norm(grad)]
    alpha_history = []
    beta_history = []
    rel_change_history = []

    direction = -grad.copy()

    cost_history = [cost]
    t_start = time.time()

    if verbose:
        print(f"[NCG] it=0 cost={cost:.6e} ||g||={_norm(grad):.3e}")

    for k in range(1, max_iter + 1):

        gnorm = _norm(grad)
        if gnorm <= gtol:
            if verbose:
                print(f"[NCG] Converged (grad) at iter {k}")
            break

        # Ensure descent direction
        if _dot(grad, direction) >= 0:
            direction = -grad.copy()

        # -----------------------
        # Line search
        # -----------------------
        alpha, cost_new, status = armijo_line_search(
            image=x,
            direction=direction,
            grad=grad,
            current_cost=cost,
            args=args,
            alpha0=armijo_alpha0,
            rho=armijo_rho,
            c1=armijo_c1,
            alpha_min=armijo_alpha_min,
            max_iter=armijo_max_iter
        )

        # fallback
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

        # -----------------------
        # Fletcher–Reeves beta
        # -----------------------
        numerator = _dot(grad_new, grad_new)
        denominator = _dot(grad, grad) + 1e-16
        beta = numerator / denominator

        # Optional restart
        if restart and (k % x.size == 0):
            beta = 0.0

        beta_history.append(float(beta))
        alpha_history.append(float(alpha))
        gradnorm_history.append(_norm(grad_new))

        # New direction
        direction = -grad_new + beta * direction

        # Update
        x = x_next
        cost_prev = cost
        cost = cost_new
        grad = grad_new
        cost_history.append(cost)

        if verbose:
            print(
                f"[NCG] it={k} cost={cost:.6e} "
                f"||g||={_norm(grad):.3e} alpha={alpha:.2e} beta={beta:.2e}"
            )

        # stopping by Δf
        denom = abs(cost) + abs(cost_prev) + 1e-16
        rel_change = 2 * abs(cost - cost_prev) / denom

        rel_change_history.append(rel_change)

        if rel_change <= ftol:
            if verbose:
                print(f"[NCG] Converged (Δf small) at iter {k}")
            break

    total_time = time.time() - t_start

    info = {
        "cost_history": cost_history,
        "gradnorm_history": gradnorm_history,
        "alpha_history": alpha_history,
        "beta_history": beta_history,
        "rel_change_history": rel_change_history,
        "niter": len(cost_history) - 1,
        "time": total_time
    }   

    return x, info


def plot_stats(info, title):

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # --- Cost ---
    axs[0].semilogy(info["cost_history"])
    axs[0].set_title("Objective Function")
    axs[0].set_ylabel("Cost")

    # --- Gradient Norm ---
    axs[1].semilogy(info["gradnorm_history"])
    axs[1].set_title("Gradient Norm")
    axs[1].set_ylabel("||∇f||")

    # --- Step Size ---
    axs[2].plot(info["alpha_history"])
    axs[2].set_title("Step Size")
    axs[2].set_ylabel("Alpha")

    # Shared X label
    fig.supxlabel("Iteration")
    
    fig.suptitle(title, fontsize=14)

    for ax in axs:
        ax.grid(True)

    plt.tight_layout()
    plt.show()



