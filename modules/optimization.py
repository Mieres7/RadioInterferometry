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
    
def entropy(image, epsilon):
    log_term = cp.log(image + epsilon)
    
    # Cost function
    cost = cp.sum(image * log_term - image)
    # Gradient
    grad = log_term
    
    return cost, grad

'''
Objective function
'''

def obj_function(image, V_obs, weights):

    backend = get_backend('cupy')
    if backend == 'cupy':
        sys = cp
    else:
        sys = np

    V_pred = forward_op(image)
    residual = V_pred - V_obs

    # cost
    f_cost = 0.5 * sys.sum(weights * sys.abs(residual)**2)

    # gradient
    grad = adjoint_op(weights * residual)

    return f_cost, grad
