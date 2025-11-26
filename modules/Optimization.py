import cupy as cp

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

from modules.interferometry import adjoint_op, forward_op

def obj_function(image, V, W, reg_lambda, ref_func):
    pass
