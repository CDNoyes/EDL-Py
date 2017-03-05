""" Utilities for working with differential algebraic variables """

import numpy as np
from pyaudi import gdual_double as gd

# TODO: This gradient gets the first derivative of a function wrt each variable.
# Need to write a second that uses the polynomial differentiation method to obtain the gradient of the expansion?
# Map inversion method?
    
    
def evaluate(da_array, da_vars, pts):
    """ 
    Evaluates each da in da_array at each pt in pts. 
    The variables in da_vars must match the order in pts.
    """
    
    delta = ['d'+da_var for da_var in da_vars]
    eval_pt = {da_var:0 for da_var in delta}    # Preallocate the evaluation dictionary
    new_pts = []
    for pt in pts:
        # eval_pt = {da_var:element for da_var,element in zip(delta,pt)} # This is probably slow since we're constructing a new dict every time
        eval_pt.update(zip(delta,pt))
        new_pt = [da.evaluate(eval_pt) for da in da_array]
        new_pts.append(new_pt)    
    
    return np.array(new_pts)
def gradient(da, da_vars):
    """ da_vars is 1-d list/array with string names in the order the gradient should be given """
     
    g = np.zeros(len(da_vars))
    for var in da.symbol_set:
        ind = da_vars.index(var)
        g[ind] = da.partial(var).constant_cf
    return g
    
    
def jacobian(da_array, da_vars):
    """ Forms the Jacobian matrix, i.e. the gradient of a vector valued function """
    return np.array([gradient(da, da_vars) for da in da_array])
    
def hessian(da, da_vars):
    """ Retrieves the 2nd order coefficients forming the 2D Hessian matrix of a scalar function """
    n = len(da_vars)
    
    # Get the gradient but without taking the constant term
    g = np.zeros(len(da_vars), dtype=type(da))
    for var in da.symbol_set:
        ind = da_vars.index(var)
        g[ind] = da.partial(var)
    
    # Now the hessian is simply the gradient of each element of the gradient
    # h = np.zeros((n,n))
    # for ind2,gda in enumerate(g):
        # for var in da.symbol_set:
            # ind1 = da_vars.index(var)
            # h[ind1,ind2] = gda.partial(var).constant_cf

    h = [gradient(gda,da_vars) for gda in g]    # This is simple but we can cut the work nearly in half by exploiting symmetry
        
    return np.array(h)        

def vhessian(da_array,da_vars):
    """ Computes the tensor comprising the hessian of a vector valued function """
    return np.array([hessian(da,da_vars) for da in da_array])

def const(da_array, array=False):
    """ Collects the constant part of each generalized dual variable and returns a list or numpy array. """
    # TODO: What if not all the variables in the array are gd?
    if isinstance(da_array, list) and not array:
        return [da.constant_cf for da in da_array]
    else:
        return np.array([da.constant_cf for da in da_array])
        
def make(values, names, orders, array=False):
    """ Turn an array of constant values into an array of generalized duals with specified names and orders. Orders may be an integer. """
    if isinstance(orders, int):
        orders = [orders for _ in names]    
    if array:
        return np.array([gd(v,n,o) for v,n,o in zip(values, names, orders)])

    else:    
        return [gd(v,n,o) for v,n,o in zip(values, names, orders)]