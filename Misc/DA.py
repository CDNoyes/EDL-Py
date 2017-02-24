""" Utilities for working differential algebraic variables """

import numpy as np
from pyaudi import gdual_double as gd
    
def gradient(da, da_vars):
    """ da_vars is 1-d list/array with string names in the order the gradient should be given """
    da_vars = ['d'+x for x in da_vars]
    g = np.zeros(len(da_vars))
    z = {(key):0 for key in da_vars}
    for ind,var in enumerate(da_vars):
        z[var] = 1
        g[ind] = da.get_derivative(z)
        z[var] = 0
    return g
    
    
def jacobian(da_array, da_vars):
    return np.array([gradient(da,da_vars) for da in da_array])
    
def hessian(da, da_vars):
    n = len(da_vars)
    g = np.zeros((n,n))
    
    if da.order < 2: # Return zeros if insufficient order 
        return g
        
    da_vars = ['d'+x for x in da_vars]
    
    
    z = {(key):0 for key in da_vars}
    
    # This will get the diagonal elements:
    for ind,var in enumerate(da_vars):
        z[var] = 2
        g[ind,ind] = da.get_derivative(z)
        z[var] = 0
        
    # Off diagonal elements:    
        
        
    return g        


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