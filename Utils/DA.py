""" Utilities for working with differential algebraic variables """

import numpy as np
from pyaudi import gdual_double as gd
from pyaudi import abs
# TODO: This gradient gets the first derivative of a function wrt each variable.
# Need to write a second that uses the polynomial differentiation method to obtain the gradient of the expansion?
# Map inversion method?

def invert(map):
    # First subtract the constant part of the Map such that Map(0)=0
    # Let Map = M + N  where M is the linear part and N is the remaining nonlinear part
    # Then the Map's inverse derivatives can be found via fixed point iteration: Map^-1 = M^-1 o (IMap - N o Map^-1)
    return

def odeint(f, x0, args, integrationOrder=None):
    ''' If integrationOrder is None then the order of the expansion of the initial conditions is used. '''


    return

def __step(fun, zi, ti,tnew, args, integrationOrder):
    ''' Used in odeint '''
    for _ in range(integrationOrder+1):
        f = fun(ti, zi, *args) # Get the taylor polynomials of the ode, including wrt to ti (or maybe tnew?)
        # zi = zi + antiderivation(f) # no idea what to do. Map inverse? integrate componentwise?

    return zi

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
        new_pt = [da.evaluate(eval_pt) if isinstance(da,gd) else da for da in da_array]
        new_pts.append(new_pt)

    return np.array(new_pts)

def differentiate(da, da_vars):
    g = np.zeros(len(da_vars), dtype=gd)
    for var in da.symbol_set:
        ind = da_vars.index(var)
        g[ind] = da.partial(var)
    return g


def gradient(da, da_vars):
    """ da_vars is 1-d list/array with string names in the order the gradient should be given """
    g = np.zeros(len(da_vars))
    if isinstance(da,gd): # It may be the case that the DA was differentiated and became a constant
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
    g = differentiate(da, da_vars)

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
    try:
        da_array[0] # Check to see if its a scalar
        if array:
            return np.array([const(da, array=array) for da in da_array])
        else:
            return [const(da, array=array) for da in da_array]
    except:
        # print "Not an array"
        if isinstance(da_array, gd):
            return da_array.constant_cf
        else:
            return da_array
    # if not array:
    #     return [da.constant_cf if isinstance(da,gd) else da for da in da_array]
    # else:
    #     return np.array([da.constant_cf if isinstance(da,gd) else da for da in da_array])

def const_dict(da_dict):
    con_dict = {}
    for key,val in da_dict.items():
        try:
            con_dict[key] = val.constant_cf
        except:
            con_dict[key] = val
    return con_dict

def make(values, names, orders, array=False):
    """ Turn an array of constant values into an array of generalized duals with specified names and orders. Orders may be an integer. """
    if isinstance(orders, int):
        orders = [orders for _ in names]
    if array:
        return np.array([gd(v,n,o) for v,n,o in zip(values, names, orders)],dtype=gd)

    else:
        return [gd(v,n,o) for v,n,o in zip(values, names, orders)]


def radians(x):
    return x*np.pi/180.0


def degrees(x):
    return x*180.0/np.pi

def sign(x):
    if isinstance(x,gd):
        xc = x.constant_cf
    else:
        xc = x
    if abs(xc)<1e-6:
        return x-xc             # So its still a DA variable but with absolutely 0 part
    else:
        return xc/abs(xc)       # This will never be a DA variable so we can use the numerical value


# ################################################################
#                        Special Functions                       #
# ################################################################

# The following changes were made to allow Bspline to have DA coefficients:

# In scipy.interpolate._bsplines.py

#   In function _get_dtype add the following check:
#       elif np.issubdtype(dtype, gd):
#           return gd

# In scipy.interpolate._fitpack_impl.py

#   In function splder change the line:
#       c = (c[1:-1-k] - c[:-2-k])* k / dt => c = (c[1:-1-k] - c[:-2-k])*(k/dt)    # For some reason, the dual variables couldn't be divided by a float, but multiplying worked fine


from Utils.memoize import memoize

@memoize
def __getSplIdx(x,t):
    idx = []
    for xi in x:
        for i,ti in enumerate(t):
            if ti > xi:
                idx.append(i-1) # This gives the index of the lower bound such that t[idx] < xi < t[idx+1]
                break
    return idx


def splev(x, spline):
    """ Implements De Boor's algorithm to evaluate a B-spline at points in x """
    if isinstance(spline,tuple):
        t,c,n = spline
    else:
        t,c,n = spline.tck

    L = __getSplIdx(x,t)
    Ln = [Li-n for Li in L]

    B = []
    for j,xj  in enumerate(x):
        d = np.array(c[Ln[j]:L[j]+1])
        dnext = np.zeros_like(d)

        for k in range(1,n+1):
            for i in range(Ln[j]+k,L[j]+1):
                idx = i-Ln[j]
                alpha_kn = (xj - t[i])/(t[i+n+1-k] - t[i])
                dnext[idx] = (1-alpha_kn)*d[idx-1] + alpha_kn*d[idx]
            d = dnext
        B.append(d[-1])

    return np.array(B)
