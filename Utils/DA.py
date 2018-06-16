""" Utilities for working with differential algebraic variables """

import numpy as np
from pyaudi import gdual_double as gd
from pyaudi import gdual_vdouble as gdv
from pyaudi import abs

from .memoize import memoize

dual = (gd, gdv)


def odeint(fun, x0, iv, args=(), min_order=4, max_order=40, tol=1e-3):

    x_collect = [x0]
    vars = ["x{}".format(i) for i in range(len(x0))] + ['t']

    eval_pt = np.random.random((len(x0))).tolist()
    zero_pt = np.zeros_like(eval_pt).tolist()
    div = np.diff(iv)

    for t, dt in zip(iv, div):
        x = np.array([gd(0, var, 1) for var in vars[:-1]])  # Initialize
        for order in range(2, max_order):
            x_next = _step(fun, t, x0, x, dt, order, *args)
            change = np.linalg.norm(evaluate(x-x_next, vars, [eval_pt+[dt]])[0][0])  # norm of just the new terms
            # Update for next iteration
            x = x_next
            # Convergence criteria
            if change <= tol and order >= min_order:
                # print("Terminating at order = {}".format(order))
                break
        x0 = evaluate(x, vars, [zero_pt+[dt]])[0]  # evaluate the endpoint model
        x_collect.append(x0.squeeze())
    return np.array(x_collect).squeeze()


def _step(fun, t0, x0, x, dt, order, *args):

    x0 = np.array([gd(x0i, 'x{}'.format(i), order) for i, x0i in enumerate(x0)])
    t = gd(t0, 't', order)
    x += 0*t  # This is done purely to promote the order of x

    f = fun(x, t, *args)            # Expand around the current time and state
    F = np.array([da.integrate('t') if isinstance(da, gd) else da*t for da in f], dtype=gd)  # Anti-derivation
    return x0 + F


def evaluate(da_array, da_vars, pts):
    """
    Evaluates each da in da_array at each pt in pts.
    The variables in da_vars must match the order in pts.
    """

    delta = ['d'+da_var for da_var in da_vars]
    eval_pt = {da_var: 0 for da_var in delta}    # Preallocate the evaluation dictionary
    new_pts = []
    for pt in pts:
        try:
            pt[0]
        except IndexError:  # pt is a scalar
            pt = [pt]
        # eval_pt = {da_var:element for da_var,element in zip(delta,pt)} # This is probably slow since we're constructing a new dict every time
        eval_pt.update(zip(delta, pt))
        new_pt = [da.evaluate(eval_pt) if isinstance(da, (gd,gdv)) else da for da in da_array]
        new_pts.append(new_pt)

    return np.array(new_pts)


def differentiate(da, da_vars):
    """Differentiates a generalized dual variable wrt da_vars"""
    g = np.zeros(len(da_vars), dtype=gd)
    if isinstance(da, gd):
        for var in da.symbol_set:
            ind = da_vars.index(var)
            g[ind] = da.partial(var)
    return g


def gradient(da, da_vars):
    """Returns the numerical value of the gradient of a generalized dual variable
        Inputs:
            da is a generalized dual variable
            da_vars is 1-d list/array with string names in the order the gradient should be given
    """
    g = np.zeros(len(da_vars))
    if isinstance(da, gd): # It may be the case that the DA was differentiated and became a constant
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

    # This is simple but we can cut the work nearly in half by exploiting symmetry
    h = [gradient(gda, da_vars) for gda in g]    

    return np.array(h)


def vhessian(da_array, da_vars):
    """ Computes the tensor comprising the hessian of a vector valued function """
    return np.array([hessian(da, da_vars) for da in da_array])


def const(da_array, array=False):
    """ Collects the constant part of each generalized dual variable and returns a list or numpy array. """
    try:
        da_array[0]  # Check to see if its a scalar
        if array:
            return np.array([const(da, array=array) for da in da_array])
        else:
            return [const(da, array=array) for da in da_array]
    except (IndexError, TypeError):
        if isinstance(da_array, dual):
            return da_array.constant_cf
        else:
            return da_array


def const_dict(da_dict):
    con_dict = {}
    for key, val in da_dict.items():
        try:
            con_dict[key] = val.constant_cf
        except AttributeError:
            con_dict[key] = val
    return con_dict


def make(values, names, orders, array=False, vectorized=False):
    """ Turn an array of constant values into an array of generalized duals
        with specified names and orders. Orders may be an integer.
    """
    if vectorized:
        cls = gdv
    else:
        cls = gd
    if isinstance(orders, int):
        orders = [orders for _ in names]
    if array:
        return np.array([cls(v, n, o) for v, n, o in zip(values, names, orders)], dtype=cls)

    else:
        return [cls(v, n, o) for v, n, o in zip(values, names, orders)]


def compute_jacobian(function, expansion_point, args=()):
    x = make(expansion_point, ["x{}".format(i) for i in range(len(expansion_point))], 1, True)
    y = function(x, *args)
    return jacobian(y, ["x{}".format(i) for i in range(len(expansion_point))])


def compute_gradient(function, expansion_point, args=()):
    x = make(expansion_point, ["x{}".format(i) for i in range(len(expansion_point))], 1, True)
    y = function(x, *args)
    return gradient(y, ["x{}".format(i) for i in range(len(expansion_point))])


def radians(x):
    return x*np.pi/180.0


def degrees(x):
    return x*180.0/np.pi


def sign(x):
    if isinstance(x, dual):
        xc = x.constant_cf
    else:
        xc = x
    if abs(xc) < 1e-6:
        return x-xc  # So its still a DA variable but with absolutely 0 part
    else:
        return xc/abs(xc)  # This will never be a DA variable so we can use the numerical value


def clip(x, lower, upper):
    """ Clips a DA variable between its upper and lower bounds.
        Works for scalar gdual, vectorized gdual, but not currently iterables.
        Also works for non-duals and in this case simply wraps numpy.clip.
    """

    if isinstance(x, dual):
        from Utils.smooth_sat import symmetric_sat
        m = (lower+upper)*0.5
        r = (upper-lower)*0.5
        return m + symmetric_sat(x-m, r)
    else:
        try:
            x[0]  # Iterable?
            return np.clip(x, lower, upper)
        except (TypeError, IndexError):
            return np.clip([x], lower, upper)[0]

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



@memoize
def __getSplIdx(x,t):
    idx = []
    for xi in x:
        for i,ti in enumerate(t):
            if ti > xi:
                idx.append(i-1)  # This gives the index of the lower bound such that t[idx] < xi < t[idx+1]
                break
    return idx


def splev(x, spline):
    """ Implements De Boor's algorithm to evaluate a B-spline at points in x """
    if isinstance(spline, tuple):
        t, c, n = spline
    else:
        t, c, n = spline.tck

    L = __getSplIdx(x, t)
    Ln = [Li-n for Li in L]

    B = []
    for j, xj in enumerate(x):
        d = np.array(c[Ln[j]:L[j]+1])
        dnext = np.zeros_like(d)

        for k in range(1, n+1):
            for i in range(Ln[j]+k,L[j]+1):
                idx = i-Ln[j]
                alpha_kn = (xj - t[i])/(t[i+n+1-k] - t[i])
                dnext[idx] = (1-alpha_kn)*d[idx-1] + alpha_kn*d[idx]
            d = dnext
        B.append(d[-1])

    return np.array(B)
