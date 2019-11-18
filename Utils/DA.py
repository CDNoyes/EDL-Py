""" Utilities for working with differential algebraic variables """

import numpy as np
from pyaudi import gdual_double as gd
from pyaudi import gdual_vdouble as gdv
from pyaudi import abs

try:
    from .memoize import memoize
except:
    from memoize import memoize 

dual = (gd, gdv)


def interp(xnew, x, y):
    """ Linearly interpolates DA variables
    
        Assumes x is sorted and increasing 
        xnew need not be sorted 
    
    """
    x = np.asarray(x)
    if x[-1] <= x[0]:
        print("Warning: da.interp, x array is not increasing")
        return None 
    shape = [len(xnew)]
    shape.extend(np.shape(y[0])) # This allows y to be a nd 
    ynew = np.zeros(shape, dtype=type(gd))

    for i,xn in enumerate(xnew):

        if xn <= x[0]:
            ynew[i] = y[0]
            continue
        elif xn >= x[-1]:
            ynew[i] = y[-1]
            continue

        idx = (np.abs(xn - x)).argmin()
        if x[idx] >= xn:
            upper = idx 
            lower = upper - 1
        else:
            lower = idx 
            upper = lower + 1
        ynew[i] = y[lower] + (xn-x[lower])*(y[upper]-y[lower])/(x[upper] - x[lower])
    return ynew 

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
        except (TypeError, IndexError):  # pt is a scalar
            pt = [pt]
        # eval_pt = {da_var:element for da_var,element in zip(delta,pt)} # This is probably slow since we're constructing a new dict every time
        eval_pt.update(zip(delta, pt))
        new_pt = [da.evaluate(eval_pt) if isinstance(da, dual) else da for da in da_array]
        new_pts.append(new_pt)

    return np.array(new_pts).squeeze()


def differentiate(da, da_vars):
    """Differentiates a generalized dual variable wrt da_vars"""
    if isinstance(da, dual):
        g = np.zeros(len(da_vars), dtype=type(da))
        for ind, var in enumerate(da_vars):
            g[ind] = da.partial(var)
    else:
        g = np.zeros_like(da_vars)
    return g


def gradient(da, da_vars):
    """Returns the numerical value of the gradient of a generalized dual variable
        Inputs:
            da is a generalized dual variable
            da_vars is 1-d list/array with string names in the order the gradient should be given
    """
    g = [0]*len(da_vars)
    if isinstance(da, dual):  # It may be the case that the DA was differentiated and became a constant
        g = [da.partial(var).constant_cf for var in da_vars]
    
    try:
        return np.array(g).squeeze()
    except ValueError:
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


def coeff(da, da_vars, index):
    # from itertools import permutations, product 

    n = len(da_vars)

    assert len(da.symbol_set) == n, "Incorrect number of variables specified"
    if not index: # 0
        return da.constant_cf

    shape = [n]*index
    C = np.zeros(shape)
    print(C.shape)
    x = np.array([da])
    for iteration in range(index):
        x = np.array([differentiate(xda, da_vars) for xda in x.flat])

    x = const(x, array=True).reshape(shape)
    return x

    # Using the find_cf interface - hard 
    # P = list(permutations(range(index+1), n))
    # U = list(product([0, index], repeat=n))
    # print(U)   

    # K = [p for p in P+U if sum(p)==index]
    # print(K)

    # for idx in K:
    #     matrix_idx = # Need to map multi-index to array index and that isnt trivial...
    #     C[matrix_idx] = da.find_cf(idx)

    # # TODO: If da.symbol_set is ordered differently than da_vars, the matrix indices needs to be permuted
    # return C


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
        from Utils.smooth_sat import cosh_sat as saturate
        m = (lower+upper)*0.5
        r = (upper-lower)*0.5
        return m + saturate(x-m, r, tuning=200)
    else:
        try:
            x[0]  # Iterable?
            return np.clip(x, lower, upper)
        except (TypeError, IndexError):
            return np.clip([x], lower, upper)[0]


def relax(f, x):
    """ Returns second-order over- and under-approximations of f """
    try:
        from Regularize import Split, AbsRegularize
    except:
        from . import Regularize
        Split = Regularize.Split
        AbsRegularize = Regularize.AbsRegularize

    scalar = np.ndim(np.squeeze(x)) == 0
    vectorized = np.ndim(np.squeeze(x)) == 2
    print("Scalar: {}".format(scalar))
    print("Vectorized: {}".format(vectorized))

    if scalar:
        x = [x]

    names = ["x{}".format(i) for i, _ in enumerate(x)]
    x = make(x, names, 2, True, vectorized=vectorized)
    z = make(np.zeros_like(x), names, 2, True, vectorized=vectorized)

    if scalar:
        x = x[0]

    F = f(x)
    G = gradient(F, names)
    H = hessian(F, names)

    if scalar:
        # P = max(0., H[0])
        # N = min(0., H[0])
        P = abs(H[0])
        N = -P 
    else:
        # P, N = Split(H)
        P = AbsRegularize(H)
        N = -P 

    Fcvx = F.constant_cf + G.dot(z) + 0.5*z.dot(P).dot(z)
    Fccv = F.constant_cf + G.dot(z) + 0.5*z.dot(N).dot(z)
    return Fcvx, Fccv, names



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


################ Tests #######################################

def test_differentiation_scalar():

    n = 50                                        # Input size
    names = ["x{}".format(i) for i in range(n)]
    A = np.random.random((n,))
    B = np.random.random((n, n))
    B = 0.5*(B+B.T)

    x = make(np.zeros((n,)), names, 2, True, False)
    print(x.nbytes)
    print(B.nbytes)
    F = A.dot(x) + 0.5*x.T.dot(B).dot(x)

    g = gradient(F, names)
    h = hessian(F, names)

    assert np.allclose(A, g), "Scalar gradient computation failed."
    assert np.allclose(B, h), "Scalar Hessian computation failed."
    print("Scalar differentiation test passed.\n")


def test_differentiation_vector():
    
    
    n = 2          # Input size
    m = 3         #  Vectorization size

    names = ["x{}".format(i) for i in range(n)]

    A = np.random.random((m, n))
    B = np.random.random((m, n, n))
    B = 0.5*(B+np.transpose(B, (0, 2, 1)))

    x0 = np.random.random((n, m))
    x = make(x0, names, 2, True, True)

    print(x.shape)
    F = A.dot(x) #+ 0.5*x.dot(B).dot(x)

    print(const(0.5*x.dot(B).dot(x), True))
    quad = np.array([0.5*x.dot(Bi).dot(x) for Bi in B])
    print(quad[0])
    print(const(quad, True))

    print(np.array([0.5*xi.dot(Bi).dot(xi) for xi, Bi in zip(x0.T, B)]))
    # print(A[0])
    # print(gradient(F[0], names))

    # g = jacobian(F, names)
    # print(g)
    # h = vhessian(F, names)
    # print(g.shape)
    # print(h.shape)
    # print(B[0])
    # print(F[0])
    # print(hessian(F[0], names))
    # print(h)
    # assert np.allclose(A, g), "Vectorized gradient computation failed."
    # assert np.allclose(B, h), "Vectorized Hessian computation failed."

    # print("Vectorized differentiation test passed.\n")


def test_scalar_relax():
    import pyaudi as pa
    import matplotlib.pyplot as plt 

    def fun(x):
        return pa.sin(5*x) + pa.exp(pa.abs(x))
        # return -x**2 + x**3 + 1

    x0 = 0.5
    fcvx, fccv, da_vars = relax(fun, 0.5)

    print(fcvx)
    print(fccv)

    dx = np.linspace(-0.25, 0.25, 50)/10

    F = np.array([fun(xi) for xi in x0+dx])
    
    U = evaluate(fcvx, da_vars, dx)
    L = evaluate(fccv, da_vars, dx)

    x = x0 + dx

    plt.plot(x, F, label="True")
    plt.plot(x, U, label="Cvx")
    plt.plot(x, L, label="Ccv")
    plt.legend()
    plt.show()


def test_coeff():
    import pyaudi as pa 

    x = make(range(3), 'xyz', 5)

    def fun(x, y, z):
        return pa.sin(z*y) + 1*y*z + z**2*y**2 + x + x**2
    
    f = fun(*x)
    assert np.allclose(coeff(f, 'xyz', 2), hessian(f, 'xyz')), "2nd order coeff do not match Hessian"
    print(f)
    print(coeff(f, 'xyz', 0))
    print(coeff(f, 'xyz', 1))
    print(coeff(f, 'xyz', 2))
    print(coeff(f, 'xyz', 3))
    return


def test():
    import time 
    t = []
    t.append(time.time())
    test_differentiation_scalar()
    # t.append(time.time())
    # test_differentiation_vector()
    # t.append(time.time())
    # test_scalar_relax()
    # t.append(time.time())
    test_coeff()
    t.append(time.time())


if __name__ == "__main__":
    # test()
    test_coeff()

