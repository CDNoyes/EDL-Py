""" Pure python implementation of a projected newton method for solving linearly constrained quadratic programs """

import numpy as np


def ProjectedNewton(x0, hessian, gradient, bounds, tol=1e-6, debug=False):
    """
    Solves the quadratic problem:
    min F(x) = 0.5*x'*hessian*x + gradient'*x
    subject to bounds[0] <= x <= bounds[1]]

    Inputs:
        x0          -   an initial guess, need not be feasible, length n
        hessian     -   curvature matrix of cost function, nxn matrix
        gradient    -   derivative of cost function, length n
        bounds      -   tuple of (nlower bounds, nupper bounds)
        tol         -   convergence tolerance

    Outputs:
        x           -   the converged solution, or the last iteration if max iterations is reached
        hff         -   the decomposition of the free directions of the Hessian
        fopt        -   the value of the quadratic objective function evaluated at x

    """
    # if debug:
        # print "Inputs:  "
        # print "Hessian = {}".format(hessian)
        # print "Gradient = {}".format(gradient)
        # print "Initial guess = {}".format(x0)
        # print "Lower bound: {}".format(bounds[0])
        # print "Upper bound: {}".format(bounds[1])

    iter = 0
    iterMax = 100
    n = len(x0)

    x = np.clip(x0, bounds[0], bounds[1]).squeeze()  # Make the initial point feasible
    xl = np.asarray(bounds[0])
    xu = np.asarray(bounds[1])
    hessian = np.asarray(hessian)
    gradient = np.asarray(gradient)

    if debug:
        print("After some transforms:  ")
        print("Hessian = {}".format(hessian))
        print("Gradient = {}".format(gradient))
        print("Initial feasible guess = {}\n".format(x))

    while iter < iterMax:
        g = gradient + np.dot(hessian, x)

        # Determine the active set (and the complementary inactive set)
        idu = (x-xu) == 0
        idl = (xl-x) == 0
        gu  = g>0
        gl  = g<0

        if debug:
            print("g = {}".format(g))
            print("g.shape = {}".format(g.shape))

        c = ((gl.astype(int)+idu.astype(int)) > 1) + ((gu.astype(int)+idl.astype(int)) > 1)
        f = np.logical_not(c.astype(bool))
        hff =  hessian[f,:][:,f]
        if debug:
            print("Free dirs: {}".format(f))
            print("Free hess: {}".format(hff))
            print("Free g: {}".format(gradient[f]))
            print("Free x: {}".format(x[f]))

        if len(f):
            gf = gradient[f] + np.dot(hff, x[f]).T
        else:
            gf = np.zeros_like(gradient)

        if np.any(c) and not np.all(c):
            hfc = hessian[f,:][:,c]
            gf += np.dot(hfc, x[c])

        if np.linalg.norm(gf) < tol:
            break

        dx = np.zeros((n,))
        dx[f] = -np.dot(np.linalg.inv(hff), gf)

        alpha = _armijo(_fQuad(hessian, gradient), x, dx, g, xl, xu)
        x = np.clip(x+alpha*dx, xl, xu).squeeze()
        iter += 1

    clamped = np.where(c[0])[0]
    H = hessian
    H[clamped,:][:,clamped] = 0
    return x, H


def _armijo(f, x, dx, g, xl, xu):
    gamma = 0.1
    c = 0.5
    alpha = 2*np.max(xu-xl)
    r = 0.5*gamma
    while r < gamma:
        alpha *= c
        xa = np.clip(x+alpha*dx, xl, xu)
        r = (f(x)-f(xa))/np.dot(g, (x-xa))
    return alpha


def _fQuad(h, g):
    def fQuadFun(x):
        return 0.5*np.dot(x, np.dot(h, x)) + np.dot(g, x)
    return fQuadFun


def test():
    from Regularize import Regularize
    from scipy.optimize import minimize

    n = 30       # Problem size
    N = 10      # Number of problems to solve
    for _ in range(N):
        H = (-1 + 2*np.random.random((n, n)))*3
        H = H + H.T
        H = Regularize(H, 0.1)

        H = np.eye(n)
        g = (-1 + 2*np.random.random((n,)))*5
        x = (-1 + 2*np.random.random((n,)))*3.2


        bounds = [-3*np.ones((n,)),5*np.ones((n,))]
        xo,_ = ProjectedNewton(x,H,g,[-3*np.ones((n,)),5*np.ones((n,))], debug=False)
        # Notice: This is not a QP-specific solver so we do not compare run times.
        # It is used solely to check the correctness of the solution returned.
        sol = minimize(_fQuad(H,g), x, args=(), method='SLSQP', jac=None, hess=None, hessp=None, bounds=list(zip(*bounds)), constraints=(), tol=None, callback=None, options=None)
        print("Solution difference = {}".format(np.linalg.norm(sol.x-xo)))


if __name__ == "__main__":
    test()
