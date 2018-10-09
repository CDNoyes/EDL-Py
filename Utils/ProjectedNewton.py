""" Pure python implementation of a projected newton method for solving box-constrained quadratic programs """

import numpy as np
from itertools import product 

def ProjectedNewton(x0, hessian, gradient, bounds, tol=1e-6, iter_max=10, verbose=False, check_hessian=False):
    """
    Solves the quadratic problem:
    min F(x) = 0.5*x'*hessian*x + gradient'*x
    subject to bounds[0] <= x <= bounds[1]]

    Inputs:
        x0          -   an initial guess, need not be feasible, length n
        hessian     -   positive semi-definite curvature matrix of cost function, nxn matrix
        gradient    -   derivative of cost function, length n
        bounds      -   tuple of (nlower bounds, nupper bounds)
        tol         -   convergence tolerance using norm of free gradient directions 
        iter_max    -   the maximum number of iterations to be performed
        verbose     -   boolean for printing statements 
        check_hessian - confirms the hessian meets the required positive definite condition 

    Outputs:
        x           -   the converged solution, or the last iteration if max iterations is reached
        H           -   the Hessian with clamped directions set to 0 to prevent feedback in infeasible directions


    Reference: Control-limited Differential Dynamic Programming 

    """
    bounds = np.asarray(bounds)
    assert np.min(bounds[1]-bounds[0]) > 0, "The bounds provided have an empty interior."

    if check_hessian:
        assert np.all(np.linalg.eigvalsh(hessian) > 0), "Hessian is not positive definite"

    iteration = 0
    try:
        n = len(x0)
    except TypeError:
        n = 1

    if n == 1:  # trivial scalar case
        opt = np.clip(-np.squeeze(gradient)/np.squeeze(hessian), *bounds).squeeze()
        opt = np.array([opt])
        
        if opt in bounds:
            H = [[0.]]
        else:
            H = [[np.squeeze(hessian)]]
        return opt, H

    x = np.clip(x0, bounds[0], bounds[1]).squeeze()  # Make the initial point feasible
    xl = np.asarray(bounds[0])
    xu = np.asarray(bounds[1])
    hessian = np.asarray(hessian).squeeze()
    gradient = np.asarray(gradient).squeeze()

    while iteration < iter_max:
        g = gradient + np.dot(hessian, x)

        # Determine the active set (and the complementary inactive set)
        idu = (x-xu) == 0
        idl = (xl-x) == 0
        gu = g > 0
        gl = g < 0  # = np.logical_not(gu)

        c = np.logical_or(np.logical_and(gl, idu), np.logical_and(gu, idl)).squeeze()
        f = np.logical_not(c)
        hff = hessian[f, :][:, f]

        if len(f):
            gf = gradient[f] + np.dot(hff, x[f]).T
        else:
            gf = np.zeros_like(gradient)

        if np.any(c) and not np.all(c):
            hfc = hessian[f, :][:, c]
            gf += np.dot(hfc, x[c])

        if np.linalg.norm(gf, 1) < tol:
            if verbose:
                print("QP solved successfully:")
            break

        dx = np.zeros((n,))
        try:
            dx[f] = -np.linalg.solve(hff, gf)
        except np.linalg.LinAlgError:  # Singular case
            dx[f] = -np.dot(np.linalg.pinv(hff), gf)

        alpha = _armijo(_fQuad(hessian, gradient), x, dx, g, xl, xu)
        x = np.clip(x+alpha*dx, xl, xu).squeeze()
        iteration += 1

    if verbose:
        if iteration == iter_max:
            print("Maximum number of iterations reached without convergence.")
        else:
            print("Total iterations = {}".format(iteration))


    # TODO: The following code works but there must be a vectorized way to set all of these values to zero 
    clamped = np.where(c)[0]
    H = hessian
    for pt in product(clamped, clamped):
        H[pt] = 0
    # H[clamped, :][:, clamped] = 0
    return x, H


def _armijo(f, x, dx, g, xl, xu):
    gamma = 0.1
    c = 0.5
    alpha = 2*np.max(xu-xl)
    r = 0.5*gamma
    while r < gamma:
        alpha *= c
        xa = np.clip(x+alpha*dx, xl, xu).squeeze()
        r = (f(x)-f(xa))/np.dot(g, (x-xa))
    return alpha


def _fQuad(h, g):
    def fQuadFun(x):
        return 0.5*np.dot(x, np.dot(h, x)) + np.dot(g, x)
    return fQuadFun


def test_scalar():
    """ Scalar test case """ 
    from scipy.optimize import minimize_scalar

    H = 1
    g = -1
    x = 0

    opt = ProjectedNewton(x, H, g, [-1, 10])
    sol = minimize_scalar(_fQuad(H, g), method='bounded', bounds=(-1, 10))
    assert np.allclose(opt[0], sol.x), "Projected Newton 'test_scalar' failed. Solution does not match scipy minimize_scalar."


def test_checks():
    """ Checks bounds and positive semi-definiteness of Hessian """
    n = 3

    g = np.zeros((n,))
    H = np.diag([1,-1, 0])  # Indefinite matrix 

    try:
        ProjectedNewton([0,0,0], np.eye(n), g, bounds=([-1,-1,-1], [2, 0, -1]))
        assert False, "Projected Newton 'test_checks' failed. Function did not catch empty interior defined by bounds."
    except AssertionError:
        pass 

    try:
        ProjectedNewton([0,0,0], H, g, bounds=([-1,-1,-1], [2, 0, 3]), check_hessian=True)
        assert False, "Projected Newton 'test_checks' failed. Indefinite Hessian was passed and not handled."
    except AssertionError:
        pass


def test():
    import matplotlib.pyplot as plt 
    import seaborn 
    from Regularize import Regularize, AbsRegularize
    from scipy.optimize import minimize
    from timeit import default_timer as timer 

    n = 4       # Problem size
    N = 10000      # Number of problems to solve
    print("Solving {} problems of size {}\n".format(N, n))
    tsolve = []
    for _ in range(N):
        H = (-1 + 2*np.random.random((n, n)))*30
        H = H + H.T
        # H = Regularize(H, 0.1)
        H = AbsRegularize(H)

        g = (-1 + 2*np.random.random((n,)))*25
        x = (-1 + 2*np.random.random((n,)))*50

        bounds = [-3*np.ones((n,)), 5*np.ones((n,))]
        t0 = timer()
        xo, _ = ProjectedNewton(x, H, g, bounds, verbose=False, iter_max=20, tol=1e-9)
        tPN = timer()
        tsolve.append(tPN-t0)
        # print("\tProjNewton solver time: {:.5g} s".format(tPN-t0))
        # Notice: This is not a QP-specific solver so we do not compare run times.
        # It is used solely to check the correctness of the solution returned. They are much slower for large problems (n > 300)
        if n < 100 and False:
            sol1 = minimize(_fQuad(H, g), x, args=(), method='L-BFGS-B', jac=lambda x: g + H.dot(x), hess=None, hessp=None, bounds=list(zip(*bounds)), constraints=(), tol=1e-12, callback=None, options=None)
            sol2 = minimize(_fQuad(H, g), x, args=(), method='SLSQP', jac=lambda x: g + H.dot(x), hess=None, hessp=None, bounds=list(zip(*bounds)), constraints=(), tol=1e-12, callback=None, options=None)
        # print("Solution difference (scipy)= {:.3g}".format(np.linalg.norm(sol2.x-sol1.x, np.inf)))
        # print("Solution difference (QuasiNewton)= {:.3g}".format(np.linalg.norm(sol1.x-xo, np.inf)))
        # print("Solution difference (SQP) = {:.3g}".format(np.linalg.norm(sol2.x-xo, np.inf)))
        # print(xo)
        # print(sol1)

    # plt.plot(np.sort(tsolve)*1000)

    print("Percentiles:")
    for p in [90, 95, 99]:
        print("{}% = {:.3f} ms".format(p, np.percentile(tsolve, p)*1000))
    print("Max = {:.3f} ms".format(np.max(tsolve)*1000))

    seaborn.kdeplot(np.array(tsolve)*1000, cumulative=True)
    plt.xlabel("QP Solver Time (ms)")
    plt.show()

if __name__ == "__main__":
    test_checks()
    test_scalar()
    test()