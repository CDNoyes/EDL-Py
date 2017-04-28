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
    
    if debug:
        print "Inputs:  "
        print "Hessian = {}".format(hessian)
        print "Gradient = {}".format(gradient)
        print "Initial guess = {}".format(x0)
    
    iter = 0
    iterMax = 100
    n = len(x0)
    
    x = np.clip(x0, bounds[0],bounds[1]) # Make the initial point feasible 
    if n > 1:
        x = x[0]
    xl = np.asarray(bounds[0])
    xu = np.asarray(bounds[1])
    hessian = np.asarray(hessian)
    gradient = np.asarray(gradient)
    
    if debug:
        print "After some transforms:  "
        print "Hessian = {}".format(hessian)
        print "Gradient = {}".format(gradient)
        print "Initial feasible guess = {}\n".format(x)
    
    while iter < iterMax:
        g = gradient + np.dot(hessian, x.T).T
 
        # Determine the active set (and the complementary inactive set)
        idu = (x-xu) == 0
        idl = (xl-x) == 0
        gu  = g>0
        gl  = g<0 
        if debug:
            print "g.shape = {}".format(g.shape)

        c = ((gl.astype(int)+idu.astype(int)) > 1) + ((gu.astype(int)+idl.astype(int)) > 1)
        f = ~c.astype(bool) 
        f = np.where(f[0])[0]

        hff =  hessian[f,:][:,f]
        if debug:
            print "Free dirs: {}".format(f)
            print "Free hess: {}".format(hff)
            print "Free g: {}".format(gradient[f])
            print "Free x: {}".format(x[f])
        if len(f):
            gf = gradient[f] + np.dot(hff,x[f].T).T
        else:
            gf = np.zeros_like(gradient)

        if np.any(c) and not np.all(c):
            # print "Some clamped direction"
            c = np.where(c)[0]
            hfc =  hessian[f,:][:,c]
            gf += np.dot(hfc,x[c])
        
        if np.linalg.norm(gf) < tol:
            break 
        
        dx = np.zeros((n,))
        dx[f] = -np.dot(np.linalg.inv(hff), gf)

        alpha = armijo(fQuad(hessian,gradient), x, dx, g, xl, xu)
        x = np.clip(x+alpha*dx, xl, xu)
        if n > 1:
            x = x[0]
        iter += 1
        
    # fopt = fQuad(hessian,gradient)(x)
    # print "Total iterations: {}".format(iter)
    # print x 
    # print fopt 
    clamped = np.where(c[0])[0]
    H = hessian
    H[clamped,:][:,clamped] = 0
    return x, H # To do: return the full Hessian with clamped elements set to zero already, rather than do it outside 
    
def armijo(f,x,dx,g,xl,xu):
    gamma = 0.1
    c = 0.5 
    alpha = 2*np.max(xu-xl)
    r = 0.5*gamma 
    while r<gamma:
        alpha *= c 
        xa = np.clip(x+alpha*dx,xl,xu)[0]
        r = (f(x)-f(xa))/np.dot(g, (x-xa).T)
    return alpha 
        
def fQuad(h,g):
    def fQuadFun(x):
        return 0.5*np.dot(x, np.dot(h,x.T)) + np.dot(g,x)
    return fQuadFun

        
if __name__ == "__main__":
    from Regularize import Regularize 
    
    n = 1 
    N = 100
    for _ in range(N):
        H = (-1 + 2*np.random.random((n,n)))*3
        H = H + H.T 
        H = Regularize(H, 0.01)
        g = (-1 + 2*np.random.random((n,)))*5
        x = (-1 + 2*np.random.random((n,)))*3.2 
            
            
        ProjectedNewton(x,H,g,[-3,5])
        