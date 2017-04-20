""" Pure python implementation of a projected newton method for solving linearly constrained quadratic programs """


import numpy as np 

def ProjectedNewton(x0, hessian, gradient, bounds, tol=1e-6):
    """
    Inputs:
    x0          -   an initial guess, need not be feasible, length n 
    hessian     -   curvature matrix of cost function, nxn matrix 
    gradient    -   derivative of cost function, length n  
    bounds      -   tuple of (nlower bounds, nupper bounds)
    tol         -   convergence tolerance 
    
    """
    iter = 0
    iterMax = 100 
    n = len(x0)
    
    x = np.clip(x0, bounds[0],bounds[1]) # Make the initial point feasible 
    xl = np.asarray(bounds[0])
    xu = np.asarray(bounds[1])
    hessian = np.asarray(hessian)
    gradient = np.asarray(gradient)
    
    while iter < iterMax:
        g = gradient + np.dot(hessian, x)
        
        # Determine the active set (and the complementary inactive set)
        idu = (x-xu) == 0
        idl = (xl-x) == 0
        gu  = g>0
        gl  = g<0 

        c = ((gl+idu) > 1) + ((gu+idl) > 1)
        f = ~c 
        c = np.where(c)[0]
        f = np.where(f)[0]

        hff =  hessian[f,:][:,f]
        gf = gradient[f] + np.dot(hff,x[f])

        if np.any(c):
            print "Some clamped direction"
            hfc =  hessian[f,:][:,c]
            gf += np.dot(hfc,x[c])
        
        if np.linalg.norm(gf) < tol:
            break 
        
        dx = np.zeros((n,))
        dx[f] = -np.dot(np.linalg.inv(hff), gf)

        alpha = armijo(fQuad(hessian,gradient), x, dx, g, xl, xu)
        x = np.clip(x+alpha*dx, xl, xu)
        iter += 1
    fopt = fQuad(hessian,gradient)(x)
    print "Total iterations: {}".format(iter)
    print x 
    print fopt 
    return x, hff, fopt
    
def armijo(f,x,dx,g,xl,xu):
    gamma = 0.1
    c = 0.5 
    alpha = 2*np.max(xu-xl)
    r = 0.5*gamma 
    while r<gamma:
        alpha *= c 
        xa = np.clip(x+alpha*dx,xl,xu)
        r = (f(x)-f(xa))/np.dot(g.T, (x-xa))
    return alpha 
        
def fQuad(h,g):
    def fQuadFun(x):
        return 0.5*np.dot(x.T, np.dot(H,x)) + np.dot(g.T,x)
    return fQuadFun

        
if __name__ == "__main__":
    H = [[2,0],[0,2]]
    g = [1,1]
    x = [1,3]
    
    ProjectedNewton(x,H,g,[-3,3])