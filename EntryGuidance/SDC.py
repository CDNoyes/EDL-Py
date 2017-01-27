""" State-Dependent Coefficient factorizations of longitudinal entry dynamics """

from numpy import sin, cos, tan, dot
import numpy as np
from scipy.linalg import solve as matrix_solve
from scipy.integrate import simps as trapz
import matplotlib.pyplot as plt

from functools import partial

# def range(edl_model):
    # """ SDC using downrange as the independent variable """
    
    # # Assumes the state vector is [r,v,gamma]
    
    # def A(x,L,D):
        # r,v,fpa = x
        # w1 = np.ones(3)/np.sqrt(3)
        # w2 = np.ones(2)/np.sqrt(2)
        
        # A_sdc = [ [0,0,tan(fpa)/fpa], [w1[0]*v_prime/r, w1[1]*v_prime/v, w1[2]*v_prime/fpa], [-w2[0]*g/(r*v**2) + 1/r**2,-w2[1]*g/v**3] ]
        # return np.array(A_sdc)
        
    # def B(x,L):
        # r,v,fpa = x
        # return np.array([0,0,L/(cos(fpa)*v**2)]).shape = (3,1)
        
    # return A,B    
    
    

# def SDRE(x0, tf, A, B, C, Q, R, F, z):    
    
def ASRE(x0, tf, A, B, C, Q, R, F, z, max_iter=10, tol=0.01):
    """ Approximating Sequence of Riccati Equations """
    from scipy.integrate import odeint
    from scipy.interpolate import interp1d
    
    n_discretize = 250                      # Generally small effect on solution quality
    interp_type = 'cubic'
    
    # Problem size
    n = x0.size
    m = R(x0).shape[0]
    
    t = np.linspace(0, tf, n_discretize)
    tb = t[::-1]                            # For integrating backward in time
    
    converge = tol + 1
    print "Max iterations: {}".format(max_iter)
    
    for iter in range(max_iter):
        print "Current iteration: {}".format(iter+1)
        
        if not iter: # LTI iteration
            
            # Riccati equation for feedback solution
            Pf = dot(C(x0).T,dot(F(x0),C(x0))).flatten()
            Pv = odeint(dP, Pf, tb, args=(A, B, C, Q, R, lambda t: x0, lambda t: np.zeros((m,1))))
            Pi = interp1d(tb, Pv, kind=interp_type, fill_value=(Pv[0],Pv[-1]), axis=0, bounds_error=False)
            
            # Feedforward solution
            sf = dot(C(x0).T,dot(F(x0),z(tf))).T[0]
            s = odeint(ds, sf, tb, args=(A, B, C, Q, R, lambda t: x0, lambda t: np.zeros((m,1)), Pi, z))
            si = interp1d(tb, s, kind=interp_type, fill_value=(s[0],s[-1]), axis=0, bounds_error=False)
            
            # Compute new state trajectory and control        
            x = odeint(dynamics, x0, t, args=(A, B, R, Pi, lambda t: np.zeros((m,1)), si))
            u = compute_control(B, R, Pv, x, np.zeros((n_discretize,m)), s, n, t)
            J = compute_cost(t, x, u, C, Q, R, F, z)

        else: # LTV iterations until convergence        
            xf = x[-1]
            xi = interp1d(t, x, kind=interp_type, fill_value=(x[0],x[-1]), axis=0, bounds_error=False)
            ui = interp1d(t, u, kind=interp_type, fill_value=(u[0],u[-1]), axis=0, bounds_error=False)
            
            # Riccati equation for feedback solution
            Pf = dot(C(xf).T,dot(F(xf),C(xf))).flatten()
            Pv = odeint(dP, Pf, tb, args=(A, B, C, Q, R, xi, ui))
            Pi = interp1d(tb, Pv, kind=interp_type, fill_value=(Pv[0],Pv[-1]), axis=0, bounds_error=False)
            
            # Feedforward solution
            sf = dot(C(xf).T,dot(F(xf),z(tf))).T[0]
            s = odeint(ds, sf, tb, args=(A, B, C, Q, R, xi, ui, Pi, z))
            si = interp1d(tb, s, kind=interp_type, fill_value=(s[0],s[-1]), axis=0, bounds_error=False)
            
            # Compute new state trajectory and control        
            xold = np.copy(x)
            Jold = J
            x = odeint(dynamics, x0, t, args=(A, B, R, Pi, ui, si))
            u = compute_control(B, R, Pv, x, u, s, n, t)
            J = compute_cost(t, x, u, C, Q, R, F, z)
            converge = (J-Jold)/J
        
        print "Current cost: {}".format(J)
        if converge <= tol:
            print "Convergence achieved. "
            break
            
    return x,u       
    
    
def compute_control(B,R,Pv,X,U,S,n,T):
    u_new = []
    for x, u, s, pv,t in zip(X,U,S[::-1],Pv[::-1],T):
        pv.shape = (n,n)
        u_new.append( -dot(matrix_solve(R(t),B(x,u).T), (dot(pv,x)-s)) )
        
    return np.array(u_new)  
    
def compute_cost(t, x, u, C, Q, R, F, z):
    e = z(t).flatten() - np.array([[dot(C(xi),xi)] for xi in x]).flatten()
    integrand = np.array([dot(ei,dot(Q(xi),ei)) + dot(ui,dot(R(ti),ui)) for ti,xi,ui,ei in zip(t,x,u,e)]).flatten()
    J0 = 0.5*dot(e[-1],dot(F(x[-1]),e[-1]))
    return J0[0,0] + trapz(integrand, t)
    
    
def dP(p, t, A, B, C, Q, R, X, U):  
    """ Riccati equation """
    n = int(np.sqrt(p.size))
    p.shape = (n,n)      # Turn into a matrix
    x = X(t)
    u = U(t)
    a = A(x)
    b = B(x,u)
    c = C(x)
    q = Q(x)
    # r = R(x)
    r = R(t)
    if 0:
        print r.shape
        print b.shape
        print c.shape
        print p.shape
        print a.shape
    s = dot(b,matrix_solve(r,b.T))

    return (-dot(c.T,dot(q,c)) - dot(p,a) - dot(a.T,p) + dot(p,dot(s,p))).flatten()
    


def ds(s, t, A, B, C, Q, R, X, U, P, z):    
    """ Differential equation of the feedfoward term. """
    p = P(t)
    n = int(np.sqrt(p.size))
    p.shape = (n,n)      # Turn into a matrix
    x = X(t)
    u = U(t)
    a = A(x)
    b = B(x,u)
    c = C(x)
    # r = R(x)
    r = R(t)
    q = Q(x)
    S = dot(b,matrix_solve(r,b.T))
    return -dot((a - dot(S,p)).T,s) - dot(c.T,dot(q,z(t))).T[0]


def dynamics(x, t, A, B, R, P, U, s):
    p = P(t)
    n = int(np.sqrt(p.size))
    p.shape = (n,n)      # Turn into a matrix
    a = A(x)
    u = U(t)
    b = B(x,u)
    # r = R(x)
    r = R(t)
    S = dot(b,matrix_solve(r,b.T))
    return dot(a - dot(S,p),x) + dot(S,s(t))
    
# ########################## #
# Test Functions ########### #
# ########################## #

# def F8_A(x):
    # return np.array([[-0.877 + 0.47*x[0] + 3.846*x[0]^2-x[0]*x(3), -0.019*x[1], 1-0.088*x[0]]
                     # [0, 0, 1]
                     # [-4.208-0.47*x[0]-3.56*x[0]^2, 0, -0.396]])
                     
# def F8_B(x,u):
    # return
def replace_nan(x,replace=1):
    if np.isnan(x):
        return replace
    else:
        return x
        
def IP_A(x):
    return np.array([[0,1],[4*4.905*replace_nan(sin(x[0])/x[0],1), -0.4]])
    
def IP_B(x,u):
    return np.array([[0],[10]])
    
def IP_z(t):
    return np.array([[sin(t)]])
    
def IP_R(t):
    return np.array([[1 + 200*np.exp(-t)]])
    
def test_IP():
    R = np.array([10])
    R.shape = (1,1)
    C = np.array([[1,0]])
    x0 = np.zeros((2)) + 1
    Q = np.array([[1.0e3]])
    F = np.array([[1.0e1]])
    tf = 10
    
    x,u = ASRE(x0, tf, IP_A, IP_B, lambda x: C, lambda x: Q, lambda x: R, lambda x: F, IP_z, max_iter=2, tol=0.1)
    # x,u = ASRE(x0, tf, IP_A, IP_B, lambda x: C, lambda x: Q, IP_R, lambda x: F, IP_z, max_iter=5, tol=0.1)
    t = np.linspace(0,tf,250)
    plt.figure()
    plt.plot(t,x)
    plt.plot(t,sin(t),'k--',label='Reference')
    plt.figure()
    plt.plot(t,u)
    plt.show()
    
if __name__ == '__main__':    
    test_IP()