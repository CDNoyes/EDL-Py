""" Riccati equation based nonlinear control methods """

from numpy import sin, cos, tan, dot
import numpy as np
from scipy.linalg import solve as matrix_solve
from scipy.integrate import simps as trapz
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from functools import partial
from itertools import product


# ################################################################################################
#                                State Dependent Riccati Equation                                #
# ################################################################################################


def SDRE(x, tf, A, B, C, Q, R, z, n_points=200):  
    from scipy.linalg import solve_continuous_are as care
    
    T = np.linspace(0, tf, n_points)
    dt = tf/(n_points-1.0)
    X = [x] 
    U = [np.zeros(R(x).shape[0])]
    K = []
    for iter, t in zip(range(n_points), T):
        if not (iter)%np.ceil(n_points/10.):
            print "Step {}".format(iter)
        a = A(x)
        b = B(x)
        c = C(x)
        q = Q(x)
        # r = R(x)
        r = R(t)
        S = dot(b, matrix_solve(r, b.T))
        qc = dot(c.T, dot(q, c))

        # Solve the CARE:
        p = care(a, b, qc, r)
        K.append(sdre_feedback(b,r,p))
        
        # Solve the feedforward control:
        s = -matrix_solve((a-dot(S,p)).T, dot(c.T,dot(q,z(t))))
        xnew = odeint(sdre_dynamics, x, np.linspace(t,t+dt,10), args=(a, b, r, p, s))
        x = xnew[-1,:]
        u = sdre_control(x, b, r, p, s)
        X.append(x)
        U.append(u)
        
    J = sdre_cost(T, X[:-1], U[:-1], C, Q, R, z)
    print "Cost: {}".format(J)
    
    return np.array(X), np.array(U), np.array(K)   
    
def sdre_control(x, b, r, p, s):   
    return -dot(matrix_solve(r,b.T), dot(p,x)-np.reshape(s,-1))
    
def sdre_dynamics(x, t, a, b, r, p, s):
    """ Closed-loop dynamics integrated in SDRE """
    S = dot(b,matrix_solve(r,b.T))
    return (dot(a - dot(S,p),x) + np.reshape(dot(S,s),-1))
    
def sdre_cost(t, x, u, C, Q, R, z):
    """ Estimation of the final cost in SDRE"""
    e = z(t).flatten() - np.array([[dot(C(xi),xi)] for xi in x]).flatten()
    integrand = np.array([dot(ei,dot(Q(xi),ei)) + dot(ui,dot(R(ti),ui)) for ti,xi,ui,ei in zip(t,x,u,e)]).flatten()
    return trapz(integrand,t)    
    
def sdre_feedback(b,r,p):
    return dot(matrix_solve(r,b.T),p)
    
    
# ################################################################################################
#                           Approximating Sequence of Riccati Equations                          #
# ################################################################################################
   
def ASRE(x0, tf, A, B, C, Q, R, F, z, max_iter=10, tol=0.01, n_discretize=250):
    """ Approximating Sequence of Riccati Equations """
    from scipy.interpolate import interp1d
    
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
            converge = np.abs(J-Jold)/J
        
        print "Current cost: {}".format(J)
        if converge <= tol:
            print "Convergence achieved. "
            break
            
        # Reshape Pv and output    
        K = [sdre_feedback(B(xc,np.zeros(m)),R(tc),np.reshape(p,(n,n))) for tc,xc,p in zip(t,x,Pv[::-1])]    
        
    return x, u, np.array(K)       
    
    
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
    """ Used in ASRE """
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
    
    
# ################################################################################################
#                                         Test Functions                                         #
# ################################################################################################

# def F8_A(x):
    # return np.array([[-0.877 + 0.47*x[0] + 3.846*x[0]^2-x[0]*x(3), -0.019*x[1], 1-0.088*x[0]]
                     # [0, 0, 1]
                     # [-4.208-0.47*x[0]-3.56*x[0]^2, 0, -0.396]])
                     
# def F8_B(x,u):
    # return
def replace_nan(x,replace=1.):
    if np.isnan(x):
        return replace
    else:
        return x
        
def IP_A(x):
    return np.array([[0,1],[4*4.905*replace_nan(sin(x[0])/x[0],1), -0.4]])
    
def IP_B(x,u):
    return np.array([[0],[10]])
 
def IP_z(t):
    return np.array([[sin(t)+cos(2*t-1)]])
    
def IP_R(t):
    return np.array([[1 + 200*np.exp(-t)]])
    
def test_IP():
    import time
    R = np.array([10])
    R.shape = (1,1)
    C = np.array([[1,0]])
    x0 = np.zeros((2)) + 1
    Q = np.array([[1.0e3]])
    F = np.array([[1.0e1]])
    tf = 15
    
    # x,u = ASRE(x0, tf, IP_A, IP_B, lambda x: C, lambda x: Q, lambda x: R, lambda x: F, IP_z, max_iter=2, tol=0.1) # Constant R
    t_init = time.time()
    x,u,K = ASRE(x0, tf, IP_A, IP_B, lambda x: C, lambda x: Q, IP_R, lambda x: F, IP_z, max_iter=5, tol=0.01)      # Time-varying R
    t_asre = -t_init + time.time()
    
    t = np.linspace(0,tf,u.size)
    plt.figure(1)
    plt.plot(t,x[:,0],label='ASRE')
    plt.plot(t,IP_z(t).flatten(),'k--',label='Reference')
    plt.figure(2)
    plt.plot(t,u,label='ASRE')
    
    Kplot = np.transpose(K,(1,2,0))
    plt.figure(3)
    for gain in product(range(K.shape[1]),range(K.shape[2])):
        plt.plot(t,Kplot[gain],label='ASRE {}'.format(gain))    
    
    t_init = time.time()
    x,u,K = SDRE(x0, tf, IP_A, lambda x: IP_B(x,0), lambda x: C, lambda x: Q, IP_R, IP_z, n_points=750)      # Time-varying R
    t_sdre = -t_init + time.time()

    print "ASRE: {} s".format(t_asre)
    print "SDRE: {} s".format(t_sdre)
    
    t = np.linspace(0,tf,u.size)
    plt.figure(1)
    plt.plot(t,x[:,0],label='SDRE')
    plt.title('Output history')
    plt.figure(2)
    plt.plot(t,u,label='SDRE')
    plt.title('Control history')
    plt.legend()
    
    Kplot = np.transpose(K,(1,2,0))
    plt.figure(3)
    for gain in product(range(K.shape[1]),range(K.shape[2])):
        plt.plot(t[:-1],Kplot[gain],label='SDRE {}'.format(gain))    
    plt.title('Feedback gains')    
    plt.legend()    
    
    plt.show()
    
    
if __name__ == '__main__':    
    test_IP()