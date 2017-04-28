""" Riccati equation based nonlinear control methods """

from numpy import sin, cos, tan, dot, arccos
import numpy as np
from scipy.linalg import solve as matrix_solve
from scipy.integrate import simps as trapz
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from functools import partial
from itertools import product


def controller(A, B, C, Q, R, z, method='SDRE',**kwargs):

    x = np.array([kwargs['current_state'][0],kwargs['velocity'], kwargs['fpa'], kwargs['lift'], kwargs['drag']])
    t = kwargs['velocity']
    u = np.clip(sdre_step(x, t, A, B, C, Q, R, z, h=0)[0],-1,1)

    return arccos(u)*np.sign(kwargs['bank'])

# ################################################################################################
#                                State Dependent Riccati Equation                                #
# ################################################################################################

def SDREC(x, tf, A, B, C, Q, R, Sf, r, n_points=200, h=0):
    """ Version of SDRE from the Lunar Lander paper with terminal constraints 
        Assumes Q,R,Sf are constant matrices 
        r is the vector of final constraints such that Cx=r
    """
    from scipy.linalg import solve_continuous_are as care
    
    T = np.linspace(0, tf, n_points)
    dt = tf/(n_points-1.0)
    X = [x] 
    U = [np.zeros(R.shape[0])]
    K = []
    
    for iter, t in zip(range(n_points), T):
        if not (iter)%np.ceil(n_points/10.):
            print "Step {}".format(iter)
            
        a = A(x)
        b = B(x)
        c = C(x)
        
        S = DRE(t, tf, Sf, a,b,Q,R,n_points-iter)
        Kall = [sdre_feedback(b,R,s) for s in S]
        K.append(Kall[-1])
        V = integrateV(dt, c.T, a, b, Kall)
        P = integrateP(dt, b, R, V)

        u = -(Kall[-1] - np.linalg.solve(R,b.T.dot(V[-1])).dot(np.linalg.solve(P[-1],V[-1].T))).dot(x) - np.linalg.solve(R,b.T.dot(V[-1])).dot(np.linalg.solve(P[-1],r))
        U.append(u)

        x = step(x,dt,u,a,b)
        X.append(x)
        
    
    
    # J = sdre_cost(T, X[:-1], U[:-1], C, Q, R, z)
    # print "Cost: {}".format(J)
    
    return np.array(X), np.array(U), np.array(K)   

def sdrec_dynamics(x,t,a,b,u):
    return a.dot(x) + b.dot(u)
    
def step(x, dt, u, a, b):
    
    return odeint(sdrec_dynamics,x,[0,dt],args=(a,b,u))[-1]
    #dx = a*x + b*u
    
    
def integrateP(dt, B, R, V): 
    
    nConstraints = V[0].shape[1]
    P = [np.zeros((nConstraints**2,))]
    
    for i,v in enumerate(V):
        P.append(odeint(P_dynamics, P[-1], [0,dt], args=(B,R,v))[-1])
    
    return np.array([p.reshape((nConstraints,nConstraints)) for p in P])     
    
    
def P_dynamics(P,T, B,R,V):
    n = int(np.sqrt(P.size))
    P.shape = ((n,n))
    
    Pdot = -V.T.dot(B).dot(np.linalg.solve(R,B.T.dot(V)))
    return Pdot.flatten()
    
    
def riccati_dynamics(S,T,A,B,Q,R):    
    n = int(np.sqrt(S.size))
    S.shape = (n,n) # Turn into a matrix 
    # print A.shape 
    # print B.shape 
    # print Q.shape 
    # print R.shape 
    dS = A.T.dot(S) + S.dot(A) - S.dot(B).dot(np.linalg.solve(R,B.T.dot(S))) + Q 
    
    return dS.flatten()
    
def DRE(tcurrent, tfinal, Sf, A,B,Q,R,nRemaining):
    x0 = Sf.flatten()
    n = int(x0.size**0.5)
    Svec = odeint(riccati_dynamics, x0, np.linspace(tcurrent, tfinal,nRemaining), args=(A,B,Q,R))
    
    S = [s.reshape((n,n)) for s in Svec]
    return np.array(S)
    
def V_dynamics(V,T,A,B,K,Vshape):
    V.shape = Vshape 
    Vdot = (A - B.dot(K)).T.dot(V)
    return Vdot.flatten()
    
def integrateV(dt, Vf, A, B, K): 
    Vshape = Vf.shape 
    V = [Vf.flatten()]
    for i,k in enumerate(K):
        V.append(odeint(V_dynamics, V[-1], [0,dt], args=(A,B,k,Vshape))[-1])

    return np.array([v.reshape(Vshape) for v in V])    
    
def SDRE(x, tf, A, B, C, Q, R, z, n_points=200, h=0):  
    """ 
        Inputs:
            x     -   current state
            tf    -   solution horizon (independent variable need not be time)  #TODO: This needs to be a time span instead. Or split into a single step method and a propgator.
            A(x)  -   function returning SDC system matrix 
            B(x)  -   function returning SDC control matrix 
            C(x)  -   function returning SDC output matrix 
            Q(x)  -   function returning LQR tracking error matrix
            R(t)  -   function returning LQR control weight matrix
            z(t)  -   function returning the reference signal(s) at each value of the independent variable. Single step version can take a single value instead of function.
            
        Optional Inputs:
            Qf   -    final state error matrix, also the final condition for the Riccati equation when solving a terminally constrained problem 
            
        Outputs:
            x     -   state vector at n_points
            u     -   control history at n_points
            K     -   Nonlinear feedback gains
    """        
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
        s = -matrix_solve((a-dot(S,p)).T, dot(c.T,dot(q,z(t+h))))                    # Can introduce a problem-dependent offset here as anticipatory control to reduce lag
        u = sdre_control(x, b, r, p, s)
        U.append(u)

        xnew = odeint(sdre_dynamics, x, np.linspace(t,t+dt,3), args=(a, b, r, p, s))
        x = xnew[-1,:]
        X.append(x)
        
    J = sdre_cost(T, X[:-1], U[:-1], C, Q, R, z)
    print "Cost: {}".format(J)
    
    return np.array(X), np.array(U), np.array(K)   

def sdre_step(x, t, A, B, C, Q, R, z, h=0, args=()):
    from scipy.linalg import solve_continuous_are as care

    a = A(x)
    n = a.shape[0]
    b = B(x)
    c = C(x)
    q = Q(x)
    r = R(x)
    # r = R(t)
    S = dot(b, matrix_solve(r, b.T))
    qc = dot(c.T, dot(q, c))

    # Solve the CARE:
    p = care(a, b, qc, r)
    
    # Solve the feedforward control:
    s = -matrix_solve((a-dot(S,p)).T, dot(c.T,dot(q,z(t+h))))
    # for val in [x,b,r,p,s]:
        # print val.shape
    u = sdre_control(x[0:n], b, r, p, s)
    return u
    
    
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
    print "Approximating Sequence of Riccati Equations"
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
    """ A useful method for use in SDC factorizations. """
    if np.isnan(x):
        return replace
    else:
        return x

        
# ############## 2D SRP ##############  
def SRP_A(x):
    return np.array([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,-3.71/x[1],0,0]])
    
def SRP_B(x):
    return np.concatenate((np.zeros((2,2)),np.eye(2)),axis=0)
    
def SRP_C(x):
    return np.eye(4)
    
def SRP():
    m0 = 8500.
    x0 = np.array([-3200., 2700, 625.,-270.]);
    tf = 13     
    r = np.zeros((4,))
    R = np.eye(2)
    Q = np.zeros((4,4))
    S = Q
    
    x,u,K = SDREC(x0, tf, SRP_A, SRP_B, SRP_C, Q, R, S, r, n_points=100)
    t = np.linspace(0,tf,u.shape[0])
    
    plt.figure(1)
    plt.plot(x[:,0],x[:,1])
    plt.figure(3)
    plt.plot(x[:,2],x[:,3])
    
    plt.figure(2)
    plt.plot(t,u,label='SDRE')
    plt.title('Control history')
    plt.legend()
    plt.show() 
    
# ############## Inverted Pendulum ##############        
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
    x,u,K = ASRE(x0, tf, IP_A, IP_B, lambda x: C, lambda x: Q, IP_R, lambda x: F, IP_z, max_iter=50, tol=0.1)      # Time-varying R
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
    x,u,K = SDRE(x0, tf, IP_A, lambda x: IP_B(x,0), lambda x: C, lambda x: Q, IP_R, IP_z, n_points=75,h=0.1)      # Time-varying R
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
    # test_IP()
    SRP()