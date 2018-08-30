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

def SDREC(x, tf, A, B, C, Q, R, Sf, z, n_points=100, minU=None, maxU=None):
    """ Version of SDRE from the Lunar Lander thesis with terminal constraints 
        Assumes Q,R are functions of the state but may be constant matrices 
        Sf is the final state weighting for non constrained states 
        z is the vector of final constraints such that Cx=z
    """
    n_points -= 1 
    
    if not callable(Q):
        Qf = lambda x: Q 
    else:
        Qf=Q
    if not callable(R):
        Rf = lambda x: R 
    else:
        Rf=R
    
    T = np.linspace(0, tf, n_points)
    dt = tf/(n_points-1.0)
    X = [x] 
    r = Rf(x)
    U = []
    K = []
    
    for iter, t in zip(range(n_points), T):
        if not (iter)%np.ceil(n_points/10.):
            print( "Step {}".format(iter))
            
        a = A(x)
        b = B(x)
        c = C(x)
        q = Qf(x)
        r = Rf(x)
        
        S = DRE(t, tf, Sf, a,b,q,r,n_points-iter)
        Kall = [sdre_feedback(b,r,s) for s in S]
        K.append(Kall[-1])
        V = integrateV(dt, c.T, a, b, Kall)
        P = integrateP(dt, b, r, V)

        u = -(Kall[-1] - np.linalg.solve(r,b.T.dot(V[-1])).dot(np.linalg.solve(P[-1],V[-1].T))).dot(x) - np.linalg.solve(r,b.T.dot(V[-1])).dot(np.linalg.solve(P[-1],z))
        if maxU is not None and np.linalg.norm(u) > maxU:
            u *= maxU/np.linalg.norm(u)
        if minU is not None and np.linalg.norm(u) < minU:
            u *= minU/np.linalg.norm(u)    
        U.append(u)

        x = step(x, dt, u, a, b)
        X.append(x)

    # J = sdre_cost(T, X[:-1], U[:-1], C, Q, R, z)
    # print "Cost: {}".format(J)
    U.append(u)
    K.append(K[-1])
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
            tf    -   solution horizon (independent variable need not be time) 
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
            K     -   time-varying feedback gains
    """        
    from scipy.linalg import solve_continuous_are as care
    
    T = np.linspace(0, tf, n_points)
    dt = tf/(n_points-1.0)
    X = [x] 
    U = [np.zeros(R(x).shape[0])]
    K = []
    for iter, t in zip(range(n_points), T):
        if not (iter)%np.ceil(n_points/10.):
            print( "Step {}".format(iter))
        
        # Get current system approximation 
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
        
        # Save the feedback gains 
        K.append(sdre_feedback(b,r,p))
        
        # Solve the feedforward control:
        s = -matrix_solve((a-dot(S,p)).T, dot(c.T,dot(q,z(t+h))))                    # Can introduce a problem-dependent offset here as anticipatory control to reduce lag
        u = sdre_control(x, b, r, p, s)
        U.append(u)
        
        # Step forward 
        x = odeint(sdre_dynamics, x, np.linspace(t,t+dt,3), args=(a, b, r, p, s))[-1]
        X.append(x)
        
    J = sdre_cost(T, X[:-1], U[:-1], C, Q, R, z)
    print( "Cost: {}".format(J))
    
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
   
def asre_feedback(x, u, B, R, Pv, n):

    P = [Pi.reshape((n,n)) for Pi in Pv]
    # return [dot(matrix_solve(R(xi),B(xi, ui).T),Pi) for xi,ui,Pi in zip(x,u,P)]
    return [dot(matrix_solve(R(ui),B(xi, ui).T),Pi) for xi,ui,Pi in zip(x,u,P)] # R AS FUNCTION OF CONTROL
   
   
   
def asre_integrateV(dt, Vf, A, B, K, x, u): 
    Vshape = Vf.shape 
    V = [Vf.flatten()]
    for xi,ui,k,dti in zip(x[::-1], u[::-1], K[::-1], dt[::-1]):
        a = A(xi)
        b = B(xi,ui)
        V.append(odeint(V_dynamics, V[-1], [0,dti], args=(a,b,k,Vshape))[-1])

    return np.array([v.reshape(Vshape) for v in V])     
   
def asre_Pdynamics(P, t, V, B, R, n):
    P.shape = (n,n)
    return -V.T.dot(B).dot(np.linalg.solve(R,B.T.dot(V))).flatten()
    
def asre_integrateP(dt, V, B, R, x, u, n):
    P = [np.zeros((n,n)).flatten()]
    for xi,ui,Vi,dti in list(zip(x,u,V,dt))[::-1]:
        # P.append(odeint(asre_Pdynamics, P[-1], [0,dti], args=(Vi,B(xi,ui),R(xi),n))[-1])
        P.append(odeint(asre_Pdynamics, P[-1], [0,dti], args=(Vi,B(xi,ui),R(ui),n))[-1]) # R AS FUNCTION OF CONTROL
    return np.array([p.reshape((n,n)) for p in P])
   
def asrec_dynamics(x,t,A,B,R,K,P,V,z, ubounds=None):   
    u = asrec_control(x,A,B,R,K,P,V,z, ubounds[0],ubounds[1])
    return A.dot(x) + B.dot(u)
   
def asrec_control(x,A,B,R,K,P,V,z,ul=None,ub=None):
    rb = np.linalg.solve(R,B.T)
    
    try:
        u = -(K - rb.dot(V).dot(np.linalg.solve(P,V.T))).dot(x) - rb.dot(V).dot(np.linalg.solve(P,z))
    except np.linalg.LinAlgError as e:
        u = -(K - rb.dot(V).dot(np.linalg.lstsq(P,V.T)[0])).dot(x) - rb.dot(V).dot(np.linalg.lstsq(P,z)[0])
        
    
    if ub is not None and np.linalg.norm(u) > ub:
        u *= ub/np.linalg.norm(u)
    if ul is not None and np.linalg.norm(u) < ul:
        u *= ul/np.linalg.norm(u)    
        
    return u
    
def asrec_cost(t, x, u, Q, R, F):
    integrand = 0.5*np.array([dot(xi.T,dot(Q(xi),xi)) + dot(ui,dot(R(ui),ui.T)) for xi,ui in zip(x,u)]).flatten() # # R AS FUNCTION OF CONTROL
    J0 = 0.5*dot(x[-1].T,dot(F,x[-1]))
    return J0 + trapz(integrand, t)    
    
def ASREC(x0, t, A, B, C, Q, R, F, z, max_iter=50, tol=0.01, maxU=None, minU=None, guess=None):
    """ Approximating Sequence of Riccati Equations with Terminal Constraints Cx=z """
    from scipy.interpolate import interp1d
    
    interp_type = 'cubic'
    
    # Problem size
    n = x0.size
    m = R(x0).shape[0]
    
    n_discretize = len(t)
    dt = np.diff(t)
    tb = t[::-1]                            # For integrating backward in time
    
    converge = tol + 1
    Jold = -1e16
    print( "Approximating Sequence of Riccati Equations")
    start_iter = 0 
    if guess is not None:
        start_iter = 1 
        x = guess['state']
        u = guess['control'] 
        
    for iter in range(start_iter, max_iter):
        print( "Current iteration: {}".format(iter+1))
        
        if not iter: # LTI iteration
            u = [np.zeros((m))]*n_discretize
            x = [x0]*n_discretize  # This is the standard approach, but it seems like it would be far superior to actually integrate the system using the initial control
            
        # Riccati equation for feedback solution
        Pf = F.flatten()
        Pv = odeint(dP, Pf, tb, args=(A, B, lambda x: np.eye(n), Q, R, lambda t: x0, lambda t: np.zeros((m,1))))[::-1]          
        K = asre_feedback(x, u, B, R, Pv, n)
        V = asre_integrateV(dt, C.T, A, B, K, x, u)[::-1]
        P = asre_integrateP(dt, V, B, R, x, u, n)[::-1]
        
        # Compute new state trajectory and control   
        x = [x0]
        u = [u[0].T]
        for stage in range(n_discretize-1):
            x.append(odeint(asrec_dynamics, x[-1], [0, dt[stage]], args=(A(x[-1]), B(x[-1],u[-1]), R(x[-1]), K[stage], P[stage], V[stage], z, (minU,maxU)))[-1]) 
            u.append(asrec_control(x[-2], A(x[-2]), B(x[-2],u[-1]), R(u[-1]), K[stage], P[stage], V[stage], z, minU,maxU).T) # R AS FUNCTION OF CONTROL
            
        J = asrec_cost(t, x, u, Q, R, F)
        converge = np.abs(J-Jold)/J
        Jold = J 
        u = u[1:]
        u.append(np.zeros((m)))
        print( "Current cost: {}".format(J))
        
        if converge <= tol:
            print( "Convergence achieved. ")
            break
    u[-1] = u[-2]
    return np.array(x), np.array(u), np.array(K)     
    
    
    
def ASRE(x0, tf, A, B, C, Q, R, F, z, max_iter=10, tol=0.01, n_discretize=250, guess=None):
    """ Approximating Sequence of Riccati Equations """
    from scipy.interpolate import interp1d
    
    interp_type = 'cubic'
    
    # Problem size
    n = x0.size
    m = R(x0).shape[0]
    
    t = np.linspace(0, tf, n_discretize)
    tb = t[::-1]                            # For integrating backward in time
    
    converge = tol + 1
    print( "Approximating Sequence of Riccati Equations")
    print( "Max iterations: {}".format(max_iter))
    start_iter = 0 
    if guess is not None:
        start_iter = 1 
        timeGuess = guess['time']
        stateGuess = guess['state']
        controlGuess = guess['control']
        x = np.interp(t, timeGuess,stateGuess)
        u = np.interp(t, timeGuess,controlGuess)
    
    for iter in range(start_iter, max_iter):
        print( "Current iteration: {}".format(iter+1))
        
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
        
        print( "Current cost: {}".format(J))
        if converge <= tol:
            print( "Convergence achieved. ")
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
    # if 0:
        # print r.shape
        # print b.shape
        # print c.shape
        # print p.shape
        # print a.shape
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

def replace_nan(x,replace=1.):
    """ A useful method for use in SDC factorizations. """
    if np.isnan(x) or np.isinf(x):
        return replace
    else:
        return x

        
# ############## SRP (TIME) ##############  
def SRP_A(x):
    return np.array([[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,-3.71*replace_nan(1/x[2],1),0,0,0]])
    
def SRP_B(x):
    return np.concatenate((np.zeros((3,3)),np.eye(3)),axis=0)

def SRP_Bu(x,u):
    return np.concatenate((np.zeros((3,3)),np.eye(3)),axis=0)    
    
def SRP_C(x):
    return np.eye(6)
    
def SRP(N=20):
    from scipy.integrate import cumtrapz 
    import time 
    from TrajPlot import TrajPlot as traj3d
    m0 = 8500.
    x0 = np.array([-3200., 400, 2600, 625., -60, -270.])
    tf = 15
    r = np.zeros((6,))
    R = lambda x: np.eye(3)
    # R = lambda x: np.diag([replace_nan(1/np.abs(x[i]),1) for i in range(3)])
    Q = lambda x: np.zeros((6,6))
    S = np.zeros((6,6))
    
    from functools import partial 
    solvers = [
               # partial(SDREC, tf=tf, A=SRP_A, B=SRP_B, C=SRP_C, Q=Q, R=R, Sf=S, z=r, n_points=N, maxU=70,minU=0),
               partial(ASREC, t=np.linspace(0,tf,50), A=SRP_A, B=SRP_Bu, C=np.eye(6), Q=Q, R=R, F=S, z=r, tol=1e-2, maxU=70,minU=40)
              ]
    # labels = ['SDRE','ASRE']
    labels = ['ASRE']
    
    for solver,label in zip(solvers,labels):
        t0 = time.time()
        x,u,K = solver(x0)
        print( "{} solution time: {} s".format(label,time.time()-t0))
        
        t = np.linspace(0,tf,x.shape[0])
        T = np.linalg.norm(u,axis=1)
        m = m0*np.exp(-cumtrapz(T/(9.81*290),t,initial=0))
        print( "Prop used: {} kg".format(m0-m[-1]))
        
        
        # from scipy.interpolate import splrep, splev, splder, BSpline
        # xsp = [splrep(t,x[:,i]) for i in range(3)]
        # print len(xsp[0][0])
        # print len(xsp[0][1])
        # vsp = [splder(spl) for spl in xsp]
        # print xsp[0][1]
        # print vsp[0][1]
        # tsp = np.linspace(0,tf,10)
        
        plt.figure(6)
        plt.plot(t,x[:,0:3])
        # for i in range(3):
            # plt.plot(tsp, splev(tsp, xsp[i]),'o--')
        plt.xlabel('Time (s)')
        plt.ylabel('Positions (m)')
        
        plt.figure(1)
        plt.plot(t,x[:,3:6])
        # for i in range(3):
            # plt.plot(tsp, splev(tsp, xsp[i], 1),'o--')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocities (m/s)')

        
        plt.figure(3)
        plt.plot(np.linalg.norm(x[:,3:5],axis=1),x[:,5])
        plt.xlabel('Horizontal Velocity (m/s)')
        plt.ylabel('Vertical Velocity (m/s)')
        
        plt.figure(2)
        plt.plot(t,u[:,0]/T,label='x - {}'.format(label))
        plt.plot(t,u[:,1]/T,label='y - {}'.format(label))
        plt.plot(t,u[:,2]/T,label='z - {}'.format(label))
        plt.xlabel('Time (s)')
        plt.title('Control Direction')
        plt.legend()
        
        plt.figure(5)
        plt.plot(t,T)
        plt.xlabel('Time')
        plt.title('Thrust accel ')
        
        plt.figure(4)
        plt.plot(t,m)
        plt.xlabel('Time')
        plt.title('Mass')
        
        traj3d(*(x[:,0:3].T), T=300*u/np.tile(T,(3,1)).T, figNum=7,label=label)
        
        # plt.figure(8)
        # for k in range(3):
            # for j in range(3):
                # plt.plot(t, K[:,j,k],label='K[{},{}]'.format(j,k))
    plt.show() 
    return t,x,u 
    
 # ############## SRP (ALTITUDE) ##############  
def SRP_A_alt(x):
    return np.array([[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,-3.71/x[4]]])/x[4]
    
def SRP_B_alt(x):
    return np.concatenate((np.zeros((2,3)),np.eye(3)/x[4]),axis=0)

def SRP_Bu_alt(x,u):
    return np.concatenate((np.zeros((2,3)),np.eye(3)/x[4]),axis=0)    
    
def SRP_C_alt(x):
    return np.eye(5)
    
def SRP_alt():
    from scipy.integrate import cumtrapz 
    import time 
    
    m0 = 8500.
    z0 = 2600
    x0 = np.array([-3200., 400, 625., -60, -270.])
    # tf = 13
    # r = np.zeros((5,))
    r = np.array([0,0,0,0,-20])
    R = lambda x: np.eye(3)
    Q = lambda x: np.zeros((5,5))
    S = np.zeros((5,5))
       
    # z = np.linspace(z0,0,75)   
    z = np.logspace(np.log(z0)/np.log(5),0,50,base=5)   
    x,u,K = ASREC(x0=x0,t=z, A=SRP_A_alt, B=SRP_Bu_alt, C=np.eye(5), Q=Q, R=R, F=S, z=r,  tol=1e-2, maxU=70, minU=40,max_iter=2)  
      
    t = cumtrapz(1/x[:,4],z,initial=0)  
    T = np.linalg.norm(u,axis=1)
    m = m0*np.exp(-cumtrapz(T/(9.81*280),t,initial=0))
    print( "Prop used: {} kg".format(m0-m[-1])  )
    
    # plt.figure(2)
    # plt.plot(z,"o")
      
    label='ASRE'   
    plt.figure(1)
    plt.plot(np.linalg.norm(x[:,0:2],axis=1),z)
    plt.xlabel('Distance to Target (m)')
    plt.ylabel('Altitude (m)')
    plt.figure(3)
    plt.plot(np.linalg.norm(x[:,2:4],axis=1),x[:,4])
    plt.xlabel('Horizontal Velocity (m/s)')
    plt.ylabel('Vertical Velocity (m/s)')   
    
    plt.figure(2)
    plt.plot(t,u[:,0]/T,label='x - {}'.format(label))
    plt.plot(t,u[:,1]/T,label='y - {}'.format(label))
    plt.plot(t,u[:,2]/T,label='z - {}'.format(label))
    plt.xlabel('Time (s)')
    plt.title('Control Direction')
    plt.legend()
    
    plt.figure(5)
    plt.plot(t,T)
    plt.xlabel('Time')
    plt.title('Thrust accel ')
    
    plt.figure(4)
    plt.plot(t,m)
    plt.xlabel('Time')
    plt.title('Mass') 
    
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

    print( "ASRE: {} s".format(t_asre))
    print( "SDRE: {} s".format(t_sdre))
    
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
    # SRP_alt()