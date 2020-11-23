""" Implements two versions of Approximating Sequence of Riccati Equations nonlinear control method """

import matplotlib.pyplot as plt
import numpy as np
from numpy import sin, cos, dot
from scipy.linalg import solve as matrix_solve
from scipy.integrate import simps as trapz
from scipy.interpolate import interp1d
from itertools import product

from .RK4 import RK4 as odeint 

interp_type = 'cubic'

# ################################################################################################
#                           Approximating Sequence of Riccati Equations                          #
# ################################################################################################

# TODO: Implement and study the stochastic versions which simply requires adding stochastic terms to the Riccati equations
# TODO: Test state constraints 
# TODO: Implement SRP with integral control and compare 


def ASREC(x0, t, A, B, C, Q, R, F, z, m, max_iter=50, tol=0.01, guess=None, verbose=False):
    """ Approximating Sequence of Riccati Equations with Terminal Constraints Cx=z """
    # Problem size
    n = np.asarray(x0).size
    E = lambda t,x: np.zeros((n,1))
    sigma = 0 

    n_discretize = len(t)
    dt = np.diff(t)
    tb = t[::-1]                            # For integrating backward in time

    converge = tol + 1
    if verbose: print("Approximating Sequence of Riccati Equations")
    start_iter = 0
    if guess is not None:
        start_iter = 1
        x = guess['state']
        u = guess['control']
        
    for iter in range(start_iter, max_iter):
        if verbose: print("Current iteration: {}".format(iter))
        
        if not iter: # LTI iteration
            u = [np.zeros((m,))]*n_discretize
            x = np.array([x0]*n_discretize)  # This is the standard approach, but it seems like it would be far superior to actually integrate the system using the initial control

        # Riccati equation for feedback solution
        Pf = F
        P = odeint(dP, Pf, tb, args=(
            A, B, lambda t,x: np.eye(n), Q, R, lambda T: interp1d(t,x,kind=interp_type, axis=0, bounds_error=False, fill_value=(x[0], x[-1]))(T), lambda T: interp1d(t,u,kind=interp_type, axis=0, bounds_error=False, fill_value=(u[0],u[-1]))(T), E, sigma))[::-1]
        K = asre_feedback(t, x, u, B, R, P)
        V = asre_integrateV(t, C.T, A, B, K, x, u)[::-1]
        P = asre_integrateP(t, V, B, R, x, u)[::-1]  # This P has nothing to do with the previous one 

        # Compute new state trajectory and control
        xold = x 
        x = [x0]
        u = [u[0].T]
        for stage in range(n_discretize-1):
            u.append(asrec_control(x[-1], A(t[stage], x[-1]), B(t[stage],x[-1],u[-1]), R(t[stage],x[-1],u[-1]), K[stage], P[stage], V[stage], z.squeeze()).T) 
            x.append(odeint(asrec_dynamics, x[-1], [0, dt[stage]], args=(A(t[stage], x[-1]), B(t[stage], x[-1], u[-1]), R(t[stage], x[-1], u[-1]), K[stage], P[stage], V[stage], z))[-1])

        x[-1] = x[-2]
        u[0] = u[1]
        u[-1] *= 0
        J = asrec_cost(t, x, u, Q, R, lambda xf: F)
        converge = np.max(np.abs(np.array(x)-xold))
        if verbose: 
            print("Current cost: {}".format(J))
            print("Convergence criteria: {:.3g}\n".format(converge))

        if converge <= tol:
            if verbose: 
                print("Convergence achieved. ")
            break
    u[-1] = u[-2]
    return np.array(x), np.array(u), np.array(K)


def ASRE(x0, tf, A, B, C, Q, R, F, z, m, E=None, sigma=0, max_iter=10, tol=0.01, n_discretize=250, guess=None):
    """ Approximating Sequence of Riccati Equations 

            Solves a nonlinear optimal control problem of the form

                J = 0.5 * e(tf)' F e(tf) + int[e Q e + u R u] dt

            subject to stochastic dynamics: 
                dx =  (A(t,x)x + B(t,x,u)u)dt + E(t,x)dw

                where dw is a zero-mean unit covariance Wiener process 
        
            and the error e is defined by C(t,x)x(t) = z(t)

        When E is not None, and sigma is not zero, the problem solves the exponential form of the above cost.
        This places weight on the higher probability moments of J, i.e. incorporates variance information.
        See "Optimal Stochastic Linear Systems with Exponential Performance Criteria..." by Jacobson 

        Reduces to stabilization (or regulation) problem with z(t) = 0
        C(x) = I implies full-state feedback 

        Inputs:
            x0          -   current state
            tf          -   solution horizon (independent variable need not be time)
            A(t,x)      -   function returning SDC system matrix
            B(t,x,u)    -   function returning SDC control matrix
            C(t,x)      -   function returning SDC output matrix
            Q(t,x)      -   function returning LQR tracking error matrix
            R(t,x,u)    -   function returning LQR control weight matrix
            F(xf)       -   function returning LQR terminal weight matrix 
            z(t)        -   function returning the reference signal(s) at each value of the independent variable. 

        Optional Inputs:
            E(t,x)      -   function returning the stochastic term
            sigma       -   scalar value determining the weight on higher order moments in the cost function 
            max_iter    -   maximum number of iterations 
            tol         -   convergence tolerance in relative change in objective
            n_discretize-   number of time points at which to compute the optimal solution.

        Outputs:
            x     -   state vector at n_points
            u     -   control history at n_points
            K     -   time-varying feedback gains  
    
    """


    # Setup a "pocket" algorithm to store the best result
    pocket = {'cost': 1e16}

    # Problem size
    n = x0.size

    if E is None:
        E = lambda t,x: np.zeros((n,1))

    t0 = 0
    t = np.linspace(t0, tf, n_discretize)
    tb = t[::-1]                            # For integrating backward in time

    converge = tol + 1
    print("Approximating Sequence of Riccati Equations")
    print("Max iterations: {}".format(max_iter))
    start_iter = 0
    if guess is not None:
        start_iter = 1
        assert np.allclose(t, guess['time']), "guess does not match the time grid"
        x = guess['state']
        u = guess['control']
        J = compute_cost(t, x, u, C, Q, R, F, z)
        if np.isnan([J])[0]:
            print("Cost could not be evaluated on initial point")
            return

    for iter in range(start_iter, max_iter):
        print("Current iteration: {}".format(iter))

        if not iter:  # LTI iteration

            # Riccati equation for feedback solution
            Pf = dot(C(t0, x0).T,dot(F(x0),C(t0, x0)))
            Pv = odeint(dP, Pf, tb, args=(A, B, C, Q, R, lambda t: x0, lambda t: np.zeros((m,1)),E,sigma))
            Pi = interp1d(tb, Pv, kind=interp_type, fill_value=(Pv[0],Pv[-1]), axis=0, bounds_error=False)

            # Feedforward solution
            sf = dot(C(t0, x0).T,dot(F(x0),z(tf))).T.squeeze()
            s = odeint(ds, sf, tb, args=(A, B, C, Q, R, lambda t: x0, lambda t: np.zeros((m,1)), Pi, z, E, sigma))
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
            Pf = dot(C(tf, xf).T,dot(F(xf),C(tf, xf)))
            Pv = odeint(dP, Pf, tb, args=(A, B, C, Q, R, xi, ui, E, sigma))
            Pi = interp1d(tb, Pv, kind=interp_type, fill_value=(Pv[0],Pv[-1]), axis=0, bounds_error=False)

            # Feedforward solution
            sf = dot(C(tf, xf).T,dot(F(xf),z(tf))).T.squeeze()
            s = odeint(ds, sf, tb, args=(A, B, C, Q, R, xi, ui, Pi, z, E, sigma))
            si = interp1d(tb, s, kind=interp_type, fill_value=(s[0],s[-1]), axis=0, bounds_error=False)

            # Compute new state trajectory and control
            xold = x 
            x = odeint(dynamics, x0, t, args=(A, B, R, Pi, ui, si))
            u = compute_control(B, R, Pv, x, u, s, n, t)
            J = compute_cost(t, x, u, C, Q, R, F, z)
            converge = np.max(np.abs(np.array(x)-xold))
            if np.isnan([J])[0]:
                print("Cost could not be evaluated on current trajectory")
                break
            # converge = np.abs(J-Jold)/J

        print("Current cost: {}".format(J))
        if J < pocket['cost']:
            pocket['cost'] = J
            pocket['state'] = x
            pocket['control'] = u
            # Reshape Pv and output
            pocket['feedback'] = np.array([sdre_feedback(B(tc,xc,uc),R(tc,xc,uc),p) for tc,xc,uc,p in zip(t,x,u,Pv[::-1])])

        if converge <= tol:
            print("Convergence achieved. ")
            return x, u, np.array([sdre_feedback(B(tc,xc,uc),R(tc,xc,uc),p) for tc,xc,uc,p in zip(t,x,u,Pv[::-1])])
    print("Best cost found {}".format(pocket['cost']))
    return pocket['state'], pocket['control'], pocket['feedback']


def asre_feedback(t, x, u, B, R, P):

    return [dot(matrix_solve(R(ti, xi, ui), B(ti, xi, ui).T), Pi) for ti, xi, ui, Pi in zip(t, x, u, P)] 


def asre_integrateV(t, Vf, A, B, K, x, u):
    V = [Vf]
    dt = np.diff(t)
    for ti, xi, ui, k, dti in zip(t[::-1], x[::-1], u[::-1], K[::-1], dt[::-1]):
        a = A(ti, xi)
        b = B(ti, xi, ui)
        V.append(odeint(V_dynamics, V[-1], [0, dti], args=(a, b, k,))[-1])

    return np.array(V)


def asre_Pdynamics(P, t, V, B, R):
    return -V.T.dot(B).dot(np.linalg.solve(R, B.T.dot(V)))


def asre_integrateP(t, V, B, R, x, u):
    dt = np.diff(t)
    n = V[0].shape[1]
    P = [np.zeros((n, n))]
    for ti, xi, ui, Vi, dti in list(zip(t, x, u, V, dt))[::-1]:
        P.append(odeint(asre_Pdynamics, P[-1], [0, dti], args=(Vi, B(ti, xi, ui), R(ti, xi, ui)))[-1]) 
    return np.array(P)


def asrec_dynamics(x, t, A, B, R, K, P, V, z):
    u = asrec_control(x, A, B, R, K, P, V, z.squeeze())
    return A.dot(x) + B.dot(u)


def asrec_control(x, A, B, R, K, P, V, z):
    rb = np.linalg.solve(R, B.T)

    try:
        u = -(K - rb.dot(V).dot(np.linalg.solve(P, V.T))).dot(x) - rb.dot(V).dot(np.linalg.solve(P, z))
    except np.linalg.LinAlgError:
        u = -(K - rb.dot(V).dot(np.linalg.lstsq(P, V.T)[0])).dot(x) - rb.dot(V).dot(np.linalg.lstsq(P, z)[0])

    return u


def compute_control(B,R,Pv,X,U,S,n,T):
    u_new = []
    for x, u, s, pv, t in zip(X,U,S[::-1],Pv[::-1],T):
        u_new.append( -dot(matrix_solve(R(t,x,u),B(t,x,u).T), (dot(pv,x)-s)) )

    return np.array(u_new)


def asrec_cost(t, x, u, Q, R, F):

    integrand = np.array([dot(xi,dot(Q(ti,xi),xi)) + dot(ui,dot(R(ti,xi,ui),ui)) for ti,xi,ui in zip(t,x,u)]).flatten()
    J0 = 0.5*dot(x[-1],dot(F(x[-1]),x[-1]))
    return J0.squeeze() + trapz(integrand, t)


def compute_cost(t, x, u, C, Q, R, F, z): # This is for ASRE 
    e = np.array([z(ti).squeeze() - dot(C(ti,xi),xi) for ti,xi in zip(t,x)]).squeeze()

    integrand = np.array([dot(ei,dot(Q(ti,xi),ei)) + dot(ui,dot(R(ti,xi,ui),ui)) for ti,xi,ui,ei in zip(t,x,u,e)]).flatten()
    J0 = 0.5*dot(e[-1],dot(F(x[-1]),e[-1]))
    return J0.squeeze() + trapz(integrand, t)


def dP(p, t, A, B, C, Q, R, X, U, E, sigma):
    """ Riccati equation """
    
    x = X(t)
    u = U(t)
    a = A(t, x)
    b = B(t, x, u)
    c = C(t, x)
    e = E(t,x)
    q = Q(t, x)
    r = R(t, x, u)
    s = dot(b, matrix_solve(r,b.T))
    # 
    return (-dot(c.T,dot(q,c)) - dot(p,a) - dot(a.T,p) + dot(p,dot(s,p))) - sigma*p.dot(e).dot(e.T).dot(p) # could try np.multi_dot here 


def ds(s, t, A, B, C, Q, R, X, U, P, z, E, sigma):
    """ Differential equation of the feedfoward term. """
    p = P(t)
    x = X(t)
    u = U(t)
    a = A(t, x)
    b = B(t, x, u)
    c = C(t, x)
    e = E(t, x)
    r = R(t, x, u)
    q = Q(t, x)
    S = dot(b,matrix_solve(r,b.T))
    return -dot((a - dot(S,p)).T,s) - dot(c.T,dot(q,z(t))).T.squeeze() - sigma*p.dot(e).dot(e.T).dot(s)


def dynamics(x, t, A, B, R, P, U, s):
    """ Used in ASRE """
    p = P(t)
    u = U(t)
    a = A(t, x)
    b = B(t, x, u)
    r = R(t, x, u)
    S = dot(b, matrix_solve(r, b.T))
    return dot(a - dot(S, p), x) + dot(S, s(t))


def sdre_feedback(b,r,p):
    return dot(matrix_solve(r,b.T),p)


def V_dynamics(V,T,A,B,K):
    Vdot = (A - B.dot(K)).T.dot(V)
    return Vdot

# ################################################################################################
#                                         Test Functions                                         #
# ################################################################################################

def replace_nan(x, replace=1.):
    """ A useful method for use in SDC factorizations. """
    if np.isfinite(x):
        return x
    else:
        return replace


# ############## SRP (TIME) ##############
def SRP_A(t, x, bounds=(40,70)):
    Z = np.zeros((3,3))
    I = np.eye(3)

    V = np.array(x[3:6], ndmin=2)
    v = np.linalg.norm(V)**2

    At = x[6:]
    at = np.linalg.norm(At)
    g = np.array([0,0,-3.71], ndmin=2)

    A = [np.hstack([Z, I, Z]),
         np.hstack([Z, g.T.dot(V)/v,  I*np.clip(at, *bounds)/np.clip(at, 0.01, bounds[1])]),
         np.hstack([Z, Z, Z])]

    return np.concatenate(A, axis=0)

def SRP_Bu(t, x, u):
    return np.concatenate((np.zeros((6,3)),np.eye(3)), axis=0)

# def SRP_C(t, x):
#     return np.eye(6)

def SRP_Q(t,x):
    Q = np.zeros((9,9))
    Q[6,6] = 1
    Q[7,7] = 1
    Q[8,8] = 1
    return 1e-4*Q/np.linalg.norm(x[6:])

def SRP_C():
    C = np.eye(9)
    return C[:-3]


def SRP_guess(x0,N):
    x = np.array([np.linspace(xi, 1, N) for xi in x0[:6]]).T
    a = -np.ones((N,3))*x[:,3:6]/np.linalg.norm(x[:,3:6], axis=1)[:,None]
    u = np.zeros((N,3))
    guess = {}
    guess['state'] = np.concatenate((x,a), axis=1)
    guess['control'] = u 
    return guess

def SRP(N=500):
    import sys 
    from scipy.integrate import cumtrapz
    import time
    sys.path.append("./EntryGuidance")
    from TrajPlot import TrajPlot as traj3d

    bounds = (0, 700) # Control magnitude limit 
    m0 = 8500.
    x0 = np.array([-3200., 400, 2600, 625., -60, -270.])
    a0 = -70*x0[3:6]/np.linalg.norm(x0[3:6])
    x0 = np.concatenate((x0,a0),axis=0)
    tf = 15
    r = np.ones((6,))
    R = lambda t,x,u: np.eye(3)*10  # This is the penalty on acceleration rate
    S = np.zeros((9,9))

    guess = SRP_guess(x0,N)
    guess['time'] = np.linspace(0,tf,N)

    from functools import partial
    solvers = [
            #    partial(ASREC, t=np.linspace(0,tf,N), A=partial(SRP_A, bounds=bounds), B=SRP_Bu, C=SRP_C(), Q=SRP_Q, R=R, F=S, z=r, tol=1e-2, m=3, max_iter=3, guess=guess),
               partial(ASRE, tf=tf, A=partial(SRP_A, bounds=bounds), B=SRP_Bu, C=lambda t,x: SRP_C(), Q=lambda t,x: np.eye(6)*10, R=R, F=lambda xf: np.eye(6)*10, z=lambda t: 0*r, tol=1e-2, m=3, max_iter=3, guess=None, n_discretize=N)
              ]
    labels = ['ASREC','ASRE']

    for solver,label in zip(solvers,labels):
        t0 = time.time()
        x,u,K = solver(x0)
        print("{} solution time: {} s".format(label,time.time()-t0))

        t = np.linspace(0,tf,x.shape[0])
        Tmag = np.linalg.norm(x[:-1,6:],axis=1)
        T = np.clip(Tmag, *bounds)
        m = m0*np.exp(-cumtrapz(T/(9.81*290),t[:-1],initial=0))
        print("Prop used: {} kg".format(m0-m[-1]))


        plt.figure(6)
        plt.plot(t,x[:,0:3])
        plt.xlabel('Time (s)')
        plt.ylabel('Positions (m)')

        plt.figure(1)
        plt.plot(t,x[:,3:6])
        plt.xlabel('Time (s)')
        plt.ylabel('Velocities (m/s)')


        plt.figure(3)
        plt.plot(np.linalg.norm(x[:,3:5],axis=1),x[:,5])
        plt.xlabel('Horizontal Velocity (m/s)')
        plt.ylabel('Vertical Velocity (m/s)')

        plt.figure(2)
        plt.plot(t[:-1],x[:-1,6]/Tmag,label='x - {}'.format(label))
        plt.plot(t[:-1],x[:-1,7]/Tmag,label='y - {}'.format(label))
        plt.plot(t[:-1],x[:-1,8]/Tmag,label='z - {}'.format(label))
        plt.xlabel('Time (s)')
        plt.title('Control Direction')
        plt.legend()

        plt.figure(5)
        plt.plot(t[:-1],T)
        plt.xlabel('Time')
        plt.title('Thrust accel ')

        plt.figure(4)
        plt.plot(t[:-1],m)
        plt.xlabel('Time')
        plt.title('Mass')

        dirs = x[:-1,6:9]/Tmag[:,None] 
        traj3d(*(x[:-1,0:3].T), T=300*dirs*np.tile(T,(3,1)).T, figNum=7,label=label)

        # plt.figure(8)
        # for k in range(3):
            # for j in range(3):
                # plt.plot(t, K[:,j,k],label='K[{},{}]'.format(j,k))
    plt.show()
    # return t,x,u

# ############## Inverted Pendulum ##############
def IP_A(t, x):
    return np.array([[0,1],[4*4.905*replace_nan(sin(x[0])/x[0],1), -0.4]])

def IP_B(t, x, u):
    return np.array([[0],[10]])

def IP_z(t):
    return np.array([[sin(t)+cos(2*t-1)]])

def IP_R(t, x, u):
    # return np.array([[1 + 200*np.exp(-t)]])
    return np.array([[1]])

def IP_E(t, x):
    return np.diag([0., 1])

def test_IP():
    import time
    R = np.array([10])
    R.shape = (1,1)
    C = np.array([[1,0]])
    x0 = np.zeros((2)) + 1
    Q = np.array([[1.0e3]])
    F = np.array([[1.0e1]])
    tf = 15

    t_init = time.time()
    x,u,K = ASRE(x0, tf, IP_A, IP_B, lambda t,x: C, lambda t,x: Q, IP_R, lambda x: F, IP_z, E=IP_E, sigma=0, m=1,  max_iter=50, tol=0.1)      # Time-varying R
    t_asre = -t_init + time.time()

    t = np.linspace(0,tf,u.size)
    plt.figure(1)
    plt.plot(t,x[:,0],label='ASRE')
    plt.plot(t,IP_z(t).flatten(),'k--',label='Reference')
    plt.figure(2)
    plt.plot(t[:-1],u[:-1],label='ASRE')

    Kplot = np.transpose(K,(1,2,0))
    plt.figure(3)
    for gain in product(range(K.shape[1]),range(K.shape[2])):
        plt.plot(t[:-1],Kplot[gain][:-1], label='ASRE {}'.format(gain))

    plt.legend()
    print("ASRE: {} s".format(t_asre))

    plt.show()

def cliff():
    from ctrb import ctrb 

    def A(t,x):
        return np.array([[0, 0, 1, 0],[0,0,0,1], [0]*4, [0]*4])
    def B(t,x,u):
        return np.concatenate((np.zeros((2,2)), np.eye(2)), axis=0)
    def C(t,x):
        return np.eye(4)
    def E(t,x):
        return np.diag([1, 10, 0, 0])*0.0 # Kinda of like the standard deviation matrix 
    def Q(t,x):
        q = np.zeros((4,4))
        # q[1,1] = 0.1/(x[1]**12 + 1e-6)
        return q
    def R(t,x,u):
        return np.diag([1, 0.01])
    def F(xf):
        return np.diag([100, 100, 10, 10])
    def z(t):
        return np.array([10, 0, 0, 0])

    # x0 = np.zeros((4,))
    x0 = np.array([0., 0.1, 0, 0])

    print("System is controllable? {}".format(ctrb(A(0,x0), B(0,x0,0))))

    x,u,K = ASRE(x0, 3, A, B, C, Q, R, F, z, 2, E, 0)

if __name__ == '__main__':
    # cliff()
    test_IP() # Tests ASRE
    # SRP()     # Tests ASREC 
