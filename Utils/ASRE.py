""" Implements two versions of Approximating Sequence of Riccati Equations nonlinear control method """

from numpy import sin, cos, tan, dot, arccos
import numpy as np
from scipy.linalg import solve as matrix_solve
from scipy.integrate import simps as trapz
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from functools import partial
from itertools import product

# ################################################################################################
#                           Approximating Sequence of Riccati Equations                          #
# ################################################################################################

# TODO: The reshaped matrices can be integrated directly using my RK4 scheme instead of scipy.odeint, not sure which is faster 
# TODO: Implement and study the stochastic versions which simply requires adding stochastic terms to the Riccati equations
# TODO: Test state constraints 


def asre_feedback(t, x, u, B, R, Pv, n):

    P = [Pi.reshape((n,n)) for Pi in Pv]
    return [dot(matrix_solve(R(ti, xi, ui), B(ti, xi, ui).T), Pi) for ti,xi,ui,Pi in zip(t,x,u,P)] 


def asre_integrateV(t, Vf, A, B, K, x, u):
    Vshape = Vf.shape
    V = [Vf.flatten()]
    dt = np.diff(t)
    for ti,xi,ui,k,dti in zip(t[::-1], x[::-1], u[::-1], K[::-1], dt[::-1]):
        a = A(ti, xi)
        b = B(ti, xi, ui)
        V.append(odeint(V_dynamics, V[-1], [0,dti], args=(a,b,k,Vshape))[-1])

    return np.array([v.reshape(Vshape) for v in V])


def asre_Pdynamics(P, t, V, B, R, n):
    P.shape = (n,n)
    return -V.T.dot(B).dot(np.linalg.solve(R,B.T.dot(V))).flatten()


def asre_integrateP(t, V, B, R, x, u, n):
    dt = np.diff(t)
    P = [np.zeros((n,n)).flatten()]
    for ti,xi,ui,Vi,dti in list(zip(t,x,u,V,dt))[::-1]:
        P.append(odeint(asre_Pdynamics, P[-1], [0,dti], args=(Vi,B(ti,xi,ui),R(ti,xi,ui),n))[-1]) 
    return np.array([p.reshape((n,n)) for p in P])


def asrec_dynamics(x,t,A,B,R,K,P,V,z, ubounds=None):
    u = asrec_control(x,A,B,R,K,P,V,z, ubounds[0],ubounds[1])
    return A.dot(x) + B.dot(u)


def asrec_control(x,A,B,R,K,P,V,z,ul=None,ub=None):
    rb = np.linalg.solve(R,B.T)

    try:
        u = -(K - rb.dot(V).dot(np.linalg.solve(P,V.T))).dot(x) - rb.dot(V).dot(np.linalg.solve(P,z))
    except np.linalg.LinAlgError:
        u = -(K - rb.dot(V).dot(np.linalg.lstsq(P,V.T)[0])).dot(x) - rb.dot(V).dot(np.linalg.lstsq(P,z)[0])

    if ub is not None and np.linalg.norm(u) > ub:
        u *= ub/np.linalg.norm(u)
    if ul is not None and np.linalg.norm(u) < ul:
        u *= ul/np.linalg.norm(u)

    return u


def asrec_cost(t, x, u, Q, R, F):
    integrand = 0.5*np.array([dot(xi.T,dot(Q(ti,xi),xi)) + dot(ui, dot(R(ti,xi,ui), ui.T)) for ti,xi,ui in zip(t,x,u)]).flatten() 
    J0 = 0.5*dot(x[-1].T,dot(F,x[-1]))
    return J0 + trapz(integrand, t)


def ASREC(x0, t, A, B, C, Q, R, F, z, m, max_iter=50, tol=0.01, maxU=None, minU=None, guess=None):
    """ Approximating Sequence of Riccati Equations with Terminal Constraints Cx=z """
    # Problem size
    n = x0.size
    # m = R(x0).shape[0]

    n_discretize = len(t)
    dt = np.diff(t)
    tb = t[::-1]                            # For integrating backward in time

    converge = tol + 1
    Jold = -1e16
    print("Approximating Sequence of Riccati Equations")
    start_iter = 0
    if guess is not None:
        start_iter = 1
        x = guess['state']
        u = guess['control']
        
    for iter in range(start_iter, max_iter):
        print("Current iteration: {}".format(iter+1))
        
        if not iter: # LTI iteration
            u = [np.zeros((m))]*n_discretize
            x = [x0]*n_discretize  # This is the standard approach, but it seems like it would be far superior to actually integrate the system using the initial control

        # Riccati equation for feedback solution
        Pf = F.flatten()
        Pv = odeint(dP, Pf, tb, args=(A, B, lambda t,x: np.eye(n), Q, R, lambda t: x0, lambda t: np.zeros((m,1))))[::-1]
        K = asre_feedback(t, x, u, B, R, Pv, n)
        V = asre_integrateV(t, C.T, A, B, K, x, u)[::-1]
        P = asre_integrateP(t, V, B, R, x, u, n)[::-1]

        # Compute new state trajectory and control
        x = [x0]
        u = [u[0].T]
        for stage in range(n_discretize-1):
            x.append(odeint(asrec_dynamics, x[-1], [0, dt[stage]], args=(A(t[stage], x[-1]), B(t[stage],x[-1],u[-1]), R(t[stage],x[-1],u[-1]), K[stage], P[stage], V[stage], z, (minU,maxU)))[-1])
            u.append(asrec_control(x[-2], A(t[stage], x[-2]), B(t[stage],x[-2],u[-1]), R(t[stage],x[-2],u[-1]), K[stage], P[stage], V[stage], z, minU,maxU).T) 

        J = asrec_cost(t, x, u, Q, R, F)
        converge = np.abs(J-Jold)/J
        Jold = J
        u = u[1:]
        u.append(np.zeros((m)))
        print("Current cost: {}".format(J))

        if converge <= tol:
            print("Convergence achieved. ")
            break
    u[-1] = u[-2]
    return np.array(x), np.array(u), np.array(K)



def ASRE(x0, tf, A, B, C, Q, R, F, z, m, max_iter=10, tol=0.01, n_discretize=250, guess=None):
    """ Approximating Sequence of Riccati Equations """
    from scipy.interpolate import interp1d
    interp_type = 'cubic'

    # Setup a pocket type algorithm to store the best result
    pocket = {'cost': 1e16}

    # Problem size
    n = x0.size
    # m = R(tf, x0).shape[0]

    t0 = 0
    t = np.linspace(t0, tf, n_discretize)
    tb = t[::-1]                            # For integrating backward in time

    converge = tol + 1
    print("Approximating Sequence of Riccati Equations")
    print("Max iterations: {}".format(max_iter))
    start_iter = 0
    if guess is not None:
        start_iter = 1
        timeGuess = guess['time']
        stateGuess = guess['state']
        controlGuess = guess['control']
        x = np.interp(t, timeGuess, stateGuess)
        u = np.interp(t, timeGuess, controlGuess)

    for iter in range(start_iter, max_iter):
        print("Current iteration: {}".format(iter+1))

        if not iter:  # LTI iteration

            # Riccati equation for feedback solution
            Pf = dot(C(t0, x0).T,dot(F(x0),C(t0, x0))).flatten()
            Pv = odeint(dP, Pf, tb, args=(A, B, C, Q, R, lambda t: x0, lambda t: np.zeros((m,1))))
            Pi = interp1d(tb, Pv, kind=interp_type, fill_value=(Pv[0],Pv[-1]), axis=0, bounds_error=False)

            # Feedforward solution
            sf = dot(C(t0, x0).T,dot(F(x0),z(tf))).T.squeeze()
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
            Pf = dot(C(tf, xf).T,dot(F(xf),C(tf, xf))).flatten()
            Pv = odeint(dP, Pf, tb, args=(A, B, C, Q, R, xi, ui))
            Pi = interp1d(tb, Pv, kind=interp_type, fill_value=(Pv[0],Pv[-1]), axis=0, bounds_error=False)

            # Feedforward solution
            sf = dot(C(tf, xf).T,dot(F(xf),z(tf))).T.squeeze()
            s = odeint(ds, sf, tb, args=(A, B, C, Q, R, xi, ui, Pi, z))
            si = interp1d(tb, s, kind=interp_type, fill_value=(s[0],s[-1]), axis=0, bounds_error=False)

            # Compute new state trajectory and control
            # xold = np.copy(x)
            Jold = J
            x = odeint(dynamics, x0, t, args=(A, B, R, Pi, ui, si))
            u = compute_control(B, R, Pv, x, u, s, n, t)
            J = compute_cost(t, x, u, C, Q, R, F, z)
            converge = np.abs(J-Jold)/J

        print("Current cost: {}".format(J))
        if J < pocket['cost']:
            pocket['cost'] = J
            pocket['state'] = x
            pocket['control'] = u
            # Reshape Pv and output
            pocket['feedback'] = np.array([sdre_feedback(B(tc,xc,uc),R(tc,xc,uc),np.reshape(p,(n,n))) for tc,xc,uc,p in zip(t,x,u,Pv[::-1])])

        if converge <= tol:
            print("Convergence achieved. ")
            return x, u, np.array([sdre_feedback(B(tc,xc,uc),R(tc,xc,uc),np.reshape(p,(n,n))) for tc,xc,uc,p in zip(t,x,u,Pv[::-1])])
    print("Best cost found {}".format(pocket['cost']))
    return pocket['state'], pocket['control'], pocket['feedback']


def compute_control(B,R,Pv,X,U,S,n,T):
    u_new = []
    for x, u, s, pv, t in zip(X,U,S[::-1],Pv[::-1],T):
        pv.shape = (n,n)
        u_new.append( -dot(matrix_solve(R(t,x,u),B(t,x,u).T), (dot(pv,x)-s)) )

    return np.array(u_new)


def compute_cost(t, x, u, C, Q, R, F, z):
    e = z(t).flatten() - np.array([[dot(C(ti,xi),xi)] for ti,xi in zip(t,x)]).flatten()
    integrand = np.array([dot(ei,dot(Q(ti,xi),ei)) + dot(ui,dot(R(ti,xi,ui),ui)) for ti,xi,ui,ei in zip(t,x,u,e)]).flatten()
    J0 = 0.5*dot(e[-1],dot(F(x[-1]),e[-1]))
    return J0.squeeze() + trapz(integrand, t)


def dP(p, t, A, B, C, Q, R, X, U):
    """ Riccati equation """
    n = int(np.sqrt(p.size))
    p.shape = (n,n)      # Turn into a matrix
    x = X(t)
    u = U(t)
    a = A(t, x)
    b = B(t, x, u)
    c = C(t, x)
    q = Q(t, x)
    r = R(t, x, u)
    s = dot(b,matrix_solve(r,b.T))

    return (-dot(c.T,dot(q,c)) - dot(p,a) - dot(a.T,p) + dot(p,dot(s,p))).flatten()


def ds(s, t, A, B, C, Q, R, X, U, P, z):
    """ Differential equation of the feedfoward term. """
    p = P(t)
    n = int(np.sqrt(p.size))
    p.shape = (n,n)      # Turn into a matrix
    x = X(t)
    u = U(t)
    a = A(t, x)
    b = B(t, x, u)
    c = C(t, x)
    r = R(t, x, u)
    q = Q(t, x)
    S = dot(b,matrix_solve(r,b.T))
    return -dot((a - dot(S,p)).T,s) - dot(c.T,dot(q,z(t))).T.squeeze()


def dynamics(x, t, A, B, R, P, U, s):
    """ Used in ASRE """
    p = P(t)
    n = int(np.sqrt(p.size))
    p.shape = (n,n)      # Turn into a matrix
    u = U(t)
    a = A(t, x)
    b = B(t, x, u)
    r = R(t, x, u)
    S = dot(b,matrix_solve(r,b.T))
    return dot(a - dot(S,p),x) + dot(S,s(t))


def sdre_feedback(b,r,p):
    return dot(matrix_solve(r,b.T),p)


def V_dynamics(V,T,A,B,K,Vshape):
    V.shape = Vshape
    Vdot = (A - B.dot(K)).T.dot(V)
    return Vdot.flatten()

# ################################################################################################
#                                         Test Functions                                         #
# ################################################################################################

def replace_nan(x, replace=1.):
    """ A useful method for use in SDC factorizations. """
    if np.isnan(x) or np.isinf(x):
        return replace
    else:
        return x


# ############## SRP (TIME) ##############
def SRP_A(t, x):
    return np.array([[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,-3.71*replace_nan(1/x[2],1),0,0,0]])

def SRP_Bu(t, x, u):
    return np.concatenate((np.zeros((3,3)),np.eye(3)),axis=0)

def SRP_C(t, x):
    return np.eye(6)

def SRP(N=20):
    import sys 
    from scipy.integrate import cumtrapz
    import time
    sys.path.append("./EntryGuidance")
    from TrajPlot import TrajPlot as traj3d

    m0 = 8500.
    x0 = np.array([-3200., 400, 2600, 625., -60, -270.])
    tf = 15
    r = np.zeros((6,))
    R = lambda t,x,u: np.eye(3)
    # R = lambda x: np.diag([replace_nan(1/np.abs(x[i]),1) for i in range(3)])
    Q = lambda t,x: np.zeros((6,6))
    S = np.zeros((6,6))

    from functools import partial
    solvers = [
               partial(ASREC, t=np.linspace(0,tf,50), A=SRP_A, B=SRP_Bu, C=np.eye(6), Q=Q, R=R, F=S, z=r, tol=1e-2, maxU=70, minU=40, m=3),
               partial(ASREC, t=np.linspace(0,tf,50), A=SRP_A, B=SRP_Bu, C=np.eye(6), Q=Q, R=R, F=S, z=r, tol=1e-2, maxU=None, minU=None, m=3)
              ]
    labels = ['ASRE', 'ASRE (No Control Limits)']

    for solver,label in zip(solvers,labels):
        t0 = time.time()
        x,u,K = solver(x0)
        print("{} solution time: {} s".format(label,time.time()-t0))

        t = np.linspace(0,tf,x.shape[0])
        T = np.linalg.norm(u,axis=1)
        m = m0*np.exp(-cumtrapz(T/(9.81*290),t,initial=0))
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

# ############## Inverted Pendulum ##############
def IP_A(t, x):
    return np.array([[0,1],[4*4.905*replace_nan(sin(x[0])/x[0],1), -0.4]])

def IP_B(t, x, u):
    return np.array([[0],[10]])

def IP_z(t):
    return np.array([[sin(t)+cos(2*t-1)]])

def IP_R(t, x, u):
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

    t_init = time.time()
    x,u,K = ASRE(x0, tf, IP_A, IP_B, lambda t,x: C, lambda t,x: Q, IP_R, lambda x: F, IP_z, m=1,  max_iter=50, tol=0.1)      # Time-varying R
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

   
    print("ASRE: {} s".format(t_asre))

    plt.show()


if __name__ == '__main__':
    # test_IP() # Tests ASRE
    SRP()     # Tests ASREC 
