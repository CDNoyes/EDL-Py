""" Riccati equation based nonlinear control methods """

from numpy import sin, cos, tan, dot, arccos
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
def SDRE(x, tf, A, B, C, Q, R, z, m, E=None, sigma=0, n_points=200, h=0):
    """
        Inputs:
            x       -   current state
            tf      -   solution horizon (independent variable need not be time, must be increasing)
            A(t,x)  -   function returning SDC system matrix
            B(t,x)  -   function returning SDC control matrix
            C(t,x)  -   function returning SDC output matrix
            Q(t,x)  -   function returning LQR tracking error matrix
            R(t,x)  -   function returning LQR control weight matrix
            z(t+h)  -   function returning the reference signal(s) at each value of the independent variable. Single step version can take a single value instead of function.

        Outputs:
            x     -   state vector at n_points
            u     -   control history at n_points
            K     -   time-varying feedback gains
    """
    from scipy.linalg import solve_continuous_are as care

    T = np.linspace(0, tf, n_points)
    dt = T[1]
    X = [x]
    U = [np.zeros((m,1))]
    K = []
    n = len(x)

    for iter, t in zip(range(n_points-1), T):
        if not (iter)%np.ceil(n_points/10.):
            print("Step {}".format(iter))

        # Get current system approximation
        a = A(t, x)
        b = B(t, x, U[-1])
        c = C(t, x)
        q = Q(t, x)
        r = R(t, x, U[-1])
        S = dot(b, matrix_solve(r, b.T))
        qc = dot(c.T, dot(q, c))

        # Solve the CARE:
        if E is None:
            p = care(a, b, qc, r)
        else:
            e = E(t,x)
            vals, vecs = np.linalg.eigh(S - sigma*e@e.T)
            assert np.all(vals >= 0), "Must be positive semidefinite {}".format(vals)
            N = vecs @ np.diag((vals**0.5)) @ vecs.T
            p = care(a, N, qc, np.eye(n))

        # Save the feedback gains
        K.append(sdre_feedback(b, r, p))

        # Solve the feedforward control:
        s = -matrix_solve((a-dot(S, p)).T, dot(c.T, dot(q, z(t+h))))  # Can introduce a problem-dependent offset here as anticipatory control to reduce lag
        u = sdre_control(x, b, r, p, s)
        U.append(u)

        # Step forward
        x = odeint(sdre_dynamics, x, np.linspace(t,t+dt,3), args=(a, b, r, p, s))[-1]
        X.append(x)

    J = sdre_cost(T, X, U, C, Q, R, z)
    print("Cost: {}".format(J))

    return np.array(X), np.array(U), np.array(K)


def SDREC(x, tf, A, B, C, Q, R, Sf, z, m, n_points=100, minU=None, maxU=None):
    """ Version of SDRE from the Lunar Lander thesis with terminal constraints
        Assumes Q,R are functions of the state but may be constant matrices
        Sf is the final state weighting for non constrained states
        z is the vector of final constraints such that Cx=z
    """
    n_points -= 1

    T = np.linspace(0, tf, n_points)
    dt = tf/(n_points-1.0)
    X = [x]
    U = []
    K = []

    for iter, t in zip(range(n_points), T):
        if not (iter)%np.ceil(n_points/10.):
            print("Step {}".format(iter))

        a = A(t, x)
        if iter:
            b = B(t, x, U[-1])
            r = R(t, x, U[-1])

        else:
            b = B(t, x, np.zeros((m,1)))
            r = R(t, x, np.zeros((m,1)))
        c = C(t, x)
        q = Q(t, x)

        S = DRE(t, tf, Sf, a, b, c, q, r, n_points-iter) 
        Kall = [sdre_feedback(b, r, s) for s in S]
        K.append(Kall[-1])
        V = integrateV(dt, c.T, a, b, Kall)
        P = integrateP(dt, b, r, V)
        
        u = -(Kall[-1] - np.linalg.solve(r, b.T.dot(V[-1])).dot(np.linalg.solve(P[-1], V[-1].T))).dot(x) - np.linalg.solve(r, b.T.dot(V[-1])).dot(np.linalg.solve(P[-1], z)).squeeze()

        if maxU is not None and np.linalg.norm(u) > maxU:
            u *= maxU/np.linalg.norm(u)
        if minU is not None and np.linalg.norm(u) < minU:
            u *= minU/np.linalg.norm(u)

        U.append(u)

        bu = B(t, x, u)
        x = step(x, dt, u, a, bu)
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


def riccati_dynamics(S,T,A,B,C,Q,R):
    n = int(np.sqrt(S.size))
    S.shape = (n,n) # Turn into a matrix
    dS = A.T.dot(S) + S.dot(A) - S.dot(B).dot(np.linalg.solve(R,B.T.dot(S))) + Q # + C.T.dot(Q).dot(C)

    return dS.flatten()

def DRE(tcurrent, tfinal, Sf, A, B, C, Q, R, nRemaining):
    x0 = Sf.flatten()
    n = int(x0.size**0.5)
    Svec = odeint(riccati_dynamics, x0, np.linspace(tcurrent, tfinal, nRemaining), args=(A,B,C,Q,R))

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



def sdre_step(x, t, A, B, C, Q, R, z, h=0, args=()):
    from scipy.linalg import solve_continuous_are as care

    a = A(t,x)
    n = a.shape[0]
    b = B(t,x)
    c = C(t,x)
    q = Q(t,x)
    r = R(t,x)
    
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
    e = z(t).flatten() - np.array([[dot(C(ti,xi),xi)] for ti,xi in zip(t,x)]).flatten()
    integrand = np.array([dot(ei, dot(Q(ti, xi), ei)) + dot(ui, dot(R(ti, xi, ui), ui)) for ti,xi,ui,ei in zip(t,x,u,e)]).flatten()
    return trapz(integrand, t)

def sdre_feedback(b,r,p):
    return dot(matrix_solve(r, b.T), p)



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
def SRP_A(t,x):
    return np.array([[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,-3.71*replace_nan(1/x[2],1),0,0,0]])

def SRP_B(t,x):
    return np.concatenate((np.zeros((3,3)),np.eye(3)),axis=0)

def SRP_Bu(t,x):
    return np.concatenate((np.zeros((3,3)),np.eye(3)),axis=0)

def SRP_C(t,x):
    return np.eye(6)

def SRP(N=200):
    import sys 
    from scipy.integrate import cumtrapz
    import time
    sys.path.append("./EntryGuidance")
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
               partial(SDREC, tf=tf, A=SRP_A, B=SRP_B, C=SRP_C, Q=Q, R=R, Sf=S, z=r, n_points=N, maxU=70, minU=40),
               partial(SDREC, tf=tf, A=SRP_A, B=SRP_B, C=SRP_C, Q=Q, R=R, Sf=S, z=r, n_points=N, maxU=None, minU=None),
              ]
    labels = ['SDRE', 'SDRE (No Control Limits)']

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

        plt.figure(8)
        for k in range(3):
            for j in range(3):
                plt.plot(t, K[:,j,k],label='K[{},{}]'.format(j,k))
    plt.show()
    return t,x,u

 

# ############## Inverted Pendulum ##############
def IP_A(t,x):
    return np.array([[0,1],[4*4.905*replace_nan(sin(x[0])/x[0],1), -0.4]])

def IP_B(t, x, u):
    return np.array([[0],[10]])

def IP_z(t):
    return np.array([[sin(t)+cos(2*t-1)]])

def IP_R(t, x, u):
    return np.array([[1 + 1*np.exp(-t)]])

def IP_E(t, x):
    return np.diag([0., 1])

def test_IP():
    import time
    C = np.array([[1,0]])
    x0 = np.zeros((2)) + 1
    Q = np.array([[1.0e3]])
    tf = 5

    # t_init = time.time()
    for SIGMA in [-0.1, 0, 0.1, 0.2,]:
        x,u,K = SDRE(x0, tf, IP_A, IP_B, lambda t,x: C, lambda t,x: Q, IP_R, IP_z, m=1, E=IP_E, sigma=SIGMA*50, n_points=175, h=0.11) 
    # t_sdre = -t_init + time.time()

    # print("SDRE: {} s".format(t_sdre))
    # print(u.size)

        t = np.linspace(0,tf,u.size)
        plt.figure(1)
        plt.plot(t,x[:,0],label='\sigma = {}'.format(SIGMA))
        plt.plot(t, IP_z(t).squeeze(), label="Reference")
        plt.legend()
        plt.title('Output history')
        plt.figure(2)
        plt.plot(t,u,label='\sigma = {}'.format(SIGMA))
        plt.title('Control history')
        plt.legend()

        Kplot = np.transpose(K,(1,2,0))
        plt.figure(3)
        for gain in product(range(K.shape[1]),range(K.shape[2])):
            plt.plot(t[:-1],Kplot[gain],label='SDRE {}, sigma={}'.format(gain, SIGMA))
        plt.title('Feedback gains')
        plt.legend()


    plt.show()


if __name__ == '__main__':
    test_IP()
    # SRP()
