""" Riccati equation based nonlinear control methods """

from numpy import sin, cos, tan, dot, arccos
import numpy as np
from scipy.linalg import solve as matrix_solve
from scipy.integrate import simps as trapz
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from functools import partial
from itertools import product

from Utils.SDRE import SDRE, SDREC 
from Utils.ASRE import ASRE, ASREC 

def controller(A, B, C, Q, R, z, method='SDRE',**kwargs):

    x = np.array([kwargs['current_state'][0],kwargs['velocity'], kwargs['fpa'], kwargs['lift'], kwargs['drag']])
    t = kwargs['velocity']
    u = np.clip(sdre_step(x, t, A, B, C, Q, R, z, h=0)[0],-1,1)

    return arccos(u)*np.sign(kwargs['bank'])


def replace_nan(x, replace=1.):
    """ A useful method for use in SDC factorizations. """
    if np.isnan(x) or np.isinf(x): # equal to not np.isfinite()
        return replace
    else:
        return x


# ############## SRP (TIME) ##############
def SRP_A(x):
    return np.array([[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
    # return np.array([[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,-3.71*replace_nan(1/x[2],1),0,0,0]])

def SRP_B(x):
    return np.concatenate((np.zeros((3,3)),np.eye(3)),axis=0)

def SRP_Bu(x,u):
    return np.concatenate((np.zeros((3,3)),np.eye(3)),axis=0)

def SRP_C(x):
    return np.eye(6)

def SRP(N=500):
    from scipy.integrate import cumtrapz
    import time
    from TrajPlot import TrajPlot as traj3d
    m0 = 8500.
    x0 = np.array([-3200., 400, 2600, 625., -60, -270.])
    tf = 15
    r = np.ones((6,))
    R = lambda x: np.eye(3)*0.001
    # R = lambda x: np.diag([replace_nan(1/np.abs(x[i]),1) for i in range(3)])
    Q = lambda x: np.zeros((6,6))
    S = np.zeros((6,6))

    from functools import partial
    solvers = [
            #    partial(SDREC, tf=tf, A=SRP_A, B=SRP_B, C=SRP_C, Q=Q, R=R, Sf=S, z=r, n_points=N, maxU=70, minU=40),
               partial(ASREC, t=np.linspace(0, tf, 100), A=SRP_A, B=SRP_Bu, C=np.eye(6), Q=Q, R=R, F=S, z=r, tol=1e-4)
              ]
    # labels = ['SDRE','ASRE']
    labels = ['ASRE']

    for solver,label in zip(solvers,labels):
        t0 = time.time()
        x,u,K = solver(x0)
        print("{} solution time: {} s".format(label,time.time()-t0))

        t = np.linspace(0,tf,x.shape[0])
        T = np.linalg.norm(u,axis=1)
        m = m0*np.exp(-cumtrapz(T/(9.81*290),t,initial=0))
        print("Prop used: {} kg".format(m0-m[-1]))


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
    print("Prop used: {} kg".format(m0-m[-1]))

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

    print("ASRE: {} s".format(t_asre))
    print("SDRE: {} s".format(t_sdre))

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
