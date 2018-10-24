"""SDC factorizations of powered descent phase dynamics """

import numpy as np 


def replace_nan(x, replace=1.):
    """ A useful method for use in SDC factorizations. """
    if np.all(np.isfinite(x)):
        return x
    else:
        x[np.logical_not(np.isfinite(x))] = replace 
        return x 


class Unconstrained:
    """ Uses acceleration as the control variable [ax, ay, az]
        no mass dynamics 
    """
    def __init__():
        self.n = 6 
        self.m = 3 

    def A(self, t, x):
        return np.array([[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,-3.71*replace_nan(1/x[2],1),0,0,0]])

    def B(self, t, x):
        return np.concatenate((np.zeros((3,3)),np.eye(3)),axis=0)

    def Bu(self, t, x, u):
        """ Non-affine variant """
        return np.concatenate((np.zeros((3,3)),np.eye(3)),axis=0)

    def C(self, t, x):
        return np.eye(6)


class Full:
    """
        x = [r,v,m]
        u = [Tx, Ty, Tz]

        q = 0.5*rho*V**2
        D = qS/m Cd

        Thrust as control var
        Mass dynamics included 
        Aerodynamics are included but can be neglected 
        by simply setting the coefficients to zero 

    """

    def __init__(self, Cd=1.4):
        self.n = 7 
        self.m = 3
        self.cd = Cd 

    def A(self, t, x):
        r = x[0:3]
        v = x[3:6]
        m = x[-1]

        V = np.linalg.norm(v)

        Z = np.zeros((3, 3))
        I = np.eye(4)[:-1]

        Ar = np.concatenate((Z, I), axis=1)

        Av = np.zeros((3, 7))
        Av[2, -1] = -3.71/m          # gravity term

        # Aerodynamic drag
        rho0 = 0.0158   # assume we're close enough to the ground
        # Cd = 1.4        # constant drag
        S = 15.8        # area, m^2
        Dv = -0.5*rho0*V*S/m * self.cd
        Dm = -0.5*rho0*V*S/m**2 * self.cd * v
        alpha = 0.5     # weighting between mass and velocity components of drag factorization
        Av_aero = np.concatenate((Z, np.eye(3)*Dv*alpha, Dm[:, None]*(1-alpha)), axis=1)

        Am = np.zeros((1, 7))

        return np.concatenate((Ar, Av+Av_aero, Am), axis=0)

    def B(self, t, x, u):

        ve = 300 * 9.81

        r = x[0:3]
        v = x[3:6]
        m = x[-1]

        T = np.linalg.norm(u)
        Br = np.zeros((3, 3))
        Bv = np.eye(3)/m 

        if T <= 1e-5:
            Bm = np.zeros((1, 3))
        else:
            Bm = -u.T/(ve*T)

        if Bm.ndim == 1:
            Bm = Bm[None, :]
        
        return np.concatenate((Br, Bv, Bm), axis=0)

    def C(self, t, x):  # trims off the mass state since we don't want to regulate it to zero, nor constrain it to a particular final value 
        return np.eye(7)[:-1]

    def guess(self, x0, tf, N):
        x = np.array([np.linspace(xi, 1, N) for xi in x0[:6]]).T

        m = np.linspace(x0[-1], x0[-1]*0.7, N)[:, None]

        a = -np.ones((N,3))*x[:, 3:6]/np.linalg.norm(x[:, 3:6], axis=1)[:, None]
        T = a*70*m

        guess = {}
        guess['time'] = np.linspace(0, tf, N)
        guess['state'] = np.concatenate((x, m), axis=1)
        guess['control'] = T
        return guess

class MassLoss:
    """
        x = [r,v,m,-dm]
        u = [Tx, Ty, Tz]

        q = 0.5*rho*V**2
        D = qS/m Cd

        Thrust as control var
        Mass dynamics included 
        Aerodynamics are included but can be neglected 
        by simply setting the coefficients to zero 

    """

    def __init__(self, Cd=1.4):
        self.n = 8 
        self.m = 3
        self.cd = Cd 

    def A(self, t, x):
        r = x[0:3]
        v = x[3:6]
        m = x[-2]
        L = x[-1]

        V = np.linalg.norm(v)

        Z = np.zeros((3, 3))
        I = np.eye(4)[:-1]

        Ar = np.concatenate((Z, I, np.zeros((3,1))), axis=1)

        Av = np.zeros((3, 8))
        Av[2, -1] = -3.71/m          # gravity term

        # Aerodynamic drag
        rho0 = 0.0158   # assume we're close enough to the ground
        # Cd = 1.4        # constant drag
        S = 15.8        # area, m^2
        Dv = -0.5*rho0*V*S/m * self.cd
        Dm = -0.5*rho0*V*S/m**2 * self.cd * v
        alpha = 0.5     # weighting between mass and velocity components of drag factorization
        Av_aero = np.concatenate((Z, np.eye(3)*Dv*alpha, Dm[:, None]*(1-alpha), np.zeros((3,1))), axis=1)

        Am = np.zeros((1, 8))
        AL = Am
        # AL[-1] = -0.5*np.linalg.norm() 
        return np.concatenate((Ar, Av+Av_aero, Am, AL), axis=0)

    def B(self, t, x, u):

        ve = 300 * 9.81

        r = x[0:3]
        v = x[3:6]
        m = x[-2]
        L = x[-1]

        T = np.linalg.norm(u)
        Br = np.zeros((3, 3))
        Bv = np.eye(3)/m 

        if T <= 1e-5:
            Bm = np.zeros((1, 3))
        else:
            Bm = -u.T/(ve*T)

        if Bm.ndim == 1:
            Bm = Bm[None, :]
        
        Bl = -Bm/L
        return np.concatenate((Br, Bv, Bm, Bl), axis=0)

    def C(self, t, x):  # trims off the mass states since we don't want to constrain it to a particular final value 
        return np.eye(8)[:-2]

    def guess(self, x0, tf, N):
        x = np.array([np.linspace(xi, 1, N) for xi in x0[:6]]).T

        m = np.linspace(x0[-1], x0[-1]*0.7, N)[:, None]
        s = np.sqrt(m[0] - m) + 1  

        a = -np.ones((N,3))*x[:, 3:6]/np.linalg.norm(x[:, 3:6], axis=1)[:, None]
        T = a*70*m

        guess = {}
        guess['time'] = np.linspace(0, tf, N)
        guess['state'] = np.concatenate((x, m, s), axis=1)
        guess['control'] = T
        return guess



class Constrained:
    """
        Accel rates are the control
        States are now [r, v, a_thrust]
        no mass dynamics 
    """
    def __init__(self):
        self.n = 9 
        self.m = 3


    def A(self, t, x, bounds=(40, 70)):
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

    def Bu(self, t, x, u):
        return np.concatenate((np.zeros((6,3)),np.eye(3)),axis=0)


    def Q(self, t, x):
        Q = np.zeros((9,9))
        Q[6,6] = 1
        Q[7,7] = 1
        Q[8,8] = 1
        return 1e-4*Q/np.linalg.norm(x[6:])

    def C():
        C = np.eye(9)
        return C[:-3]


    def guess(x0, N):
        x = np.array([np.linspace(xi, 1, N) for xi in x0[:6]]).T
        a = -np.ones((N,3))*x[:,3:6]/np.linalg.norm(x[:,3:6], axis=1)[:,None]
        u = np.zeros((N,3))
        guess = {}
        guess['state'] = np.concatenate((x,a), axis=1)
        guess['control'] = u 
        return guess


if __name__ == "__main__":
    import time 
    import pandas as pd 
    import sys 
    sys.path.append("./Utils")

    from PDPlot import Plot, Plots 
    from ASRE import ASRE, ASREC 
    from SDRE import SDREC 

    r0 = [-3200, 100, 2600]
    v0 = [600, 0, -265]
    m0 = [8500]

    x0 = np.array(r0+v0+m0)

    t0 = 0
    tf = 16
    N = 100
    t = np.linspace(t0, tf, N)
    df_list = []
    labels = []

    model = Full()

    Q = lambda t,x: np.diag([1,1,1,10,10,10])*1e-3
    R = lambda t,x,u: np.eye(3)*1e-8 
    F = lambda xf: np.diag([1,1,1,10,10,10])*1e-1
    z = lambda t: np.zeros((6, 1)) 

    print("Running unconstrained SRP landing ")
    x,u,K = ASRE(x0, tf, model.A, model.B, model.C, Q, R, F, z, model.m, max_iter=15, tol=1e-4, n_discretize=N, guess=model.guess(x0, tf, N))

    XU = np.concatenate((x,u), axis=1)

    df = pd.DataFrame(XU, index=t, columns=['x','y','z','vx','vy','vz','mass','Tx','Ty','Tz'])
    df_list.append(df)
    labels.append("ASRE")
    # Plot(df)


    Q = lambda t,x: np.diag([1,1,1,10,10,10,0])*0
    R = lambda t,x,u: np.eye(3)*1e-10
    F = np.diag([1,1,1,10,10,10,0])*0
    z = np.ones((6, 1)) 

    # print("Running endpoint constrained SRP landing ")
    t0 = time.time()
    x,u,K = ASREC(x0, t, model.A, model.B, model.C(0,0), Q, R, F, z, model.m, max_iter=15, tol=1, guess=model.guess(x0, tf, N))
    t1 = time.time()
    XU = np.concatenate((x,u), axis=1)

    df = pd.DataFrame(XU, index=t, columns=['x','y','z','vx','vy','vz','mass','Tx','Ty','Tz'])
    df_list.append(df)
    labels.append("ASREC")

    # t2 = time.time()
    # Q = lambda t,x: np.diag([0,0,0,0,0,0,0])
    # Qf = np.diag([0,0,0,0,0,0,0])

    # x,u,K = SDREC(x0, tf, model.A, model.B, model.C, Q, R, Qf, z, n_points=N, minU=None, maxU=None)
    # # x,u,K = SDREC(x0, tf, model.A, model.B, model.C, Q, R, Qf, z, n_points=N, minU=40*8500, maxU=70*8500)
  
    # t3 = time.time()
    # XU = np.concatenate((x,u), axis=1)

    # df = pd.DataFrame(XU, index=t, columns=['x','y','z','vx','vy','vz','mass','Tx','Ty','Tz'])
    # df_list.append(df)
    # labels.append("SDREC")

    model = MassLoss()
    Q = lambda t,x: np.diag([0,0,0,0,0,0,0,1])*1e-4
    Qf = np.diag([0,0,0,0,0,0,0,1])*1e-5
    y0 = np.append(x0, 1)
    t4 = time.time()
    # x,u,K = SDREC(y0, tf, model.A, model.B, model.C, Q, R, Qf, z, n_points=N, minU=None, maxU=None)
    # x,u,K = SDREC(y0, tf, model.A, model.B, model.C, Q, R, Qf, z, n_points=N, minU=40*8500, maxU=70*8500)
    x,u,K = ASREC(y0, t, model.A, model.B, model.C(0,0), Q, R, Qf, z, model.m, max_iter=15, tol=1, guess=model.guess(x0, tf, N))

    t5 = time.time()
  
    XU = np.concatenate((x[:,:7],u), axis=1)

    df = pd.DataFrame(XU, index=t, columns=['x','y','z','vx','vy','vz','mass','Tx','Ty','Tz'])
    df_list.append(df)
    labels.append("with mass loss")




    # print("Running endpoint constrained SRP landing (neglecting drag)")
    # model = Full(Cd=0)
    # x,u,K = ASREC(x0, t, model.A, model.B, model.C(0,0), Q, R, F, z, model.m, max_iter=15, tol=1e-5, guess=model.guess(x0, tf, N))

    # XU = np.concatenate((x,u), axis=1)

    # df = pd.DataFrame(XU, index=t, columns=['x','y','z','vx','vy','vz','mass','Tx','Ty','Tz'])
    # df_list.append(df)
    # labels.append("ASREC (No Drag)")

    # print("ASREC: {:.1f} s".format(t1-t0))
    # print("SDREC: {:.1f} s".format(t3-t2))
    Plots(df_list, labels=labels)