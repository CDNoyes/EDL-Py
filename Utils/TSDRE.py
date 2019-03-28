"""
Implementation of the terminal controller from
    Near Optimal Finite-Time Terminal Controllers for Space Trajectories 
    via SDRE-based Approach Using Dynamic Programming 
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d 

try:
    from RK4 import RK4
except ModuleNotFoundError:
    from Utils.RK4 import RK4

class TSDRE:
    """ Takes an SDC-factorized class and solves
        terminally constrained optimal control problems
    """
    def __init__(self):

        self.cache = None 


    def __call__(self, tc, x0, model, problem, integration_steps=20):
        """ 
            Model is an SDC class 
            Problem is a dictionary that has 
                Q/R - LQ matrix functions defining the Lagrange cost function
                tf  - final time
                constraint - A callable of one argument (the current state) that returns the value and gradient of the terminal constraint 



        """
        tf = problem['tf']
        Q = problem['Q']
        R = problem['R']
        Ri = np.linalg.inv(R)

        t = np.linspace(tc, tf, integration_steps)
        tb = t[::-1]

        n = model.n 

        # Compute current system matrices 
        A, B, _, D = model(tc, x0)  
        c, C = problem['constraint'](x0)
        try:
            p = len(c)
            V0f = np.zeros((p, p))
        except TypeError:
            V0f = [0]

        # Compute gain matrices 
        S0 = RK4(dS0, np.zeros((n, n)), tb, args=(A, B, Q, Ri))[::-1]
        S0i = interp1d(t, S0, axis=0, bounds_error=False, fill_value=(S0[0], S0[-1]))

        P0 = RK4(dP0, C, tb, args=(A, B, Q, Ri, S0i))[::-1]
        P0i = interp1d(t, P0, axis=0, bounds_error=False, fill_value=(P0[0], P0[-1]))

        V0 = RK4(dV0, V0f, tb, args=(B, Ri, P0i))[-1]

        S1 = RK4(dS1, np.zeros((n,)), tb, args=(A, B, D, Q, Ri, S0i))[::-1]
        S1i = interp1d(t, S1, axis=0, bounds_error=False, fill_value=(S1[0], S1[-1]))

        V1f = c - C.dot(x0)  # For now, only a single constraint is considered so this is a scalar value 
        V1 = RK4(dV1, np.atleast_1d(V1f), tb, args=(A, B, D, Q, Ri, P0i, S1i))[-1]

        nu = lagrange(x0, P0[0], np.atleast_2d(V0),  V1)
        self.cache = nu 
        return control(x0, B, Ri, S0[0], S1[0], np.atleast_2d(P0[0]), nu)  # might need to use np.array( ndmin=n) if the scalar case doesn't work this way 

    # def solve(self, model, problem):
        # """ Calls __call__ method and integrates the resulting solution till tf """


def control(x, B, Ri, S0, S1, P0, nu):
    return -Ri.dot(B.T.dot(S0.dot(x) + P0.T.dot(nu) + S1))


def lagrange(x, P0, V0, V1):
    return -np.linalg.solve(V0, P0.dot(x) + V1)


# Gain differential equations 
def dS0(S, t, A, B, Q, Ri):
    SB = S.dot(B)  # + N  # but we don't really care about cross terms 
    return -(S.dot(A) + A.T.dot(S) - SB.dot(Ri).dot(SB.T) + Q)


def dP0(P, t, A, B, Q, Ri, S0):
    SB = S0(t).dot(B)  # + N  # but we don't really care about cross terms 
    return -P.dot(A - B.dot(Ri).dot(SB.T))


def dV0(V, t, B, Ri, P0):
    pb = P0(t).dot(B)
    return np.atleast_1d(pb.dot(Ri).dot(pb.T))


def dS1(S, t, A, B, D, Q, Ri, S0):
    SB = S0(t).dot(B)  # + N  # but we don't really care about cross terms 
    return -(A.T - SB.dot(Ri).dot(B.T)).dot(S) - S0(t).dot(D)


def dV1(V, t, A, B, D, Q, Ri, P0, S1):
    return np.atleast_1d(-P0(t).dot(D - B.dot(Ri).dot(B.T.dot(S1(t)))))


# Example/Demonstration Code 
def pt_constraint(xf):
    return xf, np.array([1])


def nl_constraint(x):
    return x**3-33./8.*x**2+7./8.*x-5.*np.sin(x)+147./32, 3.*x**2-33./4*x + 7./8-5.*np.cos(x)
   

def example(x0s, abc, constraint, N=50, plot_control=False, plot_contour=True):
    """ Generic scalar example for different polynomial systems, initial conditions, and terminal constraints. """
    import sys 
    sys.path.append("./EntryGuidance/SDC")
    from ScalarSystems import PolySystem 

    sys = PolySystem(*abc)
    problem = {'tf': 1, 'Q': [[0]], "R": [[1]], 'constraint': constraint}
    controller = TSDRE()
    t = np.linspace(0, problem['tf'], N)  # These are the N-1 times at which control is updated, and tf 
    dt = np.diff(t)[0]

    for x0 in x0s:

        X = [np.array([x0])]
        U = []
        Nu = []
        tc = 0 

        for i in range(N-1):
            u = controller(tc, X[-1], sys, problem)
            if 1:
                B = x0**3 + 5
                u = np.clip(u, -B, B)  # Optional saturation effects
            delta = min(dt, t[-1]-tc)
            xi = RK4(sys.dynamics(u), X[-1], np.linspace(tc, tc+delta, 3))  # _ steps per control update 
            X.append(xi[-1])
            U.append(u)
            Nu.append(controller.cache)
            tc += delta

        U.append(U[-1])

        plt.figure(1)
        plt.plot(t, X, label='X(tf) = {:.4f}'.format(X[-1][0]))
        plt.xlabel("Time")
        if plot_control:
            plt.figure(2)
            plt.plot(t, U)
            plt.xlabel("Time")
            plt.ylabel("Control")
    
    if plot_contour:
        plt.figure(1)
        z = np.linspace(-1.5, max(np.abs(x0s).max()+1,3.5), 5000)
        T, Z = np.meshgrid(np.linspace(0.99, 1.0, 3), z)
        C = constraint(np.array(Z))[0]
        plt.contour(T, Z, C, np.linspace(-1, 1, 101)*0.2, colors='k')  # Constraint, shown with thicker resolution

    plt.legend()
    plt.show()


def test():
    example([-1.5, -0.5, 1, 2], [0,1,0], pt_constraint, N=20, plot_contour=False)  # ex1 from reference 
    example([-1, 0, 2, 4], [0,0,0], nl_constraint, 20)  # ex2 from reference with add'l initial conditions
    # example([-1, 0, 4], [1, 0, -0.1], nl_constraint, 30)  

def test2d():
    import pandas as pd 

    import sys 
    sys.path.append("./EntryGuidance")
    from SDC.SDCBase import SDCBase
    from progress import progress 

    tf = 3
    N = 30
    a = 0.1
    # x0 = [13, -5.5] 
    # x0_mc = np.random.multivariate_normal(x0[:2], np.diag([1,1]), 20) 
    n_samples = 50
    x0_mc = np.random.random((n_samples,2))  # This isn't neighboring OC so we can use an arbitrary range of IC 
    x0_mc[:,0] = 6 + 6*x0_mc[:,0]
    x0_mc[:,1] = -5 + 10*x0_mc[:,1]


    class TerminalManifold(SDCBase):
        @property
        def n(self):
            """ State dimension """
            return 2

        @property
        def m(self):
            """ Control dimension """
            return 1
        
        def A(self, t, x):
            return np.array([[0,1],[0, -a*np.sqrt(1e-5+x[1]**2)]])

        def B(self, t, x):
            return np.array([[0],[1]])
        
        def C(self, *args):
            return np.eye(2)
        
    def constraint(x): 
        # return (x[0] - x[1]**2)**2, np.array([2*x[0]-2*x[1]**2, -4*x[0]*x[1] + 4*x[1]**3]).T
        # return (x[0] - x[1]**2), np.array([np.ones_like(x[0]), -2*x[1]]).T
        # return x[0]**2 + x[1]**2 - 10, np.array([2*x[0], 2*x[1]])  # This one doesn't work well for some reason, perhaps the lack of linear terms
        # return x[0]**2 + x[1]**2 + 2*x[0]*x[1]- 10, np.array([2*x[0]+2*x[1], 2*x[0]+2*x[1]]).T  
        # return x[0] + 0.1*x[0]**2 + x[1] + 0.1*x[1]**2 - 3, np.array([1+0.2*x[0], 1+0.2*x[1]]).T
        return x[0]-np.tan(x[1])*x[1]-0.1*x[1]**2, np.array([np.ones_like(x[0]), -0.2*x[1]-np.tan(x[1])-x[1]/(np.cos(x[1])**2)]).T
        # return x[1]**2 - 1, np.array([0, 2*x[1]]).T  # two vertical lines 
        # return x[0]**2 - 1, np.array([2*x[0], 0]).T  # two horizontal lines 
        # return x**2 - 1, np.diag(2*x)  # four isolated points 
        # return x[0]-x[1], np.array([1, -1]).T
        # return np.cos(x[0]), np.array([-np.sin(x[0]), 0]).T  # This one works impressively well 
        # return np.cos(x[0])+np.sin(x[1]), np.array([-np.sin(x[0]), np.cos(x[1])]).T  # This one works impressively well 
        # return x[1]*np.cos(x[0])-1, np.array([-np.sin(x[0])*x[1], np.cos(x[0])]).T 

        # return x - np.array([1,0], ndmin=np.ndim(x)).T, np.eye(2)  # 2 linear point constraints!!
        # return np.array([x[0]+x[1], x[0]-x[1]]), np.array([[1,1],[1,-1]])  # Two linear constraints whose intersection is a single point!!

    problem = {'tf': tf, 'Q': np.eye(2)*0., "R": [[1]], 'constraint': constraint}
    model = TerminalManifold()
    controller = TSDRE()

    t = np.linspace(0, tf, N)  # These are the N-1 times at which control is updated, and tf 
    dt = np.diff(t)[0]
    XMC = []
    Nuf = []
    for x0 in x0_mc:
        if progress(len(XMC), n_samples):
            print("Iter {}".format(len(XMC)))
        X = [x0]
        U = []
        Nu = []
        tc = 0 

        for i in range(N-1):
            u = controller(tc, X[-1], model, problem)
            if 0:
                B = 2.
                u = np.clip(u, -B, B)  # Optional saturation effects
            delta = min(dt, t[-1]-tc)
            xi = RK4(model.dynamics(u), X[-1], np.linspace(tc, tc+delta, 3))  # _ integration steps per control update 
            X.append(xi[-1])
            U.append(u)
            Nu.append(controller.cache)
            tc += delta
        plt.figure(1)
        plt.plot(t[:-1], Nu)
        plt.xlabel('Time')
        plt.ylabel('Lagrange Multiplier(s)')

        plt.figure(3)
        plt.plot(t[:-1], U)
        plt.xlabel('Time')
        plt.ylabel('Control')
        XMC.append(X)
        Nuf.append(Nu[-1])

    XMC = np.array(XMC)
    Nuf = np.array(Nuf).squeeze()
    print(XMC.shape)  # M samples, N points, n states
    plt.figure(2)
    # plt.scatter(XMC[:,0,1], XMC[:,0,0], c=Nuf, label='Initial State')  # Initial
    plt.scatter(XMC[:,0,1], XMC[:,0,0], c='g', label='Initial State')  # Initial
    plt.scatter(XMC[:,-1,1], XMC[:,-1,0], marker='o', label="Final State", c='r')  # Final
    for traj in XMC:
        plt.plot(traj.T[1], traj.T[0], 'b', alpha=0.2)
        

    if 1:
        xmin = XMC[:,:,0].min()-1
        xmax = XMC[:,:,0].max()+1
        x = np.linspace(xmin, xmax, 500)

        ymin = XMC[:,:,1].min()-1
        ymax = XMC[:,:,1].max()+1
        y = np.linspace(ymin, ymax, 600)
        XY = np.meshgrid(x,y)
        Z = constraint(XY)[0]
        if Z.ndim == 2:
            ctr = plt.contour(XY[1], XY[0], Z, [0], colors='k')
            label = ""
        else:
            ctr = plt.contour(XY[1], XY[0], Z[0], [0], colors='k')
            plt.contour(XY[1], XY[0], Z[1], [0], colors='m')
            label = "s"

        h = ctr.legend_elements()[0][0]
        plt.legend([h], ["Terminal Manifold"+label])
    plt.title("Phase Portrait")
    plt.xlabel('v')
    plt.ylabel('x')
    plt.show()



if __name__ == "__main__":
    # test()
    test2d()


