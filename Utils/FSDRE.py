"""
Implementation of the finite SDRE controller from
    Path Planning Using a Novel Finite Horizon Suboptimal Controller
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d 
from scipy.linalg import solve_continuous_are, solve_continuous_lyapunov, expm


class FSDRE:
    """ Takes an SDC-factorized class and solves
        finite horizon optimal control problems
    """
    def __init__(self):

        self.cache = None 

    def __call__(self, tc, x0, model, problem):
        """ 
            Model is an SDC class 
            Problem is a dictionary that has 
                Q/R - LQ matrix functions defining the Lagrange cost function
                tf  - final time
                constraint - A callable of one argument (the current state) that returns the value and gradient of the terminal constraint 



        """
        tf = problem['tf']
        tr = tc - tf  

        n = model.n 

        # Compute current system matrices 
        A, B, _, D = model(tc, x0)  
        if B.ndim == 1:
            B = B[:,None]
        Q = problem['Q'](x0)
        R = problem['R'](x0)
        S = problem['S'](x0)
        Ri = np.linalg.inv(R)
        BRB = B.dot(Ri).dot(B.T)

        try:
            Pss = -solve_continuous_are(-A, B, Q, R)
            Kf = np.linalg.inv(S-Pss)
            Acl = A - BRB.dot(Pss)
            D = solve_continuous_lyapunov(Acl, BRB)
            M1 = expm(Acl*tr)
            M2 = expm(Acl.T*tr)
            K = M1.dot(Kf-D).dot(M2) + D 
            P = np.linalg.inv(K) + Pss 
            U = control(x0, B, Ri, P)
            self.cache = U 
        except np.linalg.LinAlgError:
            if self.cache is None:
                print("LinAlg Error in controller at initial call.")
                raise ValueError 
            else:
                U = self.cache 
        return U

    # def solve(self, model, problem):
        # """ Calls __call__ method and integrates the resulting solution till tf """


def control(x, B, Ri, P):
    return -Ri.dot(B.T.dot(P.dot(x)))

   

def example(x0s, abc, N=50, tf=1):
    """ Generic scalar example for different polynomial systems, initial conditions, and terminal constraints. """
    from scipy.integrate import trapz 
    import sys 
    sys.path.append("./EntryGuidance/SDC")
    from ScalarSystems import PolySystem 
    from RK4 import RK4

    sys = PolySystem(*abc)
    problem = {'tf': tf, 'Q': lambda y: [[0]], "R": lambda y: [[1]], 'S': lambda z: [[1000]]}
    controller = FSDRE()
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
                # B = x0**3 + 5
                B = 4.1
                u = np.clip(u, -B, B)  # Optional saturation effects
            delta = min(dt, t[-1]-tc)
            xi = RK4(sys.dynamics(u), X[-1], np.linspace(tc, tc+delta, 3))  # _ steps per control update 
            X.append(xi[-1])
            U.append(u)
            Nu.append(controller.cache)
            tc += delta
        U.append(U[-1])
        u = np.array(U)

        plt.figure(1)
        plt.plot(t, X)
        plt.xlabel("Time")
        
        plt.figure(2)
        plt.plot(t, U)
        plt.xlabel("Time")
        plt.ylabel("Control")


    plt.legend()
    plt.show()


def test():
    example([-1.5, -0.5, 1, 2], [0,1,0], N=20, tf=2)  # ex1 from reference 
    # example([-1, 0, 2, 4], [0,0,0], nl_constraint, 20)  # ex2 from reference with add'l initial conditions
    # example([-1, 0, 4], [1, 0, -0.1], nl_constraint, 30)  

    # example([-1, 0, 2, 4], [-1, 0.1, 0.2], N=120, tf=30, plot_control=True)  # Just to demonstrate a long horizon is possible 
    # example([-3, -1.5, 0, 1.5, 3], [-0.1, 0.05, 0.1], N=50, tf=1, plot_control=True)  # For comparison with constrained version in ScalarSystems.py 

def test2d():
    import sys 
    sys.path.append("./EntryGuidance")
    from SDC.SDCBase import SDCBase
    from progress import progress 
    from RK4 import RK4

    tf = 5
    N = 50
    a = 0.1
    # x0 = [13, -5.5] 
    # x0_mc = np.random.multivariate_normal(x0[:2], np.diag([1,1]), 20) 
    n_samples = 100
    x0_mc = np.random.random((n_samples, 2))  # This isn't neighboring OC so we can use an arbitrary range of IC 
    x0_mc[:, 0] = 6 + 6*x0_mc[:, 0]
    x0_mc[:, 1] = -5 + 10*x0_mc[:, 1]

    class TerminalManifold(SDCBase): # Dynamics for demonstrating a 2d system 
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
        

    problem = {'tf': tf, 'Q': lambda y: np.eye(2)*1, "R": lambda y: [[1]], 'S': lambda z: np.diag([10, 1])*10}
    model = TerminalManifold()
    controller = FSDRE()

    t = np.linspace(0, tf, N)  # These are the N-1 times at which control is updated, and tf 
    dt = np.diff(t)[0]
    XMC = []
    for x0 in x0_mc:
        if progress(len(XMC)-1, n_samples):
            print("Iter {}".format(len(XMC)))
        X = [x0]
        U = []
        tc = 0 

        for i in range(N-1):
            u = controller(tc, X[-1], model, problem)
            if 1:
                B = 5.
                u = np.clip(u, -B, B)  # Optional saturation effects
            delta = min(dt, t[-1]-tc)
            xi = RK4(model.dynamics(u), X[-1], np.linspace(tc, tc+delta, 3))  # _ integration steps per control update 
            X.append(xi[-1])
            U.append(u)
            tc += delta

        plt.figure(3)
        plt.plot(t[:-1], U)
        plt.xlabel('Time')
        plt.ylabel('Control')
        XMC.append(X)

    XMC = np.array(XMC)
    print(XMC.shape)  # M samples, N points, n states
    plt.figure(2)
    plt.scatter(XMC[:,0,1], XMC[:,0,0], c='g', label='Initial State')  # Initial
    plt.scatter(XMC[:,-1,1], XMC[:,-1,0], marker='o', label="Final State", c='r')  # Final
    for traj in XMC:
        plt.plot(traj.T[1], traj.T[0], 'b', alpha=0.2)
        
    plt.title("Phase Portrait")
    plt.xlabel('v')
    plt.ylabel('x')
    plt.show()



if __name__ == "__main__":
    # test()
    test2d()


