"""
    SDC factorizations of a scalar polynomial system 

"""

import numpy as np 
from SDCBase import SDCBase
from replace import safe_divide


class PolySystem(SDCBase):
    """
        dx = ax + bx^2 + cx^3 + u

    """

    @property
    def n(self):
        return 1
    
    @property
    def m(self):
        return 1

    def __init__(self, a, b, c):
        self.abc = (a, b, c)

    def A(self, t, x):
        a,b,c = self.abc
        return np.asarray([a+b*x+c*x**2])

    def B(self, t, x):
        return np.array([[1]])

    def C(self, t, x):  
        return np.eye(1)

    def D(self, t):
        """ Affine terms """
        return np.zeros(1)


class ConstrainedPolySystem(SDCBase):
    """
        The same 1D system but with hard limit on the control 

        dx = ax + bx^2 + cx^3 + sat(u, umax)
        du = v 

        In conclusion - for this problem, including saturation effects
        did not improve performance relative to using the unconstrained version
        and simply saturating the control before applying it. Perhaps for a vector
        control problem with a norm constraint, perhaps including the saturation
        nonlinearity is helpful.

    """

    @property
    def n(self):
        return 2
    
    @property
    def m(self):
        return 1

    def __init__(self, a, b, c, umax):
        self.abc = (a, b, c)
        self.max = umax 

    def A(self, t, x):
        a,b,c = self.abc
        u = x[1]
        du = safe_divide(np.clip(u, -self.max, self.max), u, 1)
        return np.array([[a+b*x[0]+c*x[0]**2, du], [0, 0]])

    def B(self, t, x):
        return np.array([[0], [1]])

    def C(self, t, x):  
        return np.eye(2)

    def D(self, t):
        """ Affine terms """
        return np.zeros(2)


if __name__ == "__main__":
    import warnings 
    import matplotlib.pyplot as plt 
    from scipy.integrate import trapz 
    import sys 
    sys.path.append("./Utils")
    from TSDRE import TSDRE 
    from RK4 import RK4 

    umax = 4
    model = ConstrainedPolySystem(-0.1, 0.05, 0.1, umax)
    controller = TSDRE()
    N = 50
    plot_contour = True

    def pt_constraint(xf):
        return xf[0], np.array([1, 0])

    def nl_constraint(z):
        x, u = z 
        return x**3-33./8.*x**2+7./8.*x-5.*np.sin(x)+147./32, np.array([3.*x**2-33./4*x + 7./8-5.*np.cos(x), np.zeros_like(x)])

    problem = {'tf': 1, 'Q': lambda y: np.diag([0, 1]), "R": lambda y: [[0.001 + 0*(y[1]/umax)**4]], 'constraint': nl_constraint}

    t = np.linspace(0, problem['tf'], N)  # These are the N-1 times at which control is updated, and tf 
    dt = np.diff(t)[0]
    x0s = np.linspace(-3, 3, 5)
    # x0s = [-3.5, 0, 3]
    plt.figure(1)
    for x0 in x0s:
        for clip in [False]:

            # X = [np.array([x0, -np.sign(x0)*umax])] # This works well 
            X = [np.array([x0, -x0])]
            U = []
            Nu = []
            tc = 0 

            for i in range(N-1):
                xc = X[-1]
                if clip: # This seems to improve performance sometimes 
                    xc[1] = np.clip(xc[1], -umax, umax)  
                    # Interestingly, it is equivalent to ignoring the saturation effects when computing A(x)
                    # While retaining them (as necessary) when integrating 

                u = controller(tc, xc, model, problem)   # This is actually the virtual control 
                delta = min(dt, t[-1]-tc)
                xi = RK4(model.dynamics(u), xc, np.linspace(tc, tc+delta, 10))  # _ steps per control update 
                X.append(xi[-1])
                U.append(u)
                Nu.append(controller.cache)
                tc += delta

            U.append(U[-1])
            X = np.array(X).T
            J = trapz(x=t, y=np.clip(X[1], -umax, umax)**2)
            label = "J = {:.2f}".format(J)
            plt.subplot(2, 1, 1)
            if clip:
                plt.plot(t, X[0], '--', label="Saturated Between Calls, "+label)

            else:
                plt.plot(t, X[0], label=label)
            plt.ylabel("State")

            plt.subplot(2, 1, 2)
            if clip:
                plt.plot(t, np.clip(X[1], -umax, umax), '--', label="Saturated Between Calls")

            else:
                plt.plot(t, np.clip(X[1], -umax, umax))
            plt.xlabel("Time")
            plt.ylabel("Control")
    
    if plot_contour:
        plt.subplot(2, 1, 1)
        z = np.linspace(-1.5, max(np.abs(x0s).max()+1,3.5), 5000)
        T, Z = np.meshgrid(np.linspace(t[-1]-0.01, t[-1], 3), z)
        C = problem['constraint']([np.array(Z),0])[0]
        plt.contour(T, Z, C, np.linspace(-1, 1, 101)*0.2, colors='k')  # Constraint, shown with thicker resolution

    plt.legend()
    plt.show()