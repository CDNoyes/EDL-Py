''' (Quasi) Fuel Optimal Feedback for Double Integrator'''

import numpy as np


# class Controller3D:

#     def __init__(self):
#         self.controllers = [Controller1D() for i in range(3)]

#     def __call__(self, state):
#         x,y,z,u,v,w = state 

#         ux = self.controllers[0]([x,u])
#         uy = self.controllers[0]([y,v])
#         uz = self.controllers[0]([z,w])

#         return [ux, uy, uz]


class MemorizedController:
    """ An abstract base class for memorized controllers in the form of 

        Quasi Time Fuel Optimal Feedback Control of Perturbed Double Integrator  

    """

    def __init__(self, rho=0, alpha=None, eps=None):
        """ 
            rho is the time-optimality weighting factor. typically set to zero for fuel optimality 
        """
        
        assert rho >= 0, "rho must be greater than or equal to zero"
        if alpha is not None:
            assert alpha < 0.5, "alpha must be less than 0.5"
            assert alpha > 0, "alpha must be greater than zero"

        # else:
        #     self.compute_params()

        self.rho = rho 
        self.alpha = alpha 
        self._u_last = None 
        self.eps = eps

    def __call__(self, state):
        """ Computes the feedback response for a given state """ 

        
        # for pure double integrator, F = 0, G1=G=G2=1
        # alpha < 0.5
        rho = self.rho 
        alpha = self.alpha
        eps_max = 4*(1-alpha)/alpha - rho
        eps_min = 4*alpha/(1-alpha) - rho
        assert eps_max >= eps_min, "Invalid combination of alpha/rho parameters "
        if self.eps is None:
            eps = eps_min + 1e-2*(eps_max-eps_min)
        else:
            eps = self.eps
            assert eps >= eps_min and eps <= eps_max, "Invalid choice of epsilon"
        
        x1, x2 = state
        S1, S2 = self.get_curves(state, alpha, eps)
        S = S1*S2

        # Initialize the memory
        if self._u_last is None:
            if np.abs(S1) > 1 and np.abs(S2) > 1:
                u = -np.sign(np.sign(S1) + np.sign(S2))
            else:
                if np.abs(S1) < np.abs(S2):
                    u = 0
                else:
                    u = -np.sign(x2)
            self._u_last = u
            self._S_history = [S]
            self._S1_last = S1

            self.u_history = [u]
            return u
    
        if S*self._S_history[-1] < 0:  # Have to update the memory for either branch here
            if S1*self._S1_last < 0:
                u = 0
            else:
                u = -np.sign(x2)
            self._u_last = u
        else:
            if S < 0:
                u = self._u_last 

            else:
                u = -np.sign(np.sign(S1) + np.sign(S2))
        
        self._S1_last = S1

        self._S_history.append(S)
        self.u_history.append(u)    
        return u

    def reset(self):
        self._u_last = None 
        self._S_history = None 
        self.u_history = None 

    def get_curves(self, *args):
        raise NotImplementedError("User must implement get_curves method")


class Controller1D(MemorizedController):
    """ Provides a chatter-free, overshoot-free fuel optimal feedback control for planar, saturated double integrators """

    def get_curves(self, state, alpha, eps):
        x1, x2 = state
        S1 = x1 + x2*np.abs(x2)*(4+self.rho+eps)/(2*(self.rho+eps))
        S2 = x1 + x2*np.abs(x2)*0.5/(1-alpha)
        return S1, S2         

    def compute_params(self, F, G1, G2, verbose=False):
        assert G2 >= G1, "Upper bound lower than lower bound"
        G = G2

        alpha_min = (F+G2-G1)/G
        assert alpha_min < 0.5, "Cannot satisfy requirements"

        alpha = max(alpha_min, 1e-3)  # ensures non-zero 
        eps_max = 4*(1-alpha)/alpha - self.rho
        eps_min = 4*alpha/(1-alpha) - self.rho
        eps = np.clip(eps_min+.01, 0.01, eps_max)

        self.alpha = alpha 
        self.eps = eps 
        if verbose:
            print("Setting:\n  alpha = {:.3f}\n  epsilon= {:.2f}".format(alpha, eps))


    def plotS(self):
        import matplotlib.pyplot as plt 

        c = 1000/4000
        x1 = np.linspace(0, 4200, 5000)
        x2 = np.linspace(0, -450, 4000)
        
        X1, X2 = np.meshgrid(x1*c, x2*c)
        
        S1 = X1 + (4+self.rho+self.eps)/(2*self.rho+2*self.eps)*X2*np.abs(X2)
        S2 = X1 + 1/(2-2*self.alpha)*X2*np.abs(X2)
        
        plt.contour(X1/c, X2/c, S1, 0, colors='k', linestyles='dashed')
        plt.contour(X1/c, X2/c, S2, 0, colors='k', linestyles='dashed')


def controller_gravity(x, g):
    """ 1-D Fuel Optimal Feedback, accounting for a constant gravity acceleration 
        This is not memorized so chattering will occur, and does not guarantee zero overshoot 
    
    """

    x1, x2 = x
    
    c1 = (1 + 0.5*g/(1-g))/(1-g)
    xa = x2**2 * c1
    xb = (c1 + 4/g)*x2**2
    
    # for negative altitudes, a different set of checks must be used
    c2 = (0.5*g/(1+g)-1)/(1+g)  # Used for non-physical cases, to complete the semi-global stabilization property 
    xc = x2**2 * c2 
    xd = (c2 + 4/g)*x2**2
    
    u = np.zeros_like(x1)
    n1 = np.logical_and(np.logical_and(x1 >= xa, x1 > xb), x2 <= 0) 
    p1 = np.logical_and(np.logical_and(x1 <= xa, x1 < xb), x2 <= 0) 

    n2 = np.logical_and(np.logical_and(x1 >= xc, x1 > xd), x2 >= 0) 
    p2 = np.logical_and(np.logical_and(x1 <= xc, x1 < xd), x2 >= 0) 

    n = np.logical_or(n1, n2)  # H1
    p = np.logical_or(p1, p2)  # H3 

    u[n] = -1
    u[p] = 1
    return u


def test():
    """ Recreates the example from the paper: 
        Quasi Time Fuel Optimal Feedback Control 
        of Perturbed Double Integrator   
     
    """
    import sys
    import matplotlib.pyplot as plt 
    sys.path.append('./')

    from Utils.RK4 import Euler 
    

    p = 4000.
    c = 1000/p
    R = 1.738e6
    k = .5 
    mu = 4.887e12

    x0 = [4000, -120, 1050] # pos vel mass 

    controller_mfc = Controller1D(rho=1, eps=2, alpha=0.4)

    def dyn(x, t,):
        M = x[2]
    
        u = controller_mfc(x[:-1]*c,)
        dv = (((p+400*np.exp(-0.02*t))/M)*u - mu/(x[0] + R)**2)
        
        dM = -k*np.abs(u)
        if np.abs(x[0]) + np.abs(x[1]) < 1:
            return 0,0,0
        else:
            return x[1], dv, dM

    tf = 100
    t = np.linspace(0, tf, 20000)
    x = Euler(dyn, x0, t, )

    dx = np.linalg.norm(np.diff(x, axis=0), axis=1)

    iterm = np.argmin(dx)

    controller_mfc.plotS()
    plt.plot(x[:iterm, 0], x[:iterm, 1])
    plt.title("Phase Portrait")

    plt.figure()
    plt.plot(t[:iterm], controller_mfc.u_history[:iterm])
    plt.xlabel('Time (s)')
    plt.ylabel("Throttle setting")

    plt.figure()
    plt.plot(t[:iterm], x[:iterm, 2])
    plt.xlabel('Time (s)')
    plt.ylabel('Mass (kg)')

    print("Time of flight = {:.2f} s\nFinal altitude = {:.3f} m\nFinal velocity = {:.3f} m/s\nFuel consumed = {:.1f} kg".format(t[iterm],x[-1,0],x[-1,1],x0[2]-x[-1,2]))
    J = np.sum(1+np.abs(controller_mfc.u_history[:iterm]))*t[1]
    print("Objective = {:.1f}".format(J))
    plt.show()


def test_mc():
    """ Tries the feedback under various IC 
     
    """
    import sys
    import matplotlib.pyplot as plt 
    sys.path.append('./')
    import chaospy as cp 

    from Utils.RK4 import Euler 
    

    p = 4000.
    c = 1000/p
    R = 1.738e6
    k = .5 
    mu = 4.887e12

    U = [cp.Uniform(3000, 5500), cp.Uniform(-150, -50), cp.Uniform(1045, 1055)]

    X0 = cp.J(*U).sample(50, 'L').T
    # x0 = [4000, -120, 1050] # pos vel mass 

    controller_mfc = Controller1D(rho=1, eps=2, alpha=0.4)
    controller_mfc.compute_params(F=1.62*c, G1=1.1*1000/1050, G2=1.1*1000/1000, verbose=True)

    def dyn(x, t,):
        crash = x[0] <= 0 

        M = x[2]
    
        u = controller_mfc(x[:-1]*c,)
        dv = (((p+400*np.exp(-0.02*t))/M)*u - 0*mu/(x[0] + R)**2)
        
        dM = -k*np.abs(u)
        if np.abs(x[0]) + np.abs(x[1]) < 1 or crash:
            return 0,0,0
        else:
            return x[1], dv, dM

    tf = 100
    t = np.linspace(0, tf, 1000)
    plt.figure(1)
    controller_mfc.plotS()

    for x0 in X0:
        x = Euler(dyn, x0, t, )
        controller_mfc.reset()
        dx = np.linalg.norm(np.diff(x, axis=0), axis=1)

        iterm = np.argmin(dx)

        plt.plot(x[:iterm, 0], x[:iterm, 1])
        plt.title("Phase Portrait")

    plt.show()


def test_gravity_controller():
    g = 0.5  # this is scaled by umax, since u is bounded by 1, g must be less than 1

    def dyn(x, t):
        # Vectorized dynamics that stop when the velocity becomes positive
        dx = np.zeros_like(x)

        ix = np.logical_and(x[0] >= 0 , np.abs(x[0]) + np.abs(x[1]) > 0.01)
        dx[0, ix] = x[1, ix]
        
        u = controller_gravity(x, g) # 5% uncertainty in gravity 
        dx[1, ix] = u[ix] - g
        # dx[1] = u - g
        return dx 

    import sys
    import matplotlib.pyplot as plt 
    sys.path.append('./')
    import chaospy as cp 

    from Utils.RK4 import Euler 
    t = np.linspace(0, 100, 1000)
    U1 = cp.Uniform(2, 5)
    # U2 = -U1/4.
    U2 = cp.Uniform(-2, 1)
    U = [U1, U2]
    X0 = cp.J(*U).sample(50, 'L')
    X = Euler(dyn, X0, t, ) 
    for x in np.transpose(X, (2, 0, 1)):
        dx = np.linalg.norm(np.diff(x, axis=0), axis=1)
        iterm = np.argmin(dx)
        x = x[:iterm]
        u = controller_gravity(x.T, g)

        plt.plot(x.T[0], x.T[1])
    plt.plot(X0[0], X0[1], 'ko', label="Initial Condition")
    plt.figure()
    plt.plot(t[:iterm], u)
    plt.show()


if __name__ == "__main__":
    test()
    # test_mc()
    # test_gravity_controller()