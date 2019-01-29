''' (Quasi) Fuel Optimal Feedback for Double Integrator'''

import numpy as np 


class Controller1D:
    """ Provides a chatter-free, overshoot-free fuel optimal feedback control for planar, saturated double integrators """

    def __init__(self, alpha=0.4, rho=0, eps=None):
        """ 
            rho is the time-optimality weighting factor. typically set to zero for fuel optimality 
        """
        assert alpha < 0.5, "alpha must be less than 0.5"
        assert alpha > 0, "alpha must be greater than zero"
        assert rho > 0, "rho must be greater than zero"

        self.rho = rho 
        self.alpha = alpha 
        self._u_last = None 
        self.eps = eps

    def __call__(self, state):
   
        rho = self.rho 
        
        # for pure double integrator, F = 0, G1=G=G2=1
        # alpha < 0.5
        alpha = self.alpha
        eps_max = 4*(1-alpha)/alpha - rho
        eps_min = 4*alpha/(1-alpha) - rho
        assert eps_max >= eps_min, "Invalid combination of alpha/rho parameters "
        if self.eps is None:
            eps = 0.5*(eps_min+eps_max)
        else:
            eps = self.eps
            assert eps >= eps_min and eps <= eps_max, "Invalid choice of epsilon"
        
        x1, x2 = state
        S1 = x1 + x2*np.abs(x2)*(4+rho+eps)/(2*(rho+eps))
        S2 = x1 + x2*np.abs(x2)*0.5/(1-alpha)
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

    x0 = [c*4000, c*-120, 1050] # pos vel mass 
    controller_mfc = Controller1D(rho=1, eps=2, alpha=0.4)

    def dyn(x, t,):
        M = x[2]
    
        u = controller_mfc(x[:-1],)
        dv = c*(((p+400*np.exp(-0.02*t))/M)*u - mu/(x[0]/c + R)**2)
        
        dM = -k*np.abs(u)
        if np.abs(x[0]) + np.abs(x[1]) < 1:
            return 0,0,0
        else:
            return x[1], dv, dM

    tf = 100
    t = np.linspace(0, tf, 2000)
    x = Euler(dyn, x0, t, )

    dx = np.linalg.norm(np.diff(x, axis=0), axis=1)

    iterm = np.argmin(dx)

    plt.plot(x[:iterm,0]/c, x[:iterm,1]/c)
    plt.title("Phase Portrait")
    plt.figure()
    plt.plot(t[:iterm], controller_mfc.u_history[:iterm])
    plt.xlabel('Time (s)')
    plt.figure()
    plt.plot(t[:iterm], x[:iterm, 2])
    plt.xlabel('Time (s)')
    plt.figure()
    plt.plot(t[:iterm], x[:iterm, :2])
    plt.xlabel('Time (s)')
    print("Time of flight = {:.2f} s\nFinal altitude = {:.3f} m\nFinal velocity = {:.3f} m/s\nFuel consumed = {:.1f} kg".format(t[iterm],x[-1,0],x[-1,1],x0[2]-x[-1,2]))
    plt.show()


if __name__ == "__main__":
    test()
