''' (Quasi) Fuel Optimal Feedback for Double Integrator

Summary of controllers:
    Fuel Optimal - No gravity, from Athans and Falb 
    Fuel Optimal - gravity, self derived 
    Quasi Optimal - No gravity, but is robust to uncertainty 
    Quasi Optimal - gravity, in progress 

'''

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
        
        x1, x2, _ = state
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
        x1, x2, _ = state
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
        eps = np.clip(eps_min+.001, 0.01, eps_max)

        self.alpha = alpha 
        self.eps = eps 
        if verbose:
            print("Setting:\n  alpha = {:.3f}\n  epsilon= {:.2f}".format(alpha, eps))


    def plotS(self):
        import matplotlib.pyplot as plt 

        c = 1050/4000
        x1 = np.linspace(0, 4200, 5000)
        x2 = np.linspace(0, -160, 4000)
        
        X1, X2 = np.meshgrid(x1*c, x2*c)
        
        S1 = X1 + (4+self.rho+self.eps)/(2*self.rho+2*self.eps)*X2*np.abs(X2)
        S2 = X1 + 1/(2-2*self.alpha)*X2*np.abs(X2)
        
        plt.contour(X1/c, X2/c, S1, 0, colors='k', linestyles='dashed')
        plt.contour(X1/c, X2/c, S2, 0, colors='k', linestyles='dashed')


class MemorizedGravityController(MemorizedController):
    """ Implements a quasi-optimal controller accounting for both
        constant gravitational acceleration and mass dynamics 
    """
    def set_gravity(self, g):
        self.gravity = g 

    def get_curves(self, state, alpha, eps):
        x1, x2 = state
        g = self.gravity 

        s = np.sign(x2)
        c = (-s + 0.5*g/(1+s*g))/(1+s*g)
        S1 = x1 - c*x2**2
        S2 = x1 - 1/(1-alpha)*(c + 4/g)*x2**2 
        return S1, S2 


def controller_fuel(x):
    """ The purely fuel optimal controller """
    x1, x2, _ = x
    m = 100
     
    a1 = x1 + m*x2*np.abs(x2)
    b1 = x1*a1

    a2 = x1 + 0.5*x2*np.abs(x2)
    b2 = x1*a2
    eps = 1e-2

    u = np.zeros_like(x1)
    u[b1 >= eps] = -np.sign(a1[b1 >= eps])
    u[b2 <= -eps] = -np.sign(a2[b2 <= -eps])
    return u 


def controller_gravity(x, g):
    """ 1-D Fuel Optimal Feedback, accounting for a constant gravity acceleration 
        This is not memorized so chattering will occur, and does not guarantee zero overshoot 
        It is a vectorized implementation for both states and gravitational constant 
    
    """

    x1, x2, _ = x
    u = np.zeros_like(x1)

    eps = 0.1
    p = x1 - x2**2 * 0.5/(1-g) < eps
    u[p] = 1 

    return u


def controller_all(x, g, k, M):
    """ 1D Fuel Optimal Feedback in constant gravity with mass loss 
        recovers the gravity controller when k goes to zero (cannot be exactly zero in computation)
        and further the standard fuel optimal controller when g = 0

        TODO: Compare to open loop optimal from GPOPS 
        # verify switch time(s) in terms of costates 

    """
    x1, x2, m0 = x 
    u = np.zeros_like(x1)

    d = M/m0 

    # Predicted time at 0 velocity based on second order Taylor expansion 
    tf = -m0/k/d * (d - g - np.sqrt((d-g)**2 - 2*d*k*x2/m0)) 
    try:
        tf = max(0, tf)
    except ValueError:
        tf[tf < 0] = 0
    
    xf = get_state(x, tf, k, g, M)  
    zf = xf[0]      # predicted altitude at 0 velocity 
    p = zf < 0.1  # Scaled tolerance, 1/c meters. Note, this is an easy way to control the final altitude at zero velocity 
    u[p] = 1
    return u


def get_state(x0, t, k, g, M):
    " Analytical Solution to EoM for u=1"
    z0, v0, m0 = x0 
    
    m = m0 - k*t

    v = v0 - g*t - M/k*np.log(1-k*t/m0)
    z = z0 + v0*t - 0.5*g*t**2 + M/k*(t + (m0/k-t)*np.log(1-k*t/m0))
    return [z,v,m]


def example():
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

    x0 = [4000, -120, 1050]  # pos vel mass 

    rho = 1
    controller_mfc = Controller1D(rho=rho, eps=2, alpha=0.4)

    label = 'Quasi-Fuel Optimal'

    def dyn(x, t,):
        M = x[2]
    
        y = [x[0]*c, x[1]*c, M]
        u = controller_mfc(y,)
        dv = (((p+400*np.exp(-0.02*t))/M)*u - mu/(x[0] + R)**2)
        
        dM = -k*np.abs(u)
        if x[0] <= 0.1:
            return 0,0,0
        else:
            return x[1], dv, dM

    tf = 70
    t = np.linspace(0, tf, 10000)
    x = Euler(dyn, x0, t, )
    dx = np.linalg.norm(np.diff(x, axis=0), axis=1)

    iterm = np.argmin(dx)
    u = controller_mfc.u_history[:iterm]

    controller_mfc.plotS()
    plt.figure(1)
    plt.plot(x[:iterm, 0], x[:iterm, 1], label=label)
    plt.title("Phase Portrait")
    plt.xlabel('Altitude (m)')
    plt.ylabel('Velocity (m/s)')

    plt.figure(2)
    plt.plot(t[:iterm], u, label=label)
    plt.xlabel('Time (s)')
    plt.ylabel("Throttle setting")

    plt.figure(3)
    plt.plot(t[:iterm], x[:iterm, 2], label=label)
    plt.xlabel('Time (s)')
    plt.ylabel('Mass (kg)')

    print("Time of flight = {:.2f} s\nFinal altitude = {:.3f} m\nFinal velocity = {:.3f} m/s\nFuel consumed = {:.1f} kg".format(t[iterm],x[iterm,0],x[iterm,1],x0[2]-x[iterm,2]))
    J = np.sum(rho+np.abs(u))*t[1]
    print("Objective = {:.1f}".format(J))
    plt.show()


def test():
    """  Compares multiple controllers 

        In the absence of gravity and mass dynamics, 
        the fuel optimal and quasi fuel optimal produce
        nearly identical responses 

        Even with mass dynamics, the solutions remain nearly identical

    """
    import sys
    import matplotlib.pyplot as plt 
    sys.path.append('./')

    from Utils.RK4 import Euler 
    
    massloss = True
    gravity = True
    nonlinear_gravity = False
    thrust_perturb = False

    p = 4000.
    c = 1050/p
    R = 1.738e6
    k = 5.5 * massloss 
    mu = 4.887e12 * gravity 

    x0 = [4000, -120, 1050] # pos vel mass 

    rho = 0
    controller_mem_gravity = Controller1D(rho=rho, eps=3, alpha=0.4)
    controller_mem_gravity.compute_params(F=1.62*c*gravity, G1=1, G2=1+massloss*(1050/(1050-20)-1), verbose=True)
    if gravity:
        controller_g = lambda x: controller_gravity(x, c*mu/R**2)
    else:
        # controller_g = controller_fuel 
        controller_g = lambda x: controller_gravity(x, 0)

    labels = ['Quasi-Fuel Optimal', 'FO + gravity', 'FO + g + mass loss']

    for controller_mfc, label in zip([controller_mem_gravity, controller_g, lambda x: controller_all(x, gravity*c*1.62 + 0.05*nonlinear_gravity, 0.000001 + k*massloss, x0[2])], labels):
        print("\n{} controller:".format(label))
        def dyn(x, t,):
            M = x[2]
        
            y = [x[0]*c, x[1]*c, M]
            u = controller_mfc(y,)
            dv = (((p+thrust_perturb*400*np.exp(-0.02*t))/M)*u - mu/(x[0]*nonlinear_gravity + R)**2)
            
            dM = -k*np.abs(u)
            if x[0] <= 0.1:
                return 0,0,0
            else:
                return x[1], dv, dM

        tf = 70
        t = np.linspace(0, tf, 10000)
        x = Euler(dyn, x0, t, )
        dx = np.linalg.norm(np.diff(x, axis=0), axis=1)

        iterm = np.argmin(dx)

        try:
            controller_mfc.plotS()
        except:
            pass 

        plt.figure(1)
        plt.plot(x[:iterm, 0], x[:iterm, 1], label=label)
        plt.title("Phase Portrait")
        plt.xlabel('Altitude (m)')
        plt.ylabel('Velocity (m/s)')

        try:
            u = controller_mfc.u_history[:iterm]
        except:
            y = x[:iterm, 0]*c, x[:iterm, 1]*c, x[:iterm, 2]
            u = controller_mfc(y)

        plt.figure(2)
        plt.plot(t[:iterm], u, label=label)
        plt.xlabel('Time (s)')
        plt.ylabel("Throttle setting")

        plt.figure(3)
        plt.plot(t[:iterm], x[:iterm, 2], label=label)
        plt.xlabel('Time (s)')
        plt.ylabel('Mass (kg)')

        print("\tTime of flight = {:.2f} s\n\tFinal altitude = {:.3f} m\n\tFinal velocity = {:.3f} m/s\n\tFuel consumed = {:.1f} kg".format(t[iterm],x[iterm,0],x[iterm,1],x0[2]-x[iterm,2]))
        J = np.sum(rho+np.abs(u))*t[1]
        print("\tObjective = {:.1f}".format(J))
        print("\tVar[u] = {:.3f}".format(np.var(u)))

    for i in range(1, 4):
        plt.figure(i)
        plt.legend()
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
    g = 0.25  # this is scaled by umax, since u is bounded by 1, g must be less than 1

    def dyn(x, t):
        # Vectorized dynamics that stop when the velocity becomes positive
        dx = np.zeros_like(x)

        
        u = controller_gravity(x, g) # 5% uncertainty in gravity 
        if 1:   # Use this for physical cases, i.e. x1 < 0  indicates a crash 
            ix = np.logical_and(x[0] >= 0, np.abs(x[0]) + np.abs(x[1]) > 0.01)
            dx[0, ix] = x[1, ix]
            dx[1, ix] = u[ix] - g
        else:       # Use this to demonstrate that the controller stabilizes any point 
            dx[0] = x[1]
            dx[1] = u - g
        return dx 

    import sys
    import matplotlib.pyplot as plt 
    sys.path.append('./')
    import chaospy as cp 

    from Utils.RK4 import Euler 
    t = np.linspace(0, 20, 1000)
    U1 = cp.Uniform(2, 5)
    # U2 = -U1/4.
    U2 = cp.Uniform(-2.5, 0)
    U = [U1, U2]
    X0 = cp.J(*U).sample(500, 'S')
    X = Euler(dyn, X0, t, ) 
    for x in np.transpose(X, (2, 0, 1)):
        dx = np.linalg.norm(np.diff(x, axis=0), axis=1)
        iterm = np.argmin(dx)
        x = x[:iterm]
        u = controller_gravity(x.T, g)
        # if np.any(x.T[0] < 0):
        # if np.abs(x[-1,1]) > 0.1:
        #     continue 
        plt.plot(x.T[0], x.T[1], 'b')

    v0 = np.linspace(-2.3, 0, 500)
    z0_min = 0.5*v0**2/(1-g)  # For constant mass system, (with constant max control) this is a necessary condition 
    good = X0[0] - 0.5*X0[1]**2/(1-g) >= 0 
    bad = np.logical_not(good)
    plt.plot(X0[0][good], X0[1][good], 'ko', label="Initial Condition")
    plt.plot(X0[0][bad], X0[1][bad], 'ro', label="Initial Condition doomed to fail")
    plt.plot(z0_min, v0, 'k--', label='Separatrix') # the line that divides initial conditions that cannot land softly
    plt.xlabel("Altitude")
    plt.ylabel("Velocity")
    plt.legend()
    plt.show()



def verify_eq():
    import sys
    import matplotlib.pyplot as plt 
    sys.path.append('./')

    from Utils.RK4 import RK4
    # verify my formulae 

    def get_state(x0, t, k, g):
        " Analytical "
        z0,v0,m0,p0 = x0 
        
        m = m0 - k*t

        v = v0 - g*t - m0/k*np.log(1-k*t/m0)
        z = z0 + v0*t - 0.5*g*t**2 + m0/k*(t + (m0/k-t)*np.log(1-k*t/m0))

        p1 = 0.035 
        p2 = 0.1
        C = -m0*p1 + k*p2
        pm = p0 + m0*(C/m - C/m0 - p1*np.log(1-k*t/m0))/k**2  # Correct for u=1 or u = -1

        return np.array([z,v,m, pm]).T

    def get_state_true(x0, t, k, g):
        m0 = x0[2]
        
        def dyn(x,t):
            z,v,m,p = x
            dz = v
            dv = m0/m - g
            dm = -k

            p1 = 0.035 
            p2 = 0.1 - p1*t 
            dpm = m0*p2/(m**2)
            
            return [dz, dv, dm, dpm]
        return RK4(dyn, x0, t,)

    v0 = -120
    Tmax = 4000
    m0 = 1050
    gmax = 1.62
    z0 = 3252.2
    pm0 = 0.012 
    c = m0/Tmax 
    z0 = z0*c
    v0 = v0*c
    g = gmax*c 
    k = Tmax/(290*9.81)*c

    t = np.linspace(0, 45, 3000)
    X = get_state_true([z0,v0,m0,pm0], t, k, g)
    X2 = get_state([z0,v0,m0,pm0], t, k, g)
    X[:,:2] /= c
    X2[:,:2] /= c
    # plt.plot(t, X[:,-1])
    # plt.plot(t, X2[:,-1], 'o')
    plt.semilogy(t, np.abs(X-X2))
    plt.legend(('Alt','Vel','Mass','M Costate'))
    plt.title("Error between numerical integration and analytical solution")
    plt.show()

    t2 = -m0/k * (1 - g - np.sqrt((1-g)**2 - 2*k*v0/m0 ))
    print(get_state_true([z0,v0,m0,pm0], np.linspace(0,t2), k, g)[-1]) # returns scaled vars 

if __name__ == "__main__":
    verify_eq()
    # example()
    # test()
    # test_mc()
    # test_gravity_controller()