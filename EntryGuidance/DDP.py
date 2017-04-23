""" Python implementation of differential dynamic programming """

import numpy as np
from scipy.integrate import odeint 
from scipy.interpolate import interp1d 
import matplotlib.pyplot as plt 

from HPC import profile 
from Simulation import Simulation 
from InitialState import InitialState 

# from Utils.RK4 import RK4 as odeint 
from Utils.trapz import trapz, cumtrapz
from Utils.ProjectedNewton import ProjectedNewton as solveQP
import Utils.DA as da 
from pyaudi import gdual_double as gd 

def DDP(simulation, dynamics, mayerCost, lagrangeCost, finalConstraints, finalTime, initialState, controlDim, initialCostate, controlStepsize, costateStepsize, args=None):
    """ Continuous time DDP with terminal constraints (and in the future, control constraints as well)

        simulation(x0, tf, u, args)     - a trajectory simulator from 0 to finalTime using an interpolation of the controls
        dynamics                        - equations of motion of the n states subject to m controls with respect to the independent variable 
        mayerCost(xf)                   - portion of the total cost dependent on final states 
        lagrangeCost(t,x,u)             - running cost accumulated over the trajectory 
        finalConstraints(xf)            - a vector function of constraints on the final state of length p 
        finalTime                       - the final value of the independent variable (which need not be time)
        initialState                    - the fixed initial state of length n 
        controlDim                      - control dimension m
        initialCostate                  - an initial guess at the lagrange multipliers (generally all zeros will suffice)
        controlStepsize                 - a stepsize scale factor for the control update 
        costateStepsize                 - a stepsize scale factor the langrange multipler update 

    """
    
    # Get problem dimensions 
    N = 20                                      # The number of discretized segments in the trajectory 
    Nint = 5                                    # Number of integration steps between each N segments
    n = len(initialState)
    m = controlDim 
    p = len(finalConstraints(initialState))
    
    time = np.linspace(0, finalTime, N)
    dt = time[1]
    varnames = ['var{}'.format(name) for name in range(n)]
    controlnames = ['cvar{}'.format(name) for name in range(m)]
    allnames = varnames+controlnames 
    
    if mayerCost is None:
        mayerCost = lambda xf: 0 
        
    if lagrangeCost is None:
        lagrangeCost = lambda t,x,u: 0 
    if m==1:
        currentControl = [0]*N#np.zeros((N,))
    else:    
        currentControl = [[0]*m]*N #np.zeros((N,controlDim))
        
    currentCostate = initialCostate 
    
    # Integrate the system forward to get the full trajectory with current control 
    currentState = [da.make(initialState,varnames,2)]
    # currentState = [initialState]
    fullState = currentState[:] # keep a copy of the state with integrator steps 
    
    # Take a step with the initial control as a DA in order to get control hessians as well 
    currentControl[0] = gd(float(currentControl[0]),controlnames[0],2)            
    
    # Remaining propagation:
    for simTime, control in zip(np.diff(time), currentControl): # Notice we're only stepping by dt, have to be careful if anything interpolates against time 
        fullState.append(simulation(currentState[-1], control, simTime,  N=Nint)[1:])  # N=2 corresponds to taking a single fourth order step from t to t+dt, 
        currentState.append(fullState[-1][-1])                                                                        # accuracy can be enhanced by making N bigger without changing the number of control segments 
    
    currentState = np.array(currentState)
    fullState = np.asarray(np.vstack(fullState))

    if 0: # Debug plots
        if n > 2:
            plt.plot(time, [da.const(fs) for fs in fullState])
        else:
            plt.plot(da.const(fullState[:,0]), da.const(fullState[:,1]))
            
        plt.show()
        
    finalState = currentState[-1]
    
    # Get the first and second partials for every point along the trajectory 
    F = []
    for state, control in zip(currentState,currentControl):
        if not isinstance(control, gd):
            control = gd(control,controlnames[0],2)
        F.append(dynamics(state, 0,0.25, control))                         # Need to call the dynamics but these aren't generic
    
    # Estimate the cost along the current trajectory/control 
    L = lagrangeCost(time,currentState,currentControl)
    
    # Compute terminal Value function and derivatives
    Vf = mayerCost(finalState) + np.dot(finalConstraints(finalState), currentCostate) 
    Vx = da.gradient(Vf, allnames)
    Vxx = da.hessian(Vf, allnames)
    Vll = np.zeros((p,p))
    
    Vl =  finalConstraints(finalState)  # Only need the constant part here, but derivatives are used in backward integration  
    Vxl = da.jacobian(Vl, allnames)                            
    Vf = da.const([Vf])[0]                                  # Get const part after differentiating for Vx, Vxx 
    Vl = da.const(Vl)
    
    
    print Vf
    print Vx 
    print Vl 
    print Vxx 
    print Vxl 
    print F[-1]
    
    # Backward propagation of the Value function (via integration or STM methods) and optimization 
    for step in [N-2]:#range(N)[::-1]: # Could consider steping through fullstate instead of currentstate to get better estimates, and only optimize every Nint steps
        H = L[step] + np.dot(Vx[:n].T, F[step]) # Get the current hamiltonian 
        print "H: {}".format(H)
        Hxu = da.gradient(H, allnames)      # Gradient wrt both state and control
        print "H gradient: {}".format(Hxu)
        Hx = Hxu[:n]
        Hu = Hxu[n:n+m]
        Hxxuu  = da.hessian(H,allnames)     # Hessian wrt both state and control
        Hxx = Hxxuu[:n][:n]
        Huu = Hxxuu[n:n+m][0][n:n+m]
        Hxu = Hxxuu[:n][0][n:n+m]
        # Hux = Hxu.T 
        print Hxxuu
        print Hxxuu[n:n+m][0][n:n+m]
        print "State hessian: {}".format(Hxx)
        print "Control hessian: {}".format(Huu)
        l = -np.dot(np.linalg.inv(Huu), Hu.T)
        Vdot = -L + 0.5*np.dot(l.T, np.dot(Huu, l))
        print Vdot 
        Vnew = Vf - Vdot*dt  # Try to estimate this same value by STM 
        
        
    # Get update to costate 
    
    
    
    # Get upate to control. In unconstrained case, given by formula. In constrained case, obtained via constrained QP solution 
    
    
    
    # Update the control and costate and begin new iteration
    
    
def valueDynamics():
    ''' EoM for the value function '''
    pass 
    
    
    
    
    
class testProblem():
    """ A discrete time 1-d system with 1 control with bounds, and no other hard constraints 
        The cost is quadratic and all derivatives are known analytically 
        
    """
    
    def __init__(self, x0=3, N=16):
        self.x0 = x0 
        self.N = N 
        # self.fx = 1.005 # slightly unstable 
        self.fx = 0.9 + .05/N
        self.fu = 1/np.array(range(1,N+1))**0.25 + 0.5*np.sin(range(N))/np.array(range(1,N+1))**4 # time varying control dynamics 
        self.q = 1.0
        self.qf = 50.
        self.r = 1e-6*0
        self.umax = 0.25
        self.convergenceTolerance = 1e-5
        self.maxIter = 500
        
    def DDP(self):
        u = np.zeros((self.N,)) # Initial guess 
        # u = np.ones((self.N,)) # Infeasible Initial guess 
        # r = np.array(range(self.N))
        # u[:] = -self.umax
        # u[r>self.N/4.] = self.umax*2/3.
        u = np.clip(u,-self.umax,self.umax)
        plt.figure()
        J = []
        for iter in range(self.maxIter):
            print "Iteration {}".format(iter)
            # Forward propagation 
            x = self.forward(self.x0, u, self.N)
            L = self.lagrange(x,u)
            Lx = self.lx(x,u)
            Lxx = self.lxx(x,u)
            Lux = self.lux(x,u)
            Lu = self.lu(x,u)
            Luu = self.luu(x,u)
            LN = self.mayer(x[-1])
            J.append(np.sum(L) + LN)
            if len(J) > 1:
                if np.abs(J[-1]-J[-2]) < self.convergenceTolerance:
                    break
            if iter < 4 or not (iter+1)%10:            
                plt.figure(1)
                plt.plot(range(self.N),x,'--',label="{}".format(iter))
                plt.figure(2)
                plt.plot(range(self.N),u, '--',label="{}".format(iter))
            
            # Final conditions on Value function and its derivs
            V = LN  
            Vx = self.mx(x[-1])
            Vxx = self.mxx(x[-1])
            
            # Backward propagation
            k = [0]*self.N
            K = [0]*self.N
            for i in range(self.N)[::-1]:
                Qx = Lx[i] + self.fx*Vx 
                Qu = Lu[i] + self.fu[i]*Vx
                Qxx = Lxx[i] + self.fx*Vxx*self.fx  + Vx*self.fxx(x[i],u[i])
                Qux = Lux[i] + self.fu[i]*Vxx*self.fx  + Vx*self.fux(x[i],u[i])
                Quu = Luu[i] + self.fu[i]*Vxx*self.fu[i]  + Vx*self.fuu(x[i],u[i])
                
                # No control limits 
                # k[i] = (-Qu/Quu) # Shouldnt be any divide by zero error here 
                # K[i] = (-Qux/Quu)

                k[i],Quuf,fopt = solveQP([0], [[Quu]], [Qu], ([-self.umax-u[i]], [self.umax-u[i]]))
                if len(Quuf):
                    Quuf = Quuf.flatten()[0] 
                    if Quuf != 0: 
                        K[i] = -Qux/Quuf
                else:
                    K[i] = 0
                
                V += -0.5*k[i]*Quu*k[i]
                Vx = Qx-K[i]*Quu*k[i]
                Vxx = Qxx-K[i]*Quu*K[i] 
            
            # Forward correction
            x,u = self.correct(x,u,k,K)
            
        plt.figure(1)    
        plt.plot(range(self.N), x,label='Final')
        plt.title('State')
        plt.legend()
        plt.figure(2)
        plt.plot(range(self.N), u, label='Final')
        plt.title('Control')
        plt.legend()
        plt.figure(3)
        plt.plot(J,'o')
        plt.title('Cost vs Iteration')
        plt.show()
            
    def correct(self, x, u, k, K):
    
        step = 1
        J = self.evalCost(x,u)
        Jnew = J+1
        while Jnew > J: # should put a max iteration limit as well 
            xnew = [self.x0]
            unew = []
            for i in range(self.N-1):
                unew.append(np.clip(u[i] + step*k[i] + K[i]*(xnew[i]-x[i]),-self.umax,self.umax)) # This line search is actually essential to convergence 
                xnew.append(self.transition(xnew[i],unew[i],i))    
            unew.append(unew[-1]) # Just so it has the correct number of elements   
            Jnew = self.evalCost(np.array(xnew),np.array(unew).flatten())
            step *= 0.7
        return np.array(xnew),np.array(unew).flatten()
        
        
    def evalCost(self,x,u):
            L = self.lagrange(x,u)
            LN = self.mayer(x[-1])
            return sum(L) + LN 
            
            
    def forward(self, x0, u, N):
        x = [x0]
        for i in range(N-1):
            x.append(self.transition(x[-1],u[i],i))
        return np.array(x)
        

    def transition(self,x,u,N):
        return self.fx*x + self.fu[N]*u 
   
        
    def fxx(self,x,u):
        return 0 
        
    def fuu(self,x,u):
        return 0 
        
    def fux(self,x,u):
        return 0
    
    def mayer(self,x):
        return self.qf*(x-1)**2      # Focus on stabilizing the state to the origin 
    
    def mx(self,x):
        return 2*self.qf*(x-1) 
        
    def mxx(self,x):
        return 2*self.qf
        
    def mu(self,x):
        return 0 
        
    def muu(self,x):
        return 0 
        
    def lagrange(self,x,u):
        return self.q*(x-1)**2 + self.r*u**2 
        
    def lx(self,x,u):
        return 2*self.q*(x-1) 
        
    def lxx(self,x,u):
        return np.ones_like(x)*2*self.q
        
    def lux(self,x,u):
        return np.zeros_like(x)
    
    def lu(self,x,u):
        return 2*self.r*u 
        
    def luu(self,x,u):
        return 2*self.r*np.ones_like(u) 
    
    def Q(self,l,V):  # Pseudo-Hamiltonian, i.e. its discrete analogue 
        return l + V
        
    
if __name__ == "__main__":
    
    
    # from Misc.VDP import VDP 
    # from Misc.test_pyaudi import VDP        # A much simpler VDP implementation 
    # vdp = VDP()
    # mCost = None 
    # lCost = lambda t,x,u: cumtrapz(u,t)
    # fCon = lambda x: x 
    # x0 = [3,3]
    # DDP(vdp.simulate, vdp.__dynamics__, mCost, lCost, fCon, 5, x0, 1, [0,0], 1, 1)
    
    
    
    # Simplest test problem imaginable 
    system = testProblem()
    system.DDP()
    
    # Confirm backward integration 
    # dx = lambda x: -x 
    # t = np.linspace(0,5,50)
    # x0 = 2
    # y = x0*np.exp(-t) # True 
    # yf = [x0]
    # yb = [y[-1]]
    # dt = np.diff(t)
    # for delta in dt:
        # yf.append(yf[-1] + delta*dx(yf[-1]))
        # yb.append(yb[-1] + delta*-dx(yb[-1]))
    # plt.plot(t,y,'--')
    # plt.plot(t,yf)
    # plt.plot(t,yb[::-1])
    # plt.show()