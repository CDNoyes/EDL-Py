""" Python implementation of differential dynamic programming """

import numpy as np
from numpy import sin, cos 
from scipy.integrate import odeint 
from scipy.interpolate import interp1d 
import matplotlib.pyplot as plt 

from HPC import profile 
from Simulation import Simulation 
from InitialState import InitialState 

# from Utils.RK4 import RK4
from Utils.trapz import trapz, cumtrapz
from Utils.ProjectedNewton import ProjectedNewton as solveQP
from Utils.Regularize import Regularize 
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
        F.append(dynamics(state, 0, 0.25, control))                         # Need to call the dynamics but these aren't generic
    
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
    
    
class SRP(object):
    def __init__(self,dim=2):
        # Algorithm settings 
        self.maxIter = 3 
        self.convergenceTolerance = 1e-3 
               
        # System info 
        self.nx = 2*dim  # number of states 
        self.nu = dim    # number of controls 
        
        # Dynamic params 
        self.param = [3.71, 2.7e3] # gravity, exit velocity  
        self.umax = 6e5 
        self.umin = 3.5e5 
        
        
        # Boundary conditions 
        self.h0 = 2700 
        self.hf = 15 
        if dim == 3:
            self.x0 = [-3200,400,625,-60,-270,8500]
            self.xf = [0,0,0,0,-5,8500]
            self.ul = [self.umin, 0, -np.pi]
            self.uu = [self.umax, 2*np.pi, np.pi]
            self.iw = 5
        else:
            self.dynamics = self.__dynamics2d__
            self.fx = self.__fx2d__
            self.fxx = self.__fxx2d__
            self.fxu = self.__fxu2d__
            self.fu = self.__fu2d__
            self.fuu = self.__fuu2d__
            self.iw = 2 # Index of the vertical velocity, which plays in an important role in converting terms to altitude as IV
            self.x0 = np.array([-3200, 625,-270, 8500]);
            self.xl = np.array([-1e6, -1e6, -1e6, 0])
            self.xu = np.array([1e6, 1e6, -1, 1e6])
            self.xf = np.array([0,0,-5,8500])
            self.ul = np.array([self.umin, 0])
            self.uu = np.array([self.umax, 2*np.pi])
            
        # Cost function terms 
        self.q = np.diag([1]*(self.nx-1) + [0])*1e-7
        self.qf = np.diag([1]*(self.nx-1) + [0])*1e-4
        # self.r = np.eye(self.nu)*1e-7
        self.r = np.diag([1/self.umax, 1])*1e-5
        self.uf = np.array([self.umax, 1.5*np.pi])
        self.N = 50
        self.dh = (self.hf-self.h0)/(self.N-1) 
        self.h = np.linspace(self.h0,self.hf,self.N)
        
    # def compare(self):
        # u = np.array([[self.umax, 2.69] for _ in self.h])
        # x = self.propagate(u)
        
    def debug(self,state,control):
        
        g = np.array([self.fx(s,c) for s,c in zip(state,control)])
        # H = [self.fxx(s,c) for s,c in zip(state,control)]
        print g.shape
        plt.semilogy(self.h,np.abs(g[:,0,0]),label='dx/dx')
        plt.semilogy(self.h,np.abs(g[:,0,1]),label='dx/du')
        plt.semilogy(self.h,np.abs(g[:,0,2]),label='dx/dw')
        plt.semilogy(self.h,np.abs(g[:,0,3]),label='dx/dm')
        plt.legend()
        plt.show()
        
        
    def DDP(self):
        u = np.array([[self.umax*0.5, 2.69] for _ in self.h])
            
        J = []
        
        #iterate 
        iter = 0 
        x = self.propagate(u)
        # self.debug(x,u)
        L = self.lagrange(x,u)
        Lx = self.lx(x,u)
        Lxx = self.lxx(x,u)
        Lux = self.lux(x,u)
        Lu = self.lu(x,u)
        Luu = self.luu(x,u)
        LN = self.mayer(x[-1])
        J.append(np.sum(L) + LN)
        
        plt.plot(self.h,L)
        plt.figure()
        plt.plot(self.h,Lx)
        plt.figure()
        plt.plot(self.h,Lu)
        plt.show()
        
        # if len(J) > 1:
            # if np.abs(J[-1]-J[-2]) < self.convergenceTolerance:
                # break
        # if iter < 4 or not (iter+1)%10:            
        # plt.figure(1)
        # plt.plot(x[:,0], self.h, '--',label="{}".format(iter))
        # plt.title('Altitude vs Distance to Target')
        
        # plt.figure(2)
        # plt.plot(x[:,1],x[:,2], '--',label="{}".format(iter))
        # plt.title('Vertical Velocity vs Horizontal Velocity')
        
        # plt.figure(3)
        # plt.plot(self.h,x[:,3], '--',label="{}".format(iter))
        # plt.title('Mass vs altitude')
    
        # plt.figure(4)
        # plt.plot(self.h,u[:,0], '--',label="{}".format(iter))
        # plt.title('Thrust vs altitude')
    
        # plt.show()
        
        # Final conditions on Value function and its derivs
        # V = LN  # Unused?
        Vx = self.mx(x[-1])
        Vxx = self.mxx(x[-1])
        
        # Backward propagation
        k = np.zeros((self.N,self.nu))
        K = np.zeros((self.N,self.nu,self.nx))
        for i in range(self.N)[::-1]:
            # print self.fx(x[i],u[i])
            Qx  = Lx[i]  + self.fx(x[i],u[i]).T.dot(Vx) 
            Qu  = Lu[i]  + self.fu(x[i],u[i]).T.dot(Vx)
            Qxx = Lxx[i] + self.fx(x[i],u[i]).T.dot(Vxx).dot(self.fx(x[i],u[i])) + 1*np.sum([Vxi*fxxi for Vxi,fxxi in zip(Vx,self.fxx(x[i],u[i]))],axis=0)
            Qux = Lux[i] + self.fu(x[i],u[i]).T.dot(Vxx).dot(self.fx(x[i],u[i])) + 1*np.sum([Vxi*fxui.T for Vxi,fxui in zip(Vx,self.fxu(x[i],u[i]))],axis=0) # Fxu is transposed because we need Fux 
            Quu = Luu[i] + self.fu(x[i],u[i]).T.dot(Vxx).dot(self.fu(x[i],u[i])) + 1*np.sum([Vxi*fuui for Vxi,fuui in zip(Vx,self.fuu(x[i],u[i]))],axis=0)

            # No control limits 
            if False:
                Quu = Regularize(Quu, 1)
                k[i] = -np.dot(np.linalg.inv(Quu),Qu)
                K[i] = -np.dot(np.linalg.inv(Quu),Qux)
            # Bounded controls:
            else:
                print Qu 
                print Quu 
                Quu = Regularize(Quu, 1e-5)
                k[i],Quuf = solveQP(np.zeros((self.nu)), Quu, Qu, ([self.ul-u[i]], [self.uu-u[i]]),debug=False)
                K[i] = -np.linalg.inv(Quuf).dot(Qux) 

            # V += -0.5*k[i].T.dot(Quu).dot(k[i]) # Unused? 
            Vx = (Qx-K[i].T.dot(Quu).dot(k[i]))
            Vxx = (Qxx-K[i].T.dot(Quu).dot(K[i]))
        
        # Forward correction
        x,u = self.update(x,u,k,K)
        
        plt.figure(1)
        plt.plot(x[:,0], self.h, '--',label="{}".format(iter))
        plt.title('Altitude vs Distance to Target')
        
        plt.figure(2)
        plt.plot(x[:,1],x[:,2], '--',label="{}".format(iter))
        plt.title('Vertical Velocity vs Horizontal Velocity')
        
        plt.figure(3)
        plt.plot(self.h,x[:,3], '--',label="{}".format(iter))
        plt.title('Mass vs altitude')
    
        plt.figure(4)
        plt.plot(self.h,u[:,0], '--',label="{}".format(iter))
        plt.title('Thrust vs altitude')
    
        plt.show()
        
        
    def update(self,x,u,k,K):
        step = 1  # Linesearch parameter 
        J = self.evalCost(x,u)
        Jnew = J+1
        while Jnew > J: # should put a max iteration limit as well 
            xnew = [self.x0]
            unew = []
            for i in range(self.N-1):
                unew.append(np.clip(u[i] + step*k[i] + K[i].dot(xnew[i]-x[i]),self.ul,self.uu)) # This line search is actually essential to convergence 
                xnew.append(self.step(xnew[i],unew[i]))    
            unew.append(unew[-1]) # Just so it has the correct number of elements   
            Jnew = self.evalCost(np.array(xnew),np.array(unew))
            step *= 0.8
        return np.array(xnew),np.array(unew)
        
        
    def evalCost(self,x,u):
            L = self.lagrange(x,u)
            LN = self.mayer(x[-1])
            return sum(L) + LN 
        
    def propagate(self,controls):
        x = [self.x0]
        for control in controls[:-1]:
            x.append(np.clip(self.step(x[-1],control),self.xl,self.xu))
        return np.array(x) 
        
    def step(self, state, control):
        return odeint(self.dynamics, state, [0,self.dh],args=(control,))[-1]
        
        
    # Define the continuous time dynamics as well as the jacobian and hessian     
    def __dynamics2d__(self, state, altitude, control):
        
        # Unpack 
        g,Ve = self.param 
        T,theta = control 
        x,u,w,m = state 
        
        return np.array([u/w, T*cos(theta)/(m*w), (T*sin(theta)/m - g)/w, -T/(Ve*w)])
        
    def __fx2d__(self, state, control):
        # Unpack 
        g,Ve = self.param 
        T,theta = control 
        x,u,w,m = state 
        return np.eye(self.nx) + np.array([[0, 1/w, -u/w**2, 0],
                         [0, 0, -T*cos(theta)/(m*w**2), -T*cos(theta)/(m**2*w)],
                         [0, 0, -(T*sin(theta)/m - g)/w**2, -T*sin(theta)/(m**2*w)],
                         [0, 0, T/(Ve*w**2), 0]])*self.dh
                
    def __fxx2d__(self, state, control):
        # Unpack 
        g,Ve = self.param 
        T,theta = control 
        x,u,w,m = state 

        return np.array([[[0, 0, 0, 0], [0, 0, -1/w**2, 0], [0, -1/w**2, 2*u/w**3, 0], [0, 0, 0, 0]],
                        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 2*T*cos(theta)/(m*w**3), T*cos(theta)/(m**2*w**2)], [0, 0, T*cos(theta)/(m**2*w**2), 2*T*cos(theta)/(m**3*w)]],
                        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 2*(T*sin(theta)/m - g)/w**3, T*sin(theta)/(m**2*w**2)], [0, 0, T*sin(theta)/(m**2*w**2), 2*T*sin(theta)/(m**3*w)]],
                        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, -2*T/(Ve*w**3), 0], [0, 0, 0, 0]] ])*self.dh
    
    def __fxu2d__(self, state, control):
        # Unpack 
        g,Ve = self.param 
        T,theta = control 
        x,u,w,m = state 

        return np.array([[[0, 0], [0, 0], [0, 0], [0, 0]],
                        [[0, 0], [0, 0], [-cos(theta)/(m*w**2), T*sin(theta)/(m*w**2)], [-cos(theta)/(m**2*w), T*sin(theta)/(m**2*w)]],
                        [[0, 0], [0, 0], [-sin(theta)/(m*w**2), -T*cos(theta)/(m*w**2)], [-sin(theta)/(m**2*w), -T*cos(theta)/(m**2*w)]],
                        [[0, 0], [0, 0], [1/(Ve*w**2), 0], [0, 0]]])*self.dh
    
    def __fu2d__(self, state, control):
        # Unpack 
        g,Ve = self.param 
        T,theta = control 
        x,u,w,m = state 

        return np.array([[0, 0],
                        [cos(theta)/(m*w), -T*sin(theta)/(m*w)],
                        [sin(theta)/(m*w), T*cos(theta)/(m*w)],
                        [-1/(Ve*w), 0]])*self.dh            
    
    def __fuu2d__(self, state, control):
        # Unpack 
        g,Ve = self.param 
        T,theta = control 
        x,u,w,m = state 

        return np.array(  [ [[0, 0], [0, 0]],
                            [[0, -sin(theta)/(m*w)], [-sin(theta)/(m*w), -T*cos(theta)/(m*w)]],
                            [[0, cos(theta)/(m*w)], [cos(theta)/(m*w), -T*sin(theta)/(m*w)]],
                            [[0, 0], [0, 0]] ] )*self.dh
    
    
    # Define the terms of the discrete cost function and all derivatives if possible 
    def mayer(self, state):
        return np.dot((state-self.xf).T, np.dot(self.qf, (state-self.xf)))
        
    def mx(self, state):
        return np.dot(self.qf, (state-self.xf))
        
    def mxx(self, state):
        return self.qf 
            
    def lagrange(self, states, control):
        L = np.array([np.dot((state-self.xf).T, np.dot(self.q, (state-self.xf))) for state in states])
        Lu = np.array([(u-self.uf).T.dot(self.r).dot(u-self.uf) for u in control])
        return  (L + Lu)*self.dh/states[:,2]           # No angle inclusion in the control term 
        
    def lx(self, states, control):    
        return np.array([self.q.dot(state-self.xf)/state[2] -  np.dot((state-self.xf).T, np.dot(self.q, (state-self.xf)))/state[2]**2 for state in states])*self.dh
        
    def lxx(self,states,control):  
        return np.array([self.q/state[2] - 2*(self.q.dot(state-self.xf)/state[2] -  np.dot((state-self.xf).T, np.dot(self.q, (state-self.xf)))/state[2]**2)/state[2] for state in states])*self.dh
        
    def lu(self, states, control):   
        return np.array([self.r.dot(u-self.uf)/w for u,w in zip(control,states[:,2])])*self.dh
        
    def lux(self,state,control):
        return np.zeros((self.N,self.nu,self.nx))*self.dh
    
    def luu(self,state,control):
        return np.array([self.r/w for w in state[:,2]])*self.dh
            
    
# Specific implementations like this may be better than a general formulation like the above. Too much stuff to pass...
class testProblem():
    """ A discrete time 1-d system with 1 control with bounds, and no other hard constraints 
        The cost is quadratic and all derivatives are known analytically 
        
    """
    
    def __init__(self, x0=3, N=31):
        self.x0 = x0 
        self.N = N 
        self.fx = 0.99 + .04/N + 0.5/N*np.cos(range(N))*(1-np.linspace(0,1,N))
        self.fu = 1/np.array(range(1,N+1))**0.025 + 0.5*np.sin(range(N))/np.array(range(1,N+1))**4 # time varying control dynamics 
        self.q = 0 
        self.qf = 1
        self.r = 1/5000.
        self.umax = 0.25
        self.convergenceTolerance = 1e-6
        self.maxIter = 500
    
    
    def DDP(self, u=None):
        if u is None:
            u = np.zeros((self.N,)) # Initial guess 
            # u = np.ones((self.N,)) # Infeasible Initial guess 
            # u = np.linspace(0,self.umax,self.N) #linear guess

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
                Qx = Lx[i] + self.fx[i]*Vx 
                Qu = Lu[i] + self.fu[i]*Vx
                Qxx = Lxx[i] + self.fx[i]*Vxx*self.fx[i]  + Vx*self.fxx(x[i],u[i])
                Qux = Lux[i] + self.fu[i]*Vxx*self.fx[i]  + Vx*self.fux(x[i],u[i])
                Quu = Luu[i] + self.fu[i]*Vxx*self.fu[i]  + Vx*self.fuu(x[i],u[i])
                
                # No control limits 
                if False:
                    k[i] = (-Qu/Quu) # Shouldnt be any divide by zero error here 
                    K[i] = (-Qux/Quu)
                # Bounded controls:
                else:
                    if Quu < 0:
                        Quu = 0.1 
                    k[i],Quuf = solveQP([0], [[Quu]], [Qu], ([-self.umax-u[i]], [self.umax-u[i]]),debug=False)
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
            
        return u     
            
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
            step *= 0.8
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
        return self.fx[N]*x + self.fu[N]*u 
   
        
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
    uOpt = system.DDP()
    # uNew = system.DDP(uOpt)
    
    # srp = SRP()
    # uSRP = srp.DDP()
    
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