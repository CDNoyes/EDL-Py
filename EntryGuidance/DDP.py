""" Python implementation of differential dynamic programming """

import numpy as np
# from numpy import sin, cos 
from pyaudi import sin, cos
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import sys 
sys.path.append('./Utils/')
from RK4 import RK4
from trapz import trapz, cumtrapz
from ProjectedNewton import ProjectedNewton as solveQP
from Regularize import Regularize 
import DA as da 
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
    
    
    # print Vf
    # print Vx 
    # print Vl 
    # print Vxx 
    # print Vxl 
    # print F[-1]
    
    # Backward propagation of the Value function (via integration or STM methods) and optimization 
    for step in [N-2]:#range(N)[::-1]: # Could consider steping through fullstate instead of currentstate to get better estimates, and only optimize every Nint steps
        H = L[step] + np.dot(Vx[:n].T, F[step]) # Get the current hamiltonian 
        print("H: {}".format(H))
        Hxu = da.gradient(H, allnames)      # Gradient wrt both state and control
        print("H gradient: {}".format(Hxu))
        Hx = Hxu[:n]
        Hu = Hxu[n:n+m]
        Hxxuu  = da.hessian(H,allnames)     # Hessian wrt both state and control
        Hxx = Hxxuu[:n][:n]
        Huu = Hxxuu[n:n+m][0][n:n+m]
        Hxu = Hxxuu[:n][0][n:n+m]
        # Hux = Hxu.T 
        print(Hxxuu)
        print(Hxxuu[n:n+m][0][n:n+m])
        print("State hessian: {}".format(Hxx))
        print("Control hessian: {}".format(Huu))
        l = -np.dot(np.linalg.inv(Huu), Hu.T)
        Vdot = -L + 0.5*np.dot(l.T, np.dot(Huu, l))
        print(Vdot) 
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
        self.maxIter = 20
        self.convergenceTolerance = 1e-12
               
        # System info 
        self.nx = 2*dim + 1  # number of states 
        self.nu = dim    # number of controls 
        
        # Dynamic params 
        self.param = [3.71, 2.7e3] # gravity, exit velocity  
        self.umax = 6.e5 
        self.umin = self.umax #3.5e5 
        
        
        # Boundary conditions 
        self.h0 = 2700.
        self.hf = 0.
        if dim == 3:
            self.x0 = [-3200,400,625,-60,-270,8500]
            self.xf = [0,0,0,0,-15,8500]
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
            self.x0 = np.array([-3200., 625.,-270., 8500.]);
            self.xl = np.array([-1.e6, -1.e6, -1.e6, 0.])
            self.xu = np.array([1.e6, 1.e6, -1., 1.e6])
            self.xf = np.array([0.,0.,-5.,8500.])
            self.ul = np.array([self.umin, 2.1])
            self.uu = np.array([self.umax, 2.97])
            
        # Cost function terms 
        self.q = np.diag([1]*(self.nx-1) + [0])*1e-6*0
        self.qf = np.diag([1]*(self.nx-1) + [0])*1e-5
        # self.r = np.eye(self.nu)*1e-7
        self.r = np.diag([1/self.umax, 1])*1*0
        self.uf = np.array([self.umax, 3*np.pi/4])
        self.N = 30
        self.dh = (self.hf-self.h0)/(self.N-1) 
        self.h = np.linspace(self.h0,self.hf,self.N)
    
            
    
    
    def poly(self):
        
        tf = 9
        Ai = [[tf, tf**2],[tf**2/2, tf**3/3]]
        
        Au = np.concatenate((Ai,np.zeros_like(Ai)),axis=1)
        Al = np.concatenate((np.zeros_like(Ai),Ai),axis=1)
        A = np.concatenate((Au,Al),axis=0)
        B = [self.xf[1]-self.x0[1], self.xf[0]-self.x0[0]-self.x0[1]*tf, self.xf[2]-self.x0[2], self.hf-self.h0-self.x0[2]*tf]
        
        C = np.linalg.solve(A,B)
        
        t = np.linspace(0,tf,self.N)
        
        x = self.x0[0] + self.x0[1]*t + C[0]*0.5*t**2 + C[1]/3.*t**3
        z = self.h0 + self.x0[2]*t + C[2]*0.5*t**2 + C[3]/3.*t**3
        
        u = self.x0[1] + C[0]*t + C[1]*t**2
        w = self.x0[2] + C[2]*t + C[3]*t**2
        
        udot = C[0] + 2*C[1]*t
        wdot = C[2] + 2*C[3]*t
        
        mu = (np.arctan2(wdot + self.param[0], udot))%(np.pi*2)
        
        gam = -(wdot + self.param[0])/(self.param[1]*np.sin(mu))
        m = self.x0[-1]*np.exp(cumtrapz(gam,t))
        T = -m*gam*self.param[1]
        
        
        U = list(zip(T,mu))
        U = np.clip(U,self.ul,self.uu)    
        iter = 0
        X = self.propagate(U)
        t = cumtrapz(1/X[:,self.iw],self.h)

        plt.figure(1)
        plt.plot(X[:,0], self.h, '--',label="{}".format(iter))
        plt.title('Altitude vs Distance to Target')
        
        plt.figure(2)
        plt.plot(X[:,1],X[:,2], '--',label="{}".format(iter))
        plt.title('Vertical Velocity vs Horizontal Velocity')
        
        plt.figure(3)
        plt.plot(t,X[:,3], '--',label="{}".format(iter))
        plt.title('Mass vs time')
        
        
        plt.figure(1)
        plt.plot(x,z)
        plt.figure(2)
        plt.plot(u,w)
        plt.figure(5)
        plt.plot(t,mu)
        plt.plot(t,U[:,1])
        plt.figure(4)
        plt.plot(t,T)
        plt.plot(t,U[:,0])
        plt.show()
        
        return list(zip(T,mu))
        
        
    def debug(self,time, state, control, L=None):
        
        # g = np.array([self.fx(s,c) for s,c in zip(state,control)])
        # H = [self.fxx(s,c) for s,c in zip(state,control)]

        # plt.plot(time,(g[:,0,0])/self.dh,label='dx/dx')
        # plt.plot(time,(g[:,0,1])/self.dh,label='dx/du')
        # plt.plot(time,(g[:,0,2])/self.dh,label='dx/dw')
        # plt.plot(time,(g[:,0,3])/self.dh,label='dx/dm')
        # plt.legend()
        
        state = [da.const(s) for s in state]
        L = da.const(L)
        
        plt.figure()
        plt.plot(time, state)
        
        if L is not None:
            plt.figure()
            plt.plot(time,L)
        
        plt.show()
        
        
    def DDP(self):
        # u = self.poly()
        u = np.array([[self.umax*0.6, 2.9] for _ in self.h])
        
        u = np.clip(u,self.ul,self.uu) 
        
        states = ['x','u','w','m']
        controls = ['T','theta']
        all = states+controls 
        
        J = []
        
        #iterate 
        for iter in range(self.maxIter):
            print("Iteration: {}".format(iter+1))
            u = np.array([da.const(uu,array=False) for uu in u])
            x = self.propagate(u)
            x = np.array([da.const(xx,array=False) for xx in x])
            t = cumtrapz(1/x[:,self.iw],self.h)
            
            x = np.array([da.make(val,states,2) for val in x],dtype=gd) # Maybe a vectorized version can work 
            u = np.array([da.make(val,controls,2) for val in u],dtype=gd)
            
            L = self.lagrange(x,u)
            LX = da.jacobian(L,all)                                 # Jacobians wrt x and u simulatenously 
            LXX = np.array([da.hessian(l,all) for l in L],dtype=gd) # Hessians wrt to x and u simulatenously
            LN = self.mayer(x[-1])
            J.append(np.sum(da.const(L)) + da.const([LN])[0])
            # if not iter:
                # self.debug(t,x,u,L)
            if len(J) > 1:
                if np.abs(J[-1]-J[-2]) < self.convergenceTolerance:
                    break
            if iter < 4 or not (iter+1)%10:            
                plt.figure(1)
                plt.plot(da.const(x[:,0]), self.h, '--',label="{}".format(iter))
                plt.title('Altitude vs Distance to Target')
                
                plt.figure(2)
                plt.plot(da.const(x[:,1]),da.const(x[:,2]), '--',label="{}".format(iter))
                plt.title('Vertical Velocity vs Horizontal Velocity')
                
                plt.figure(3)
                plt.plot(t,da.const(x[:,3]), '--',label="{}".format(iter))
                plt.title('Mass vs time')
            
                plt.figure(4)
                plt.plot(t,da.const(u[:,0]), '--',label="{}".format(iter))
                plt.title('Thrust vs time')
                
                plt.figure(5)
                plt.plot(t,da.const(u[:,1]), '--',label="{}".format(iter))
                plt.title('Thrust angle vs time')
                # if not iter:
                    # plt.show()
                # Final conditions on value function derivs
                Vx = da.gradient(LN,all)[:self.nx]
                Vxx = da.hessian(LN,all)[:self.nx,:][:,:self.nx]


                # Backward propagation 
                k = np.zeros((self.N,self.nu))
                K = np.zeros((self.N,self.nu,self.nx))
                for i in range(self.N)[::-1]:
                    xi = self.step(x[i],u[i])
                    fX = da.jacobian(xi,all)
                    fXX = [da.hessian(xii,all) for xii in xi]
                
                    QX = LX[i] + fX.T.dot(Vx)
                    QXX = LXX[i] + fX.T.dot(Vxx).dot(fX) + np.sum([Vxi*fxxi for Vxi,fxxi in zip(Vx,fXX)],axis=0)
                    QXX = QXX.astype(float)
                    # Get some useful partitions:
                    Qx = QX[:self.nx]
                    Qu = QX[self.nx:]
                    Qxx = QXX[:self.nx,:][:,:self.nx]
                    Qux = QXX[self.nx:,:][:,:self.nx]
                    Quu = QXX[self.nx:,:][:,self.nx:]

                    
                    
                    # Bounded controls:
                    # if np.linalg.cond(Quu) > 100:
                    # print Quu
                    Quu = Regularize(Quu, 1e-7)
                    k[i], Quuf = solveQP(np.zeros((self.nu)), Quu, Qu, ([self.ul-da.const(u[i])], [self.uu-da.const(u[i])]),debug=False)
                    # print np.linalg.cond(Quuf)
                    if np.linalg.cond(Quuf) < 100:
                        # print Quuf
                        K[i] = -np.linalg.solve(Quuf,Qux)/np.linalg.cond(Quuf)
                        
                    Vx = (Qx-K[i].T.dot(Quu).dot(k[i]))
                    Vxx = (Qxx-K[i].T.dot(Quu).dot(K[i]))
                
                # Forward correction
                
                x = np.array([da.const(xx,array=False) for xx in x])
                u = np.array([da.const(uu,array=False) for uu in u])
                x,u = self.update(x,u,k,K)
        
        plt.figure()
        plt.plot(J,'o')
        plt.title('Cost vs Iteration')
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
                # xnew.append(np.clip(self.step(xnew[i],unew[i]),self.xl,self.xu))    
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
            # x.append(self.step(x[-1],control))
        return np.array(x) 
        
    def step(self, state, control):
        # return odeint(self.dynamics, state, [0,self.dh],args=(control,))[-1] # doesn't work with DA 
        return RK4(self.dynamics, state, [0,self.dh],args=(control,))[-1]
        
        
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
        print(-u/(w**2))
        return np.eye(self.nx) + np.array([[0, 1/w, -u/(w**2), 0],
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
        Lu = np.array([(u-self.uf).T.dot(self.r).dot(u-self.uf) for u in control]) #quadratic cost 
        # Lu = 2*control[:,0]*self.r[0,0] # Minimum fuel 
        # return Lu + L
        return  0.5*(L + Lu)*self.dh           # No angle inclusion in the control term 
        
    def lx(self, states, control):    
        return np.array([self.q.dot(state-self.xf)/state[2] -  0*np.dot((state-self.xf).T, np.dot(self.q, (state-self.xf)))/state[2]**2 for state in states])*self.dh
        
    def lxx(self,states,control):  
        return np.array([self.q/state[2] - 0*(self.q.dot(state-self.xf)/state[2] -  np.dot((state-self.xf).T, np.dot(self.q, (state-self.xf)))/state[2]**2)/state[2] for state in states])*self.dh
        
    def lu(self, states, control):   
        return np.array([self.r.dot(u-self.uf)/w for u,w in zip(control,states[:,2])])*self.dh
        
    def lux(self,state,control):
        return np.zeros((self.N,self.nu,self.nx))*self.dh
    
    def luu(self,state,control):
        return np.array([self.r/w for w in state[:,2]])*self.dh
    
    def estimate(self):
        x = np.array([self.x0])
        u = np.array([[5e5,2.9]])
        L = self.lagrange(x,u)
        Lx = self.lx(x,u)
        Lxx = self.lxx([self.x0],[[5e5,2.9]])
        # print L
        # print Lx
        # print Lxx
        
        vars = ['x','u','w','m']
        X = da.make(x[0], vars, 2, array=True)
        Lda = self.lagrange(np.array([X]),u)
        # print Lda
        print("L Gradient errors:")
        print(Lx - da.jacobian(Lda,vars))
        print("L Hessian errors:")
        print(Lxx - da.hessian(Lda[0],vars))
        
        F = self.step(X,u[0])
        Fx = self.fx(x[0],u[0])
        Fxx = self.fxx(x[0],u[0])
        print("Dynamic Jacobian errors:")
        print(Fx - da.jacobian(F, vars))
        print("Dynamic Hessian errors:")
        print(Fxx[0] - da.hessian(F[0], vars))
        
    
# Specific implementations like this may be better than a general formulation like the above. Too much stuff to pass...
class testProblem():
    """ A discrete time 1-d system with 1 control with bounds, and no other hard constraints 
        The cost is quadratic and all derivatives are known analytically 
        
    """
    
    def __init__(self, x0=3, N=31):
        self.x0 = x0 
        self.N = N 
        self.fx = 0.90 + .04/N + 0.5/N*np.cos(list(range(N)))*(1-np.linspace(0,1,N))
        self.fu = 1/np.array(list(range(1,N+1)))**0.025 + 0.5*np.sin(list(range(N)))/np.array(list(range(1,N+1)))**4 # time varying control dynamics 
        self.q = 0 
        self.qf = 10
        self.r = 1/500.
        self.umax = 0.15
        self.convergenceTolerance = 1e-14
        self.maxIter = 50
    
    
    def DDP(self, u=None):
        if u is None:
            u = np.zeros((self.N,)) # Initial guess 
            # u = np.ones((self.N,)) # Infeasible Initial guess 
            # u = np.linspace(0,self.umax,self.N) #linear guess

        u = np.clip(u,-self.umax,self.umax)
        
        J = []
        for iter in range(self.maxIter):
            print("Iteration {}".format(iter))
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
            if iter < 10 or not (iter+1)%10:            
                plt.figure(1)
                plt.plot(list(range(self.N)),x,'--',label="{}".format(iter))
                plt.figure(2)
                plt.plot(list(range(self.N)),u, '--',label="{}".format(iter))
                # plt.figure(4)
                # plt.plot(L, 'o', label="{}".format(iter))
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
                    k[i], Quu = solveQP([0], Quu, Qu, ([-self.umax-u[i]], [self.umax-u[i]]), verbose=False)
                    if not Quu == 0:
                        K[i] = -Qux/Quu

                
                V += -0.5*k[i]*Quu*k[i]
                Vx = Qx-K[i]*Quu*k[i]
                Vxx = Qxx-K[i]*Quu*K[i] 
            
            # Forward correction
            if True:
                x,u = self.correct(x,u,k,K)  # Backtracking line search, simply looks to reduce cost
            else:
                x,u = self.exact_ls(x,u,k,K) # Exact, optimization based line search finds the true minimum stepsize 
            
        plt.figure(1)    
        plt.plot(list(range(self.N)), x,label='Final')
        plt.title('State')
        plt.legend()
        plt.figure(2)
        plt.plot(list(range(self.N)), u, label='Final')
        plt.title('Control')
        plt.legend()
        plt.figure(3)
        plt.semilogy(J,'o')
        plt.title('Cost vs Iteration')
        plt.show()
            
        return u     
            
    def correct(self, x, u, k, K):
    
        step = 1
        J = self.evalCost(x, u)
        Jnew = J+1
        while Jnew > J: # should put a max iteration limit as well 
            xnew = [self.x0]
            unew = []
            for i in range(self.N-1):
                unew.append(np.clip(u[i] + step*k[i] + K[i]*(xnew[i]-x[i]),-self.umax,self.umax)) # This line search is actually essential to convergence 
                xnew.append(self.transition(xnew[i],unew[i],i))    
            unew.append(unew[-1]) # Just so it has the correct number of elements   
            Jnew = self.evalCost(np.array(xnew), np.array(unew).squeeze())
            step *= 0.8
        return np.array(xnew), np.array(unew).squeeze()
        
    def exact_ls(self, x, u, k, K):
        from scipy.optimize import minimize_scalar

        def fun(step):
            xnew = [self.x0]
            unew = []
            for i in range(self.N-1):
                unew.append(np.clip(u[i] + step*k[i] + K[i]*(xnew[i]-x[i]),-self.umax,self.umax)) # This line search is actually essential to convergence 
                xnew.append(self.transition(xnew[i],unew[i],i))    
            unew.append(unew[-1]) # Just so it has the correct number of elements   
            return self.evalCost(np.array(xnew), np.array(unew).squeeze())

        sol = minimize_scalar(fun, bounds=(0,1), method="bounded", tol=1e-3)
        step = sol.x 

        xnew = [self.x0]
        unew = []
        for i in range(self.N-1):
            unew.append(np.clip(u[i] + step*k[i] + K[i]*(xnew[i]-x[i]),-self.umax,self.umax)) # This line search is actually essential to convergence 
            xnew.append(self.transition(xnew[i],unew[i],i))    
        unew.append(unew[-1]) # Just so it has the correct number of elements   

        return np.array(xnew), np.array(unew).squeeze()

        
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
        return self.qf*(x-1)**2      # Drive to x = 1
    
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
    
    # srp = SRP()
    # srp.estimate()
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