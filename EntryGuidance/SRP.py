import numpy as np
from pyaudi import sin, cos
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from Utils.RK4 import RK4
from Utils.trapz import trapz, cumtrapz
from Utils.ProjectedNewton import ProjectedNewton as solveQP
from Utils.Regularize import Regularize
import Utils.DA as da
from pyaudi import gdual_double as gd

from Riccati import SRP as srp_riccati
from Riccati import ASREC,SDREC, SRP_A, SRP_B, SRP_Bu


class SRP_Riccati(object):
    def __init__(self):
 # System info
        self.nx = 6  # number of states
        self.nu = 3    # number of controls

        # Dynamic params
        self.param = [3.71, 2.7e3] # gravity, exit velocity
        self.umax = 70.
        self.umin = 40.

        self.m0 = 8500.
        self.mdry = 6000.

        # Boundary conditions
        self.x0 = np.array([-3200., 400, 2600, 625., -80, -270.])
        self.xf = np.array([0.,0.,10.,0.,0.,0.])


    def boundTime(self):

        x,y,z,u,v,w = self.x0

        # Minimum time based on full thrust toward ground, how long to reach 0 vertical velocity

        # Based on fuel available
        # tmin = -np.log(self.mdry/self.m0)*self.param[1]/self.umax  # The fastest we could burn it all, but not a true lower bound on time
        tmin = (u**2+v**2+w**2)**0.5/(self.umax-self.param[0])
        tmax = -np.log(self.mdry/self.m0)*self.param[1]/self.umin  # The slowest we could burn all of the fuel available
        return tmin,tmax

    def solve(self, tf, N=50, x0=None, guess=None, max_iter=15):
        if x0 is not None:
            self.x0 = x0
        r = self.xf
        R = lambda x: np.eye(3)
        Q = lambda x: np.zeros((6,6))
        S = np.zeros((6,6))
        x,u,K = ASREC(self.x0, t=np.linspace(0,tf,N), A=SRP_A, B=SRP_Bu, C=np.eye(6), Q=Q, R=R, F=S, z=r, tol=1e-2, maxU=70,minU=40,guess=guess,max_iter=max_iter)
        return x,u

    def constraintSatisfaction(self, t, x, u):
        xf = x[-1]

        pos = np.linalg.norm(xf[0:3]-self.xf[0:3])
        vel = np.linalg.norm(xf[3:6]-self.xf[3:6])
        print( "Final position constraint violation: {} m".format(pos))
        print( "Final velocity constraint violation: {} m/s".format(vel))   
        if np.any(x[:,2] < 0):
            print ("Subsurface flight detected")
            pos *= 1000        
        return pos, vel 

    def cost(self,t,u):
        a = np.linalg.norm(u,axis=1)
        return trapz(a**2,t)

    def optimize(self):
        tmin,tmax = self.boundTime()

        tmin = 14
        tmax = 18
        N = 30

        violation = np.linalg.norm(self.x0)
        tbest = 0

        plt.figure(1)

        for iter,tf in enumerate(np.linspace(tmin,tmax,30)):

            # try:
                # guess = {'state': x, 'control':u}
            # except:
                # guess = None

            # x,u = self.solve(tf, N=N, guess=guess)
            x,u = self.solve(tf, N=N)


            cp,cv = self.constraintSatisfaction(0,x,u)
            J = self.cost(np.linspace(0,tf,u.shape[0]),u)
            if (cp + cv) < violation:
                violation = cp + cv
                tbest = tf
                xbest = x
                ubest = u

            plt.figure(1)
            plt.plot(tf,cp,'o')
            plt.plot(tf,cv,'x')
            plt.figure(2)
            plt.plot(tf,J,'^')
            # except:
                # pass 
            
        print( "Best final time found: {}".format(tbest)  )  
        
        try:
            t = np.linspace(0,tbest,N)
            a = np.linalg.norm(ubest,axis=1)

            plt.figure(3)
            plt.plot(t,ubest)

            plt.figure(6)
            plt.plot(t,a)

            plt.figure(4)
            plt.plot(t,xbest[:,0:3])

            plt.figure(5)
            plt.plot(t,xbest[:,3:6])
        except:
            pass

        plt.show()
        return xbest, ubest

class SRP_DDP(object):
    def __init__(self,dim=2):
        # Algorithm settings
        self.maxIter = 100
        self.convergenceTolerance = 1e-12

        # System info
        self.nx = 2*dim  # number of states
        self.nu = dim    # number of controls

        # Dynamic params
        self.param = [3.71, 2.7e3] # gravity, exit velocity
        self.umax = 70
        self.umin = -self.umax


        # Boundary conditions
        if dim == 3:
            self.x0 = [-3200, 2700, 400,625,-60,-270,8500]
            self.xf = [0,0,0,0,0,0]
            self.ul = [self.umin, 0, -np.pi]
            self.uu = [self.umax, 2*np.pi, np.pi]
        else:
            self.dynamics = self.__dynamics2d__
            self.x0 = np.array([-3200.,2600, 625.,-270.])
            self.xl = np.array([-1.e6, -1.e6, -1.e6, -1.e6])
            self.xu = np.array([1.e6, 1.e6, 1.e6, 1.e6])
            self.xf = np.array([0.,0.,0.,0.])
            self.ul = np.array([self.umin, self.umin])
            self.uu = np.array([self.umax, self.umax])

        # Cost function terms
        self.q = np.diag([1]*(self.nx))*0
        self.qf = np.diag([1]*(self.nx))*1
        # self.r = np.eye(self.nu)*1e-7
        self.r = np.diag([1, 1])*1e-4/self.umax
        self.N = 50
        tf = 13.
        self.dt = float(tf)/(self.N-1)
        self.t = np.linspace(0,tf,self.N)
        print( "dt: {} s".format(self.dt))
        
        
    def debug(self, state, control, L=None):

        # g = np.array([self.fx(s,c) for s,c in zip(state,control)])
        # H = [self.fxx(s,c) for s,c in zip(state,control)]

        # plt.plot(time,(g[:,0,0])/self.dh,label='dx/dx')
        # plt.plot(time,(g[:,0,1])/self.dh,label='dx/du')
        # plt.plot(time,(g[:,0,2])/self.dh,label='dx/dw')
        # plt.plot(time,(g[:,0,3])/self.dh,label='dx/dm')
        # plt.legend()

        state = [da.const(s) for s in state]

        plt.figure()
        plt.plot(self.t, state)

        if L is not None:
            L = da.const(L)

            plt.figure()
            plt.plot(self.t, L)

        plt.show()


    def DDP(self):
        u = np.array([[-self.umax*0, 0*self.umax*0.5] for _ in self.t])

        u = np.clip(u, self.ul, self.uu)
        # u = srp_riccati()[:,[0,2]]

        states = ['x','z','u','w']
        controls = ['ax','az']
        all = states+controls

        J = []

        #iterate
        for iter in range(self.maxIter):
            print( "Iteration: {}".format(iter+1))
            u = np.array([da.const(uu,array=False) for uu in u])
            x = self.propagate(u)
            x = np.array([da.const(xx,array=False) for xx in x])


            x = np.array([da.make(val,states,2) for val in x],dtype=gd) # Maybe a vectorized version can work
            u = np.array([da.make(val,controls,2) for val in u],dtype=gd)

            L = self.lagrange(x, u)
            LX = da.jacobian(L, all)                                 # Jacobians wrt x and u simulatenously
            LXX = np.array([da.hessian(l, all) for l in L],dtype=gd) # Hessians wrt to x and u simulatenously
            LN = self.mayer(x[-1])
            J.append(np.sum(da.const(L)) + da.const([LN])[0])
            # if not iter:
                # self.debug(x,u,L)

            if len(J) > 1:
                if np.abs(J[-1]-J[-2]) < self.convergenceTolerance:
                    break
            if iter < 4 or not (iter+1)%10:
                plt.figure(1)
                plt.plot(da.const(x[:,0]), da.const(x[:,1]), '--',label="{}".format(iter))
                plt.title('Altitude vs Distance to Target')

                plt.figure(2)
                plt.plot(da.const(x[:,2]),da.const(x[:,3]), '--',label="{}".format(iter))
                plt.title('Vertical Velocity vs Horizontal Velocity')

                # plt.figure(3)
                # plt.plot(self.t,da.const(x[:,3]), '--',label="{}".format(iter))
                # plt.title('Mass vs time')

                plt.figure(4)
                plt.plot(self.t,da.const(u[:,0]), '--',label="{}".format(iter))
                plt.plot(self.t,da.const(u[:,1]), '--',label="{}".format(iter))
                plt.title('Thrust vs time')

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
                    # print Quu
                    Quu = Regularize(Quu, 1e-3)
                    k[i], Quuf = solveQP(np.zeros((self.nu)), Quu, Qu, ([self.ul-da.const(u[i])], [self.uu-da.const(u[i])]),debug=False)

                    K[i] = -np.linalg.solve(Quuf,Qux)

                    Vx = (Qx-K[i].T.dot(Quu).dot(k[i]))
                    Vxx = (Qxx-K[i].T.dot(Quu).dot(K[i]))

                # Forward correction

                x = np.array([da.const(xx,array=False) for xx in x])
                u = np.array([da.const(uu,array=False) for uu in u])
                x,u = self.update(x,u,k,K)




        plt.figure(1)
        plt.plot(da.const(x[:,0]), da.const(x[:,1]), 'k',label="Final")
        plt.title('Altitude vs Distance to Target')
        plt.legend()

        plt.figure(2)
        plt.plot(da.const(x[:,2]),da.const(x[:,3]),  'k',label="Final")
        plt.title('Vertical Velocity vs Horizontal Velocity')
        plt.legend()

        plt.figure(4)
        plt.plot(self.t,da.const(u[:,0]),  'k',label="Final")
        plt.plot(self.t,da.const(u[:,1]), 'k',label="Final")
        plt.title('Thrust vs time')
        plt.legend()

        plt.figure()
        plt.plot(J,'o')
        plt.title('Cost vs Iteration')
        plt.show()


    def update(self,x,u,k,K):
        step = 1  # Linesearch parameter
        J = self.evalCost(x,u)
        Jnew = J+1
        # print J 
        # print J+1 
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
        return RK4(self.dynamics, state, [0,self.dt],args=(control,))[-1]


    # Define the continuous time dynamics as well as the jacobian and hessian
    def __dynamics2d__(self, state, time, control):

        # Unpack
        g,Ve = self.param
        ax,az = control
        x,z,u,w = state

        return np.array([u, w, ax,az-g])


    # Define the terms of the discrete cost function and all derivatives if possible
    def mayer(self, state):
        return np.dot((state-self.xf).T, np.dot(self.qf, (state-self.xf)))

    def lagrange(self, states, control):
        L = np.array([np.dot((state-self.xf).T, np.dot(self.q, (state-self.xf))) for state in states])
        Lu = np.array([(u).T.dot(self.r).dot(u) for u in control]) #quadratic cost
        # Lu = 2*control[:,0]*self.r[0,0] # Minimum fuel
        return  0.5*(L + Lu)*self.dt

if __name__ == "__main__":
    # srp = SRP_DDP()
    # srp.DDP()

    srp = SRP_Riccati()
    srp.optimize()
