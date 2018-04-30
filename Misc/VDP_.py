import numpy as np
import chaospy as cp
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from Utils.RK4 import RK4 as odeint # About twice as slow as odeint but capable of vectorization

import time


class VDP(object):
    ''' A van der pol oscillator class.
            This serves as a simple 2D testbed for research in dynamic systems.
            The system is linear when the parameter mu is 0.
            
        The following robust controllers are implemented for comparison.
            Controller 1 is defined by the reference "A Continuous Asymptotic Tracking Control Strategy for Uncertain Nonlinear Systems"
            Controller 2 is defined by the reference "Dynamic Controller Design for a Class of Nonlinear Uncertain Systems subjected to Time-Varying Disturbance"
            
        Some conclusions:
            In the presence of time-varying disturbances, both controllers perform well.
            With a state-dependent term multiplying the control, both controllers are satisfactory but C1 provides more robust performance.
            
            Controller 1 is simpler to tune and seems to provide similarly satisfactory responses for a variety of gain settings.
            Controller 2 is not as easy to tune but when tuned correctly provides qualitatively similar responses as the first controller. 
            
            C1 seems to provide better performance overall, including when the control is non-affine. 
            
            
        Sequential Action Control - WIP     
            
    '''
    def __init__(self):
        print("VDP created.")
        self.reset()
        
    def reset(self):    
        self.samples=None 
        self.outputs=None 
        self.time = None 
        self.controller = None 
        self.controls = []
        
        
    def ref(self, t):
        k1 = 1
        a = 0. 
        k2 = -3 
        return np.array([np.sin(k1*t)+a*np.sin(k2*t),k1*np.cos(k1*t)+a*k2*np.cos(k2*t)])
        # return np.array([.1*np.cos(t) + np.sin(t)/(1+0.1*t), np.cos(t)/(1+t) - np.sin(t)/((1+0.1*t)**2)])
        # return np.array([np.zeros_like(t),np.zeros_like(t)])
        
    def __dynamics__(self, x, t, mu, e0):
        """ System dynamics, including robust controller """
        
        if self.controller ==  1:
            # Gains
            k       = 50
            alpha   = 0.6   # > 0.5 
            beta    = 5 
            
            # Current error 
            e2 = self.error(x,t)
            
            # Control computation 
            xd = self.ref(t)
            uff = xd[0] - (mu)*(1-xd[0]**2)*xd[1] # Feed-forward term, not required for good tracking but can improve performance. Half the true value of mu is used in this model.
            u = k*(e2-e0) + x[2] + uff*0
            du = k*alpha*e2 + beta*np.sign(e2)
            
        elif self.controller ==  2:
            s,ce2 = self.error(x,t)
            # PI gains
            kp = 10 
            ki = 20     # Must satisfy Kp**2 > 4Ki
            k = 1       # Gain for the nonlinear element of the integral term  
            u = kp*s + ki*x[2] + ce2
            du = s + k*np.sign(s)
        
        # if t > 5 and t < 6 :
        # u = np.clip(u,-1.6,1.8)                               # Apply control limitations if desired 
        # u *= 0                                            # Uncontrolled system 
        self.controls.append(u)
        
        # Add an external, time varying disturbance if desired 
        # d = 0 
        # d = np.sign(x[1]) + np.sin(2*t)**3
        d = 1 * np.sin(0.2*t)
        # if t < 5:
            # d *= 0 
        # elif t > 10:
            # d *= 0 
        
        # return np.array([x[1],-x[0] + mu*(1-x[0]**2)*x[1] + u,du])                            # Standard model 
        return np.array([x[1],-x[0] + mu*(1-x[0]**2)*x[1] + d + u,du])                        # Additive disturbance  
        # return np.array([x[1],-x[0] + mu*(1-x[0]**2)*x[1] + d + (1+0.9*np.sin(x[0]))*u,du])   # State dependent term in front of u 
        # return np.array([x[1],-x[0] + mu*(1-x[0]**2)*x[1] + d + (1+0.6*np.sin(u))*u,du])      # Non-affine u 
        # return np.array([x[1],-x[0] + mu*(1-x[0]**2)*x[1] + d + (1+0.6*np.sin(u)+0.9*np.cos(x[0]))*u,du]) # Both 
    
    
    def __jacobian__(self, x, mu):    
        """ Jacobian of the system dynamics. """
        return np.array([[np.zeros_like(x[0]), np.ones_like(x[0])],[-1-2*x[0]*x[1]*mu, mu*(1-x[0]**2)]])
        
       
        
        
    def __adjoint__(self, costate, t, x, mu):
        """ Adjoint dynamics for reverse sensitivity analysis """
        A = self.__jacobian__(x(t), mu)
        A = np.transpose(A, axes=(2,1,0))
        # print "In __adjoint__"

        # print A.shape
        # print costate.shape
        l = self.__SAC_GRAD(x,t)
        # return np.dot(-A.T,costate)
        return l.T - np.dot(-A, costate)
      
    def integrate_adjoint(self,t,x,mu):
        """ Integrate the adjoint backward along a trajectory to determine sensitivity """
        
        nsteps,nstate,nsamples = x.shape
        
        I = np.eye(2) # Partial of each final state to the final state vector
        I.shape = (2,2,1)
        I = np.tile(I, (1, 1, nsamples))
        
        X = interp1d(t, x, kind='cubic', axis=0, bounds_error=False, assume_sorted=True,fill_value=(x[0],x[-1]))
        x0 = [0,0]
        costates = odeint(self.__adjoint__, x0, t[::-1], args=(X,mu))[::-1]
        return np.transpose(np.asarray(costates), axes=(1,0,2))
        
    def error(self, x, t):
        xd = self.ref(t) # returns the reference trajectory AND its derivative 
        
        if self.controller == 1:
            e1 = xd[0]-x[0]
            e2 = xd[1]-x[1] + 3*e1 # The paper does not place a coefficient in front of e1, but following the same idea as sliding mode, faster convergence can be achieved by using a coefficient > 1.
            return e2 
            
        elif self.controller==2: # PI + SMC 
            c = 3 # Controls rate of convergence on sliding manifold 
        
            e1 = xd[0]-x[0]
            e2 = xd[1]-x[1]
            s = e2 + c*e1 
            return s, c*e2
            
        else: #SAC    
            e1 = xd[0]-x[0]
            e2 = xd[1]-x[1]
            e = xd-x
            q = self.__SAC_Q()
             # quadratic error wrt to reference traj 
            return err 
            
    def __SAC_Q(self):
        return np.array([[1,0],[0,0.1]])
        
    def __SAC_GRAD(self,x,t):
        e = self.error(x,t)
        Q = self.__SAC_Q()
        return Q.dot(e)
        
    def __SAC_J(self):
        err = 0.5*e.T.dot(q).dot(e)
        # Cost function 
        pass 
        
    def simulate(self, sample, tf):
        ''' Integrate a single sample. '''
        t = np.linspace(0, tf, 2500)
        self.time = t 
        e0 = self.error(sample,0)
        x = odeint(self.__dynamics__, [sample[0],sample[1],np.zeros_like(sample[0])], t, args=(sample[2],e0)).T

        return x 

    
    def monte_carlo(self, samples, tf, controller=1):
        ''' Performs a Monte Carlo. Samples is an (N,3) array-like of deltas [x1,x2,mu] '''
        self.reset()
        print("Running MC with controller {}".format(controller))
        self.controller = controller 
        X = self.simulate(samples, tf) # Vectorized version
        
        u1 = np.array(self.controls)[0::4].T
        u2 = 2*np.array(self.controls)[1::4].T
        u3 = 2*np.array(self.controls)[2::4].T
        u4 = np.array(self.controls)[3::4].T
        self.controls = (u1+u2+u3+u4)/6
        
        self.outputs = np.asarray(X)   # State trajectories
        self.samples = samples
    
        
    def plot(self,fignum=0):
        ''' Visualizations '''
        t = self.time 
        xd_ = self.ref(t)

        colors = self.samples[2] # Use solely MU to define the colors 
        # nom = np.zeros_like(self.samples)
        # nom[1] += 1 
        # colors = np.linalg.norm(self.samples-nom,axis=0)
        
        colors = 1-(colors - colors.min())/(colors.max()-colors.min()) # Largest perturbations are black 
        # colors = (colors - colors.min())/(colors.max()-colors.min())   # Smallest perturbations are black 
        
        d = 3*(self.controller-1)
        
        plt.figure(d+3)
        for u,color in zip(self.controls,colors):
            plt.plot(t[:-1],u,color='{}'.format(color))
        plt.xlabel('Time (s)')
        plt.ylabel('Applied Controls')
        
        for color,x in zip(colors,self.outputs):
            plt.figure(d+1)
            plt.plot(x[0],x[1],color='{}'.format(color))

        
            plt.figure(d+2)
            plt.plot(t, xd_[0]-x[0],label='x1 error',color='{}'.format(color))
            
        plt.figure(d+2)    
        plt.xlabel('Time (s)')
        plt.ylabel('State Error (xd-x)')
            
        plt.figure(d+1)
        # plt.plot(xd_[0],xd_[1],'r--',label='Reference')     
        plt.title('Phase Portraits')
        plt.axis('equal')

            
        
    def test(self,controller=1):
        ''' 
                Monte Carlo 
            
        '''
        from Utils.boxgrid import boxgrid 
        x0 = self.ref(0)
        if 1:
            N1 = cp.Normal(x0[0],.02)
            N2 = cp.Normal(x0[1],.02)
            # MU = cp.Normal(0.35,.09)
            # MU = cp.Uniform(0,-1)
            MU = cp.Uniform(0,1)

        delta = cp.J(N1,N2,MU)
        
        tf = 20
        
        # samples = delta.sample(75,'S')
        samples = boxgrid([(x0[0]-.03,x0[0]+.03),(x0[1]-.03,x0[1]+.03),(0,1)], N=2, interior=False).T
        print(samples.shape)
        
        t0 = time.time()
        self.monte_carlo(samples,tf,controller=controller)
        t1 = time.time()
        
        print("MC time: {} s".format(t1-t0))
                        
        self.plot()
        

if __name__ == '__main__':    
    vdp = VDP()
    # vdp.simulate([0.1,0.9,0],5)
    vdp.test(1)
    # vdp.test(2)
    plt.show()


    