import numpy as np
import chaospy as cp
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
            
    '''
    def __init__(self):
        print "VDP created."
        self.samples=None 
        self.outputs=None 
        self.time = None 
        self.controls = []
        
    def ref(self, t):
        k1 = 1
        a = 0. 
        k2 = -3 
        return np.array([np.sin(k1*t)+a*np.sin(k2*t),k1*np.cos(k1*t)+a*k2*np.cos(k2*t)])
        # return np.array([np.zeros_like(t),np.zeros_like(t)])
        
    def __dynamics__(self, x, t, mu, e0):
        """ System dynamics, including robust controller """
        
        # Gains
        k       = 200
        alpha   = 0.6   # > 0.5 
        beta    = 20 
        
        # Current error 
        e2 = self.error(x,t)
        
        # Control computation 
        u = (1+k)*(e2-e0) + x[2]
        du = (1+k)*alpha*e2 + beta*np.sign(e2)
        # u = np.clip(u,-1,1)                           # Apply control limitations if desired 
        # u *= 0                                          # Uncontrolled system 
        self.controls.append(u)
        
        # Add an external, time varying disturbance if desired 
        d = 1 * np.sin(0.2*t)
        if t < 5:
            d *= 0 
        elif t > 10:
            d *= 0 
        
        return np.array([x[1],-x[0] + mu*(1-x[0]**2)*x[1] + d + u,du])
        
    def error(self, x, t):
        xd = self.ref(t) # returns the reference trajectory AND its derivative 
        e1 = xd[0]-x[0]
        e2 = xd[1]-x[1] + e1 
        return e2 
        
    def simulate(self, sample, tf):
        ''' Integrate a single sample. '''
        t = np.linspace(0,tf,8001)
        self.time = t 
        e0 = self.error(sample,0)
        x = odeint(self.__dynamics__, [sample[0],sample[1],np.zeros_like(sample[0])], t, args=(sample[2],e0)).T

        return x 

    
    def monte_carlo(self, samples, tf):
        ''' Performs a Monte Carlo. Samples is an (N,3) array-like of deltas [x1,x2,mu] '''
         
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
        
        plt.figure(3)
        for u,color in zip(self.controls,colors):
            plt.plot(t[:-1],u,color='{}'.format(color))
        plt.xlabel('Time (s)')
        plt.ylabel('Applied Controls')
        
        for color,x in zip(colors,self.outputs):
            plt.figure(1)
            plt.plot(x[0],x[1],color='{}'.format(color))
            # plt.plot(sample[0],sample[1],'x')
            # plt.legend()
        
            plt.figure(2)
            plt.plot(t, xd_[0]-x[0],label='x1 error',color='{}'.format(color))
            # plt.plot(t, xd_[1]-x[1],label='x2 error')
            
        plt.figure(2)    
        plt.xlabel('Time (s)')
        plt.ylabel('State Error (xd-x)')
            
        plt.figure(1)
        # plt.plot(xd_[0],xd_[1],'r--',label='Reference')     
        plt.title('Phase Portraits')
        plt.axis('equal')
        plt.show()

            
        
    def test(self):
        ''' 
                Monte Carlo 
            
        '''
        x0 = self.ref(0)
        if 1:
            N1 = cp.Normal(x0[0],.02)
            N2 = cp.Normal(x0[1],.02)
            # MU = cp.Normal(0.35,.09)
            # MU = cp.Uniform(0,-1)
            MU = cp.Uniform(0,2)

        delta = cp.J(N1,N2,MU)
        
        tf = 15
        
        samples = delta.sample(500,'S')
        
        t0 = time.time()
        self.monte_carlo(samples,tf)
        t1 = time.time()
        
        print "MC time: {} s".format(t1-t0)
                        
        self.plot()
        

if __name__ == '__main__':    
    vdp = VDP()
    # vdp.simulate([0.1,0.9,0],5)
    vdp.test()

    