from pyaudi import gdual_double as gd
from pyaudi import gdual_vdouble as gdv
from pyaudi import sin

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import Utils.DA as da
from Utils.RK4 import RK4

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time

class VDP(object):
    ''' A simplified van der pol oscillator class '''
    
    def __init__(self):
        print "Initialized"
        
    def __dynamics__(self, x, t, mu):
        return np.asarray([x[1],-x[0] + mu*(1-x[0]**2)*x[1]])
        
    def simulate(self, sample, tf):
        ''' Integrate a single sample. '''
        t = np.linspace(0,tf,100)
        t0 = time.time()
        # T,x = ode45(self.__dynamics__, [sample[0],sample[1]], t, args=(sample[2],))
        x = RK4(self.__dynamics__, [sample[0],sample[1]], t, args=(sample[2],))
        t1 = time.time()
        # plt.plot(x[:,0],x[:,1])
        # y = odeint(self.__dynamics__, [sample[0],sample[1]], t, args=(sample[2],))
        # t2 = time.time()
        
        # print "ode45 {}".format(t1-t0)
        # print "odeint {}".format(t2-t1)
        # plt.plot(x[:,0],x[:,1],'--',label='odeint')
        # plt.show()
        return x

    def test(self):
        vars = ['x10','x20'] # A list of all DA variables, in order for gradient/jacobian computations
        tf = 10
        
        # Test the utilities for converting back and forth:
        x0 = [gd(3,x,6) for x in vars]
        # x0 = da.const(x0, array=False)
        # x0 = da.make(x0,vars,2, array=False)
        
        x0.append(gd(.25)) # constant mu
        x0 = np.array(x0)
        t0 = time.time()
        x = self.simulate(x0,tf)
        t_prop = time.time()-t0
        
        # Test the utilities for getting coefficieints
        stm = np.array([da.jacobian(xi, vars) for xi in x])
        
        # print stm[-1]
        
        # Test the evaluation method
        import chaospy as cp
        U = cp.Uniform(-0.2,0.2)
        N = cp.Normal(0,.07)
        J = cp.J(U,N)
        
        nSamples = 500
        pts = J.sample(nSamples)
        t0 = time.time()
        new_xf = da.evaluate(x[-1], vars, pts.T)
        t_eval = time.time()-t0
        
        pts = np.vstack( (pts, np.zeros((1,nSamples))) )
        x0 = da.const(x0,array=True)
        t0 = time.time()
        new_xf_true = np.array([self.simulate(x0+dx0,tf)[-1] for dx0 in pts.T])
        t_int = time.time()-t0
        
        err = np.linalg.norm(new_xf-new_xf_true,axis=1)
        print "Single propagation time:    {}".format(t_prop)
        print "Polynomial evaluation time: {}".format(t_eval)
        print "                            -------------------"
        print "Sum                       : {}".format(t_eval+t_prop)
        print "Numerical integration time: {}".format(t_int)
        
        from Utils.boxgrid import boxgrid
        extrema = boxgrid([(-.2,.2),(-.2,.2),(0,0)],2,False)
        extrema_traj = np.array([self.simulate(x0+dx0,tf) for dx0 in extrema])

        
        
        import matplotlib.pyplot as plt
        # Plot trajectories
        plt.figure()
        plt.plot(da.const(x[:,0]),da.const(x[:,1]))
        for traj in extrema_traj:
            plt.plot(traj[:,0],traj[:,1],'--')
            
        # Sensitivity trajectories
        t = np.linspace(0,tf,stm.shape[0])
        plt.figure()
        plt.plot(t, stm[:,0,0])
        plt.plot(t, stm[:,0,1])
        plt.plot(t, stm[:,1,0])
        plt.plot(t, stm[:,1,1])
        
        # Errors
        plt.figure()
        plt.scatter(pts[0,:],pts[1,:],c=err)
        plt.colorbar()
        
        # Predicted pts vs true pts
        plt.figure()
        plt.scatter(new_xf_true[:,0],new_xf_true[:,1],c='r')
        plt.scatter(new_xf[:,0],new_xf[:,1],marker='x')
        
        plt.show()
        
        
        
        
        
def test_hessian():
    vars = ['x10','x20'] # A list of all DA variables, in order for gradient/jacobian computations
        
    x0 = [gd(3,x,2) for x in vars]
    f = x0[0]**2 + x0[1]**3
    print da.gradient(f, vars)
    print da.hessian(f,vars)
    
    return
    
def test_opt_with_jac():
    return
    
    
def test_smooth_bank():
    from EntryGuidance.ParametrizedPlanner import HEPNR
    from functools import partial
    
    f = partial(HEPNR, t1 = 10, t2 = 30)
    t = gd(12,'t',2)
    
    F = f(t)
    print F
    print da.gradient(F,['t'])
    # print da.hessian(F,['t'])
    

        
if __name__ == "__main__":
    vdp = VDP()
    vdp.test()
    # test_hessian()
    # test_smooth_bank()