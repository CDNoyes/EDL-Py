from pyaudi import gdual_double as gd
from pyaudi import gdual_vdouble as gdv
from pyaudi import sin, cos, log

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
        
    def __dynamics__(self, x, t, mu, u):
        return np.array([x[1], -x[0] + mu*(1-x[0]**2)*x[1] + u])
        
    def simulate(self, sample, ufun, tf, N=100, mu=0.25):
        ''' Integrate a single sample. '''
        t = np.linspace(0,tf,N)
        # t0 = time.time()
        # T,x = ode45(self.__dynamics__, [sample[0],sample[1]], t, args=(sample[2],))
        x = RK4(self.__dynamics__, sample, t, args=(mu,ufun))
        # t1 = time.time()
        # plt.plot(x[:,0],x[:,1])
        # y = odeint(self.__dynamics__, [sample[0],sample[1]], t, args=(sample[2],))
        # t2 = time.time()
        
        # print "ode45 {}".format(t1-t0)
        # print "odeint {}".format(t2-t1)
        # plt.plot(x[:,0],x[:,1],'--',label='odeint')
        # plt.show()
        return x
        
    def compare_vectorized(self):
        ''' Found vectorized gdual. How does it scale? Can I evaluate all monte carlo points at once? Controller would need to be vectorized as well. 
        
            Results: Scales linearly with number of points. Does not do all vs all, each vectorized element must have the same number of elements unless scalar.
                     Propagating multiple expansion points is significantly faster than integrating individually.
        '''
        
        vars = ['x10','x20'] # A list of all DA variables, in order for gradient/jacobian computations
        tf = 10
        
        # Test the utilities for converting back and forth:
        x0 = [gd(3,x,4) for x in vars]
        T = []
        x0.append(gd(.25)) # constant mu
        x0 = np.array(x0)
        t0 = time.time()
        x = self.simulate(x0,tf)
        T.append(time.time()-t0)
        
        N = [1,2,3,4,5,10,25]
        for n in N[1:]:
            x0 = [gdv(np.linspace(-2.5,2.5,n),x,4) for x in vars]
            x0.append(gdv([.25])) # constant mu
            x0 = np.array(x0)
            t0 = time.time()
            x = self.simulate(x0,tf)
            T.append(time.time()-t0)
        
        Tdouble = []
        for n in N:
            x0 = [np.linspace(-2.5,2.5,n), np.linspace(-2.5,2.5,n), np.ones(n)*0.25]
            x0 = np.array(x0)
            t0 = time.time()
            x = self.simulate(x0,tf)
            Tdouble.append(time.time()-t0)

        
        # Tscalar = [T[0]]
        
        # for n in N[1:]:
            # Tsum = 0
            # for xp in np.linspace(-2.5,2.5,n):
                # x0 = [gd(xp,x,4) for x in vars]
                # x0.append(gd(.25)) # constant mu
                # x0 = np.array(x0)
                # t0 = time.time()
                # x = self.simulate(x0,tf)
                # Tsum += time.time()-t0
            # Tscalar.append(Tsum)            
            
            
        # print len(x[-1,0].constant_cf)
        # print x[-1,0].constant_cf
        # print x[-1,1].constant_cf
        # print (x[-1,0]+x[-1,1]).constant_cf
        
        plt.figure()
        # plt.plot(N,Tscalar,label='scalar')
        plt.plot(N,T,label='vectorized')
        plt.plot(N,Tdouble,label='vectorized doubles')
        # plt.plot(N,T[0]*np.array(N),'--',label='scalar theoretical')
        plt.xlabel('N, Number of expansion points for two states')
        plt.ylabel('Runtime (s)')
        plt.legend(loc='best')
        plt.show()
        
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
        
        
        
def test_entry():
    """ Integrate an open loop entry with gd for sensitivity analysis. """
    from EntryGuidance.Simulation import Simulation, Cycle, EntrySim, TimedSim
    from EntryGuidance.ParametrizedPlanner import HEPNR
    from EntryGuidance.InitialState import InitialState
    
    # reference_sim = Simulation(cycle=Cycle(1),output=True,**EntrySim())
    reference_sim = Simulation(cycle=Cycle(1), output=True, use_da=True, **TimedSim(50))
    bankProfile = lambda **d: HEPNR(d['time'],*[9.3607, 136.276], minBank=np.radians(30))
                                                
    x0 = InitialState()
    vars = ['r','theta','phi','v','gamma','psi','s','m']
    x0 = da.make(x0, vars, [2,0,0,2,2,0,0,0])
    print x0
    output = reference_sim.run(x0,[bankProfile])
    
    # Now evaluate the trajectory at some delta values, and rerun actual trajectories from the same perturbed input conditions
    
    
        
def test_hessian():
    vars = ['x10','x20'] # A list of all DA variables, in order for gradient/jacobian computations
        
    x0 = [gd(3,x,2) for x in vars]
    f = x0[0]**2 + x0[1]**3
    print da.gradient(f, vars)
    print da.hessian(f,vars)
    
    return
    
    
    
def test_drag_profile():
    das = [gd(1,x,4) for x in ['D1','D2']]
    V = np.linspace(2,10, 10)
    D = das[0]*V + das[1]*V**2
    h = [-2*log(2*Dd/(Vd**2)) for Dd,Vd in zip(D,V)]
    E = [1/hi + 0.5*v**2 for hi,v in zip(h,V)]
    fpa = [(hi-hj) for hi,hj,ei,ej,di in zip(h[:-1],h[1:],E[:-1],E[1:],D)]
    
    print fpa[-1]
    
def test_smooth_bank():
    from EntryGuidance.ParametrizedPlanner import HEPNR,HEPBankReduced
    from functools import partial
    
    das = [gd(da.radians(15),'minBank',2), gd(da.radians(85),'maxBank',2)]
    
    f = partial(HEPBankReduced, t1 = 10, t2 = 60, minBank=das[0], maxBank=das[1])
    t = np.linspace(0,150,300)
    F = (f(t))
    print [cos(Fi) for Fi in F]
    # print da.gradient(F,['t'])
    # print da.hessian(F,['t'])
    plt.plot(t,da.const(F))
    plt.show()

        
if __name__ == "__main__":
    vdp = VDP()
    # vdp.compare_vectorized()
    vdp.test()
    # test_hessian()
    # test_smooth_bank()
    # test_entry()
    # test_drag_profile()