from pyaudi import gdual_double as gd
import DA as da
import numpy as np
from scipy.integrate import odeint
from ode45 import ode45
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
        
        # Test the utilities for converting back and forth:
        x0 = [gd(3,x,1) for x in vars]
        x0 = da.const(x0, array=False)
        x0 = da.make(x0,vars,1, array=False)
        
        x0.append(gd(1)) # constant mu

        x = self.simulate(x0,6)
        # Test the utilities for getting coefficieints
        stm = da.jacobian(x[-1], vars)
        print stm
    
def RK4(fun, x0, iv, args):
    x = [np.array(x0)]
    div = np.diff(iv)
    for t,dt in zip(iv,div):
        x.append(rk4_step(fun, t, x[-1], dt, args))
    return np.array(x)

def rk4_step(f, iv, x, h, args):
    
    k1 = f(x,          iv,       *args)
    k2 = f(x+0.5*h*k1, iv+0.5*h, *args)
    k3 = f(x+0.5*h*k2, iv+0.5*h, *args)
    k4 = f(x+    h*k3, iv+h,     *args)  
    
    return x + h*(k1 + 2*k2 + 2*k3 + k4)/6.0
        
if __name__ == "__main__":
    vdp = VDP()
    vdp.test()