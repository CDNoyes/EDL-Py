import numpy as np

def RK4(fun, x0, iv, args):
    """ Pure python implementation of a common 4th-order Runge Kutta integrator """
    x = [np.asarray(x0)]
    div = np.diff(iv)
    for t,dt in zip(iv,div):
        x.append(rk4_step(fun, t, x[-1], dt, args))
    
    return np.asarray(x)

def rk4_step(f, iv, x, h, args):
    """ Takes a single 4th-order step """
    k1 = f(x[:],       iv,       *args)
    k2 = f(x+0.5*h*k1, iv+0.5*h, *args)
    k3 = f(x+0.5*h*k2, iv+0.5*h, *args)
    k4 = f(x+    h*k3, iv+h,     *args)  
    
    return x + h*(k1 + 2*k2 + 2*k3 + k4)/6.0