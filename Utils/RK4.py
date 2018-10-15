import numpy as np


def RK4(fun, x0, iv, args=()):
    """ Pure python implementation of a common
        4th-order Runge Kutta integrator
    """
    x = [np.asarray(x0)]
    div = np.diff(iv)
    for t,dt in zip(iv, div):
        x.append(_rk4_step(fun, t, x[-1], dt, args))

    return np.asarray(x)


def _rk4_step(f, iv, x, h, args):
    """ Takes a single 4th-order step """
    k1 = f(x[:],       iv,       *args)
    k2 = f(x+0.5*h*k1, iv+0.5*h, *args)
    k3 = f(x+0.5*h*k2, iv+0.5*h, *args)
    k4 = f(x+1.0*h*k3, iv+h,     *args)

    return x + h*(k1 + 2*k2 + 2*k3 + k4)/6.0


def RK45(fun, x0, iv, args=(), tol=1e-4):
    T = DOPRI45()
    # Step through adaptively then output at the points in iv 
    x = [np.asarray(x0)]
    h = 1 # Initial stepsize 
    t = iv[0] # Current time 


    xc = _step(fun, t, x[-1], h, args, T) # candidate step
    err = np.max(np.abs(xc[0] - xc[1]))
    rel_err = err/tol 



def _step(f, iv, x, h, args, tableau):
    """ A general stepping method given a butcher tableau """
    k = [ f(x[:], iv, *args) ]

    for ai, ci in zip(tableau.a[1:], tableau.c[1:]):
        dx = h*np.dot(np.moveaxis(k, 0, 2), ai)
        # dx2 = h*np.sum( aii*ki for aii,ki in zip(ai, k))  # Equivalent, should test to see which is faster 
        knew = f(x+dx, iv + ci*h, *args)
        k.append( knew )

    # This allows for stepsize control with matrix valued differential equations 
    if np.ndim(tableau.b) > 1 and np.ndim(x) > 1:
        x = np.expand_dims(x, -1) # This allows for proper broadcasting 

    return x + h*np.dot( np.moveaxis(k, 0, 2), tableau.b)



class Tableau:

    @property
    def a(self):
        raise NotImplementedError

    @property 
    def b(self):
        raise NotImplementedError

    @property
    def c(self):
        raise NotImplementedError 


class DOPRI45(Tableau):
  
    @property 
    def c(self):
        return [0.0, 0.2, 0.3, 0.8, 8/9, 1.0, 1.0]

    @property 
    def b(self):
        return np.array([[35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0], 
                [5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0, -92097.0/339200.0, 187.0/2100.0, 1.0/40.0]]).T

    @property 
    def a(self):
        return [ [0],
                [0.2],
                [3.0/40.0, 9.0/40.0],
                [44.0/45.0, -56.0/15.0, 32.0/9.0],
                [19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0],
                [9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0],
                [35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0]]
                
                

class RungeKutta4(Tableau):

    @property
    def c(self):
        return [0.0, 0.5, 0.5, 1.0]

    @property
    def b(self):
        return np.array([1/6, 1/3, 1/3, 1/6]) 

    @property
    def a(self):
        return [ [0],
                 np.array([0.5]),
                 np.array([0.0, 0.5]),
                 np.array([0.0, 0.0, 1.0]) ]

    

def test():

    def dyn(x,t):
        return t*x 

    # x0 = np.array([1,2,3])
    x0 = np.eye(3)
    print(_rk4_step(dyn, 1, x0, 0.5, ()))
    print(_step(dyn, 1, x0, 0.5, (), RungeKutta4()))
    x = _step(dyn, 1, x0, 0.5, (), DOPRI45())
    print(x[:,:,0])
    print(x[:,:,1])

if __name__ == "__main__":
    test()