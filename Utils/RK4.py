import numpy as np
from scipy.interpolate import interp1d 

# TODO: now that a generalized step method exists, we can simply write two general solvers: fixed step and adaptive 


# For SDE, we assume the function returns the time derivative first, and the wiener process coefficients second
def EulerS(fun, x0, iv, args=()):
    """ Euler-Maruyama method """

    x = [np.asarray(x0)]
    n = len(x0)
    div = np.diff(iv)
    for t, dt in zip(iv, div):
        dw = np.random.normal(scale=np.abs(dt)**0.5, size=n)
        a, b = fun(x[-1], t, *args)
        x.append(x[-1] + a*dt + np.dot(b, dw))

    return np.asarray(x)


def _fixed_step_integrator(tableau):

    def Integrator(fun, x0, iv, args=()):
        x = [np.asarray(x0)]
        div = np.diff(iv)
        for t, dt in zip(iv, div):
            x.append(_step(fun, t, x[-1], dt, args, tableau))

        return np.asarray(x)
    return Integrator 


def RK45(fun, x0, iv, args=(), tol=1e-4, hmin=1e-6):

    T = _DOPRI45()
    # Step through adaptively then output at the points in iv 

    s = np.sign(iv[-1]-iv[0])  # allows for decreasing stepsizes 
    tf = iv[-1] 
    x = [np.asarray(x0)]
    t = [iv[0]]
    h = 1*s     # Initial stepsize 
    tc = iv[0]  # Current time 

    while t[-1]*s < tf*s:
        # print("Current time: {:.3g}".format(tc))
        if h*s > (tf-tc)*s:
            h = (tf-tc)

        accept = False 
        attempts = 0
        while not accept and attempts < 10:
            xc = _step(fun, tc, x[-1], h, args, T) # candidate step
            xc = np.moveaxis(xc, -1, 0)
            tnew = tc + h 
            scale = np.abs(x[-1]) + np.abs(x[-1]-xc[0]) + 1e-16 
            err = np.max(np.abs(xc[0] - xc[1])/scale) 
            rel_err = err/tol 
            h, accept = _step_control(np.abs(h), hmin, 0.2, 10, rel_err, order=5)
            h *= s 
            attempts += 1
        # print("maximum err in step ~ {:.1g} x tolerance".format(rel_err))
        tc = tnew 
        t.append(tnew)
        x.append(xc[0].squeeze())

    if len(iv) > 2: # interpolate solution onto desired timepoints 
        x = interp1d(t, x, axis=0, kind='linear')(iv)
        return x 
    else:  # Otherwise just give the dense output 
        return np.array(t), np.array(x)


def _step(f, iv, x, h, args, tableau):
    """ A general stepping method given a butcher tableau """
    k = [ f(x.copy(), iv, *args) ]
    move_axis = np.ndim(x)
    for ai, ci in zip(tableau.a[1:], tableau.c[1:]):
        dx = h*np.dot(np.moveaxis(k, 0, move_axis), ai)
        # dx = h*np.sum( aii*ki for aii,ki in zip(ai, k))  # Equivalent, should test to see which is faster 
        knew = f(x+dx, iv + ci*h, *args)
        k.append(knew)

    # This allows for stepsize control with matrix valued differential equations 
    if np.ndim(tableau.b) > 1: 
        x = np.expand_dims(x, -1)  # This allows for proper broadcasting in the return function 

    return x + h*np.dot( np.moveaxis(k, 0, move_axis), tableau.b)


def _step_control(h, hmin, shrink, grow, err, order):
    
    alpha = -1/order 
    accept_previous_step = err <= 1

    if accept_previous_step: # Determine the next step size
        if err < 1e-3:   # 1000x smaller than our tolerance allows -> let it grow
            scale = grow 
        else:
            scale = np.min((grow,  err**alpha))

        hnew = np.max((hmin, h*scale))
    else:                       # Determine a new stepsize to retry 
        if h <= hmin:  # bad situation, error is too large and minimum stepsize is already being used 
            hnew = h 
            accept_previous_step = True 
        else:
            hnew = np.max((hmin, h*np.max((shrink, 0.9*err**alpha))))

    return hnew, accept_previous_step


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


class _DOPRI45(Tableau):

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
                
                
class _RungeKutta4(Tableau):

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

    
class _Euler(Tableau):

    @property
    def c(self):
        return [0]

    @property
    def b(self):
        return np.array([1]) 

    @property
    def a(self):
        return [0]


RK4 = _fixed_step_integrator(_RungeKutta4())
Euler = _fixed_step_integrator(_Euler())


def test_adaptive():
    import matplotlib.pyplot as plt 

    def dyn(x, t):
        return t*x 

    x0 = np.random.random((3,3))  
    # x0 = np.array([1,2,3])

    t = np.linspace(0, 5)[::-1]
    x = RK4(dyn, x0, t)
    x.shape = (t.size, 9)
    ty, y = RK45(dyn, x0, [t[0], t[-1]], tol=1e-3)  # gets the adaptive steps
    # ty, y = RK45(dyn, x0, t, tol=1e-5) # gets the fixed time points 
    y.shape = (ty.size, 9)
    plt.plot(t, x)
    plt.plot(ty, y, 'o')
    plt.show()


def test_fixed_step():
    """ Verifies use of the fixed step integrator interface """

    import matplotlib.pyplot as plt 

    def dyn(x, t):
        return -t*x 

    # x0 = np.random.random((3,3))  
    x0 = np.array([1,2,3])

    t = np.linspace(0, 5, 100)

    xe = Euler(dyn, x0, t)
    xrk4 = RK4(dyn, x0, t)

    plt.plot(t, xe, label="Euler")
    plt.plot(t, xrk4, 'o', label="RK4")
    plt.legend()
    plt.show()


def test_stochastic():
    import matplotlib.pyplot as plt 

    def dyn(x, t):
        return -t*x, np.eye(3)*0.25 #np.diag(np.abs(x)**0.5)*0.09

    # x0 = np.random.random((3,3))  
    x0 = np.array([1,2,3])
    alpha = 0.1 
    t = np.linspace(0, 5, 100)
    for i in range(100):
        x = EulerS(dyn, x0 + np.random.random((3,))*0.1-0.05, t)
        if not i:
            plt.plot(t, x, 'k', alpha=alpha, label="EM Sample Paths")
        else:
            plt.plot(t, x, 'k', alpha=alpha)

    xd = Euler(lambda x,t: dyn(x,t)[0], x0, t)
    plt.plot(t, xd, 'r', label="Deterministic")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # test_fixed_step()
    test_stochastic()