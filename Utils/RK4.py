import autograd.numpy as np

def RK4(fun, x0, iv, args=()):
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


def RK4_STM(fun, x0, iv, args):
    """ STM based sensitivity computation """
    from autograd import jacobian
    from scipy.interpolate import interp1d


    x = RK4(fun, x0, iv, args)
    n = len(x0)

    def temp(X):
        return fun(X[1:], X[0], *args)

    J = jacobian(temp, argnum=0)

    STM0 = np.eye(n)
    X = interp1d(iv, x, kind='cubic', assume_sorted=True, axis=0) # Create interp object from integration
    STM = RK4(stm_dyn, STM0, iv, args=(X,J))
    return x, np.array(STM)

def stm_dyn(stm, t, X, jac):
    x = X(t)
    dstm = jac(np.insert(x,0,t))[:,1:].dot(stm)
    return dstm
