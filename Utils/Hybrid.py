""" Utilities for working with hybrid systems """

import numpy as np 
from functools import partial 
from scipy.interpolate import interp1d 

from RK4 import RK4
import DA as da 

def get_system_index(t, s):
    # Best solution, O(1), s must be sorted 
    return np.searchsorted(s, t, side="right")


def switched_system(x, t, f, s, *args):
    """ Defines integrable dynamics of the form
        
        x_dot = f_i(x, t, *args)   for   s_i <= t < s_(i+1)

    """
    i = get_system_index(t, s)
    return f[i](x, t, *args)


def test_objective(x, t):
    # Integrate forward, compute cost function
    L = 0.5*np.array([xi.dot(xi) for xi in x])
    J = np.trapz(L, t) 
    return J 


def test():
    import matplotlib.pyplot as plt 

    A1 = np.array([[-1, 0],[1, 2]])
    A2 = np.array([[1, 1],[1, -2]])
    
    s = [0.3, 0.5, 0.7] # initial guess at switching times 
    s_opt = [0.518, 0.688, 0.791]

    f1 = lambda x, t: A1.dot(x)
    f2 = lambda x, t: A2.dot(x)
    F = [f1, f2, f1, f2]
    dyn = lambda x, t: switched_system(x, t, F, s)

    x0 = np.array([1., 0.])
    tf = 1 
    t = np.linspace(0, tf, 500)
    x = RK4(dyn, x0, t)
    xopt = RK4(switched_system, x0, t, (F, s_opt))

    Optimize(F, x0, s, t,)




    plt.plot(t, x, label="Guess")
    plt.plot(t, xopt, '--', label="Optimal")
    plt.legend()
    plt.show()


def get_df(F, x, t, s, i):
    """ 
        Constructs the vector fi - fi+1
    """
    y = interp1d(t, x, axis=0)(s)
    df = F[i](y, s)-F[i+1](y, s) 
    return df


def get_stm(dyn, x0, t, args):
    names = ["x{}".format(i) for i in range(len(x0))]

    y0 = da.make(x0, names, 1)
    y = RK4(dyn, y0, t, args)

    return [da.const(yi) for yi in y], y, [da.jacobian(yi, names) for yi in y]


def Optimize(F, x0, s0, t ):
    assert len(F) == len(s0)+1, "Number of dynamics must be 1 greater than number of switches"
    return 


def Iteration(F, x0, s0, t ):
    x, xda, STM = get_stm(switched_system, x0, t, (F,s0))

    J = test_objective(xda, t)

    # ISTM = [np.linalg.inv(stm) for stm in STM] # Inverses will be used more than once so it makes sense to store them 
    # We actually only need to invert the STMs corresponding to t[j]

    # dxds = np.zeros((len(x0, len(s0))))
    dxds = []
    for n, switch in enumerate(s0):
        df = get_df(F, x, t, switch, n)
        j = np.argmin(np.abs(t-switch)) # nth switch corresponds most closely to t[j]
        stmj = STM[j]
        istmj = np.linalg.inv(stmj)

        for ti, stmi in zip(t, STM): # The sensitivity dx/dsj is the sum over the sensitivity of xi to sj
            if ti < t[j] # times before the current switching time are clearly not affected by changing the switching time

        STM_ti = STMf*np.linalg.inv(STMi)
        dxdsi = STM1.dot(F[n](xf, t[-1]) - F[n+1](xf, t[-1]))
        dxds.append(dxdsi)

    dxds = np.array(dxds).T


    return    

"""
Algorithm:
    Given: Initial switches s
    Integrate the system forward to get state transition matrices 
    Compute the cost function's sensitivities = dJ/dx * dx/dsi
    Take a gradient descent step, clipping the steps so that switch times remain ordered. 
    if np.any(np.diff(s) < 0): use a smaller delta 
"""

if __name__ == "__main__":
    test()
