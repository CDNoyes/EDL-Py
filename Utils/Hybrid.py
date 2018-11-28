
""" Utilities for working with hybrid systems

Algorithm:
    Given: Initial switches s
    Integrate the system forward to get state transition matrices 
    Compute the cost function's sensitivities = dJ/dx * dx/dsi
    Take a gradient descent step, clipping the steps so that switch times remain ordered. 
    if np.any(np.diff(s) < 0): use a smaller delta 
"""

import numpy as np
import numba as nb
from functools import partial 
from scipy.interpolate import interp1d 
from scipy.optimize import minimize 

from RK4 import RK4
import DA as da 


@nb.njit
def get_system_index_numba(val, arr):
    for idx in range(len(arr)):
        if arr[idx] > val:
            return idx
    return -1 # len(arr)

def get_system_index(t, s):
    # Best solution, O(1), s must be sorted 
    return np.searchsorted(s, t, side="right")


def switched_system(x, t, f, s, *args):
    """ Defines integrable dynamics of the form
        
        x_dot = f_i(x, t, *args)   for   s_i <= t < s_(i+1)

    """
    # i = get_system_index(t, s)
    i = get_system_index_numba(t, s)
    return f[i](x, t, *args)


def Optimize(Objective, F, x0, s0, t ):
    assert len(F) == len(s0)+1, "Number of dynamics must be 1 greater than number of switches"

    names = ["x{}".format(i) for i in range(len(x0))]

    # Create a single callable function that returns objective and gradient 
    def obj(s):

        # DA-based forward propagation 
        y0 = da.make(x0, names, 1, array=True)
        y = RK4(switched_system, y0, t, (F, s)).squeeze()
        
        # Compute the objective and its gradient wrt to final state 
        xf = da.const(y[-1])
        yf = da.make(xf, names, 1, True)
        J = Objective(yf)  # For now we assume the cost is of Mayer form 
        dJdx = da.gradient(J, names)

        # Compute sensitivity of final state to each switch time 
        dxds = []
        STMf = da.jacobian(y[-1], names)
        for i, si in enumerate(s):
            idx = np.argmin(np.abs(t-si)) # TODO: Interpolate solution and get exact state/stm 
            x = da.const(y[idx])
            STMi = da.jacobian(y[idx], names)
            STM = STMf.dot(np.linalg.inv(STMi))

            dxdsi = STM.dot(F[i](x, si) - F[i+1](x, si))
            dxds.append(dxdsi)

        dxds = np.array(dxds)
        return J.constant_cf, np.squeeze(dJdx.dot(dxds.T))

    # Write the well-orderedness constraint as a linear constraint As >= 0
    A = (np.eye(3, k=1) - np.eye(3))[:2]
    C = {"type": "ineq", "fun": lambda s: A.dot(s), "jac": lambda s: A}
    sol = minimize(obj, s0, method="SLSQP", jac=True, bounds=[(0, None)]*len(s0), constraints=C) # TODO: Need to add positive constraint, and well-ordered constraint 

    return sol

# def test_objective(x, t):
#     # Integrate forward, compute cost function
#     L = 0.5*np.array([xi.dot(xi) for xi in x])
#     J = np.trapz(L, t) 
#     return J 

def test_objective(xf):
    # return 0.5*xf.dot(xf) # LQ
    return xf[1]**2


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

    sol = Optimize(test_objective, F, x0, s, t,)
    print(sol)
    xopt = RK4(switched_system, x0, t, (F, sol.x))

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




# def Iteration(F, x0, s0, t ):
#     x, xda, STM = get_stm(switched_system, x0, t, (F,s0))

#     J = test_objective(xda, t)

#     # ISTM = [np.linalg.inv(stm) for stm in STM] # Inverses will be used more than once so it makes sense to store them 
#     # We actually only need to invert the STMs corresponding to t[j]

#     # dxds = np.zeros((len(x0, len(s0))))
#     dxds = []
#     for n, switch in enumerate(s0):
#         df = get_df(F, x, t, switch, n)
#         j = np.argmin(np.abs(t-switch))  # nth switch corresponds most closely to t[j]
#         stmj = STM[j]
#         istmj = np.linalg.inv(stmj)

#         for ti, stmi in zip(t, STM): # The sensitivity dx/dsj is the sum over the sensitivity of xi to sj
#             if ti < t[j] # times before the current switching time are clearly not affected by changing the switching time

#         STM_ti = STMf*np.linalg.inv(STMi)
#         dxdsi = STM1.dot(F[n](xf, t[-1]) - F[n+1](xf, t[-1]))
#         dxds.append(dxdsi)

#     dxds = np.array(dxds).T

#     return    



    

if __name__ == "__main__":
    test()
