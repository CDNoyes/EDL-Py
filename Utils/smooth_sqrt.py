import numpy as np
from pyaudi import gdual_double as gd
from pyaudi import gdual_vdouble as gdv
import pyaudi as pa
import DA as da 

""" From Virtuous Smooth for Global Optimization 

This technique actually extends to any function on [0,inf)
with the following required properties:
    f(0) = 0, 
    f increasing and concave on [0,inf), 
    f', f'' are defined on (0,inf),
    but f'(0) is undefined (hence the need for approximation)

The resulting approximation is differentiable everywhere, including at 0
and for root functions provides a lower bound  on f, useful in smooth relaxations
of f required for global optimization.

Examples include 
f(x) = x**p for 0 < p < 1
f(x) = log(1+x)
f(x) = arcsinh(sqrt(x))
"""

def smooth_sqrt(x, delta=1e-2):
    if da.const(x) >= delta:
        return pa.sqrt(x)

    f = np.sqrt(delta)
    fp = 0.5/f
    fpp = -0.25
    A = f/(delta**3) - fp/(delta**2) + 0.5*fpp/delta
    B = -3*f/delta**2 + 3*fp/delta - fpp
    C = 3*f/delta - 2*fp + 0.5*delta*fpp
    return A*x**3 + B*x**2 + C*x


def smooth_sqrt_vec(x, delta=1e-5):
    """ Vectorized Implementation"""
    y = np.zeros_like(x)

    i = x >= delta
    y[i] = np.sqrt(x[i])

    j = np.invert(i)
    f = np.sqrt(delta)
    fp = 0.5/f
    fpp = -0.25
    A = f/(delta**3) - fp/(delta**2) + 0.5*fpp/delta
    B = -3*f/delta**2 + 3*fp/delta - fpp
    C = 3*f/delta - 2*fp + 0.5*delta*fpp
    y[j] = A*x[j]**3 + B*x[j]**2 + C*x[j]
    return y 

def test():
    import matplotlib.pyplot as plt 

    x = np.linspace(0,0.1,1000)
    y = np.sqrt(x)
    plt.plot(x,y)

    for delta in [0.09, 0.01, 1e-4]:
        z = smooth_sqrt_vec(x, delta=delta) 

        plt.plot(x,z,'--', label=f"{delta}")
    plt.legend()
    plt.show()

def DA_example():
    x = np.linspace(0,0.1,11)
    dx = gd(0, 'x', 2)
    
    z = [smooth_sqrt(xi+dx, delta=0.01) for xi in x]
    print(z)

if __name__ == "__main__":
    # test()
    DA_example()