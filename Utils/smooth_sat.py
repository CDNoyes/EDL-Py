import numpy as np
from pyaudi import gdual_double as gd
from pyaudi import gdual_vdouble as gdv
import pyaudi as pa


def smooth_sat(y, b=5.522):
    """
        An (infinitely) differentiable approximation to the standard saturation function
        on the interval [0,1] using the logistic function.
        Default parameters were found via optimization using scipy's curve_fit.
        For the interval [0,1] the optimal values = [ 1., 5.3013]
        For the interval [0,0.96] the optimal values = [ 0.96., 5.522]
    """
    a = 0.96  # Maximum value of the saturation function
    c = a/2.
    try:
        if isinstance(y[0], (gd, gdv)):
            return np.array([a/(1+pa.exp(-b*(yi-c))) for yi in y])
        else:
            return a/(1+np.exp(-b*(y-c)))
    except (IndexError, TypeError):
        return a/(1+pa.exp(-b*(y-c)))


def symmetric_sat(x, bound=1, tuning_parameter=1e-3):
    """ From Avvakumov et al,
        "Boundary value problem for ordinary differential equations
        with applications to optimal control"

        The smaller the tuning parameter is, the more tightly this will
         approximate the saturation function from [-bound, bound].
    """
    return 0.5*bound*(pa.sqrt(tuning_parameter + (x/bound + 1)**2) -
                      pa.sqrt(tuning_parameter + (x/bound - 1)**2))

def another_sat(x, bound=1, tuning=10):
    """Another saturation function, from 
    Saturated Robust Adaptive Control for Uncertain Nonlinear Systems using a new approximate model
    
    """

    return 0.5/tuning*pa.log(pa.cosh(tuning*(x+bound))/pa.cosh(tuning*(x-bound)))
    # return 0.5/tuning*np.log(np.cosh(tuning*(x+bound))/np.cosh(tuning*(x-bound)))


def test():
    from matplotlib import pyplot as plt 
    x = np.linspace(-1.2,1.2,500)
    y = np.clip(x,-1,1)

    y1 = [symmetric_sat(xi, 1, 1e-3) for xi in x]
    y2 = [another_sat(xi, 1, 20) for xi in x]
    # y2 = another_sat(x, 1, 20)

    plt.plot(x, y, label="Saturation")
    plt.plot(x, y1, label="Symmetric")
    plt.plot(x, y2, label="Another")
    plt.legend()
    plt.show()

def compare():
    # See which version of saturation is more accurately represented by DA variables

    

if __name__ == "__main__":
    # opt()
    test()