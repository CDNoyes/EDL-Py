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


def sqrt_sat(x, bound=1, tuning=1e-3):
    """ From Avvakumov et al,
        "Boundary value problem for ordinary differential equations
        with applications to optimal control"

        The smaller the tuning parameter is, the more tightly this will
         approximate the saturation function from [-bound, bound].
    """
    return 0.5*bound*(pa.sqrt(tuning + (x/bound + 1)**2) -
                      pa.sqrt(tuning + (x/bound - 1)**2))


def cosh_sat(x, bound=1, tuning=10):
    """Another saturation function, from 
    Saturated Robust Adaptive Control for Uncertain Nonlinear Systems using a new approximate model
    
    """

    return 0.5/tuning*pa.log(pa.cosh(tuning*(x+bound))/pa.cosh(tuning*(x-bound)))
    # return 0.5/tuning*np.log(np.cosh(tuning*(x+bound))/np.cosh(tuning*(x-bound)))


def test():
    from matplotlib import pyplot as plt 
    x = np.linspace(-1.2,1.2,500)
    y = np.clip(x,-1,1)

    y1 = [sqrt_sat(xi, 1, 1e-3) for xi in x]
    y2 = [cosh_sat(xi, 1, 30) for xi in x]

    plt.plot(x, y, label="Saturation")
    plt.plot(x, y1, label="Sqrt")
    plt.plot(x, y2, label="Cosh")
    plt.legend()
    plt.show()


def compare_error():
    """ Determines the settings needed to produce similar levels of error 
    
    Results: 
        cosh with 200 : 1.7e-3 max error
        sqrt with 1e-5: 1.6e-3 max error 
    """
    
    from matplotlib import pyplot as plt 
    
    x = np.linspace(-1.5, 1.5, 20000)
    y = np.clip(x, -1, 1)

    ycosh = np.array([cosh_sat(xi, bound=1, tuning=200) for xi in x])
    ysqrt = np.array([sqrt_sat(xi, bound=1, tuning=1e-5) for xi in x])

    ecosh = np.abs(y-ycosh)
    esqrt = np.abs(y-ysqrt)

    print("Cosh: Mean Error = {}".format(ecosh.mean()))
    print("Cosh: Max Error  = {}".format(ecosh.max()))
    print("Sqrt: Mean Error = {}".format(esqrt.mean()))
    print("Sqrt: Max Error  = {}".format(esqrt.max()))

    plt.semilogy(x, ecosh, label='Cosh')
    plt.semilogy(x, esqrt, label='Sqrt')
    plt.vlines([-1,1], 1e-6, 1)
    plt.legend()
    plt.show()


def compare():
    from matplotlib import pyplot as plt 
    import DA as da 

    # See which version of saturation is more accurately represented by DA variables
    x_exp = np.linspace(-1.5, 1.5, 20)
    names = ['x']*x_exp.size

    for order in [2,3,]:
        z = da.make(x_exp, names, order)


        #  Each is a list of DA's.
        y1 = [sqrt_sat(xi, 1, 1e-5) for xi in z]
        y2 = [cosh_sat(xi, 1, 200) for xi in z]

        #  Now we sample each one in a neighborhood and compute the error 
        x_eval = np.linspace(-0.1, 0.1, 101)

        s1 = da.evaluate(y1, ['x'],  x_eval)
        s2 = da.evaluate(y2, ['x'], x_eval)

        true = np.array([np.clip(xi+x_eval, -1, 1) for xi in x_exp]).T
        e1 = np.abs(s1-true)
        e2 = np.abs(s2-true)

        E1 = np.mean(e1, axis=0)
        E2 = np.mean(e2, axis=0)
        plt.figure(1)
        plt.semilogy(x_exp, E1, label="sqrt sat, O={}".format(order))
        plt.figure(2)
        plt.semilogy(x_exp, E2, label="cosh sat, O={}".format(order))

    for i in range(1,3):
        plt.figure(i)    
        plt.xlabel("Expansion Points")
        plt.ylabel("Average Error in a Neighborhood of Width = {}".format(x_eval[-1]-x_eval[0]))
        plt.legend()
    plt.show()


if __name__ == "__main__":
    # opt()
    test()
    # compare_error()
    # compare()