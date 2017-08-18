import numpy as np
from pyaudi import gdual_double as gd
from pyaudi import exp

def smooth_sat(y, a=1, b=5.3013, c=0.5):
    """
        An (infinitely) differentiable approximation to the standard saturation function
        on the interval [0,1].
        Default parameters were found via optimization using scipy's curve_fit.
        Optimal values = [ 1., 5.3013, 0.5]
    """
    try:
        if isinstance(y[0],gd):
            return np.array([a/(1+exp(-b*(yi-c))) for yi in y])
        else:
            return a/(1+np.exp(-b*(y-c)))
    except:
        return a/(1+exp(-b*(y-c)))

def opt():
    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt

    y_raw = np.linspace(-2.5,3.5,500)
    y = np.clip(y_raw,0,1)

    aopt = curve_fit(sat, y_raw, y,bounds=(0,(1,10,1)))
    print "Optimal values = {}".format(aopt[0])

    y_approx = sat(y_raw,*aopt[0])
    plt.plot(y_raw,y)
    plt.plot(y_raw,y_approx,'--')

    plt.show()

    return aopt[0]

if __name__ == "__main__":
    opt()
