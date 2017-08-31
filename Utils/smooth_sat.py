import numpy as np
from pyaudi import gdual_double as gd
from pyaudi import exp

def smooth_sat(y, a=1, b=5.3013, c=0.5):
    """
        An (infinitely) differentiable approximation to the standard saturation function
        on the interval [0,1] using the logistic function.
        Default parameters were found via optimization using scipy's curve_fit.
        For the interval [0,1] the optimal values = [ 1., 5.3013, 0.5]
    """
    try:
        if isinstance(y[0],gd):
            return np.array([a/(1+exp(-b*(yi-c))) for yi in y])
        else:
            return a/(1+np.exp(-b*(y-c)))
    except:
        return a/(1+exp(-b*(y-c)))


def tanh_sat(y,k,y0):
    """
    The logistic function and hyberbolic tangent are related and can produce identical results.
    Optimal = [2.65,0.5]
    """
    return 0.5 + 0.5*np.tanh(k*(y-y0))


def atan_sat(y,k,y0,c,a):
    return c + a*(np.arctan(k*(y-y0)) - np.arctan(-k*(y-y0)))



def opt():
    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt

    y_raw = np.linspace(-2.5,3.5,500)
    y = np.clip(y_raw,0,0.96)

    aopt = curve_fit(smooth_sat, y_raw, y,bounds=(0,(1,10,1)))
    print "(Logistic) Optimal values = {}".format(aopt[0])

    # topt = curve_fit(tanh_sat, y_raw, y)
    # print "(Tanh) Optimal values = {}".format(topt[0])

    copt = curve_fit(atan_sat, y_raw, y)
    print "(Tan inv) Optimal values = {}".format(copt[0])


    y_log = smooth_sat(y_raw,*aopt[0])
    # y_tanh = tanh_sat(y_raw,*topt[0])
    y_atan = atan_sat(y_raw, *copt[0])
    plt.plot(y_raw,(y))
    plt.plot(y_raw,(y_log),'--')
    plt.plot(y_raw,y_atan)
    # plt.plot(y_raw,(y_tanh),'*')


    # err = (np.arccos(y) - np.arccos(y_approx))*180/np.pi
    # plt.figure()
    # plt.plot(y_raw,err)
    # plt.ylabel("Error between sat - smooth_sat (deg)")
    plt.show()

    return aopt[0]

if __name__ == "__main__":
    opt()
