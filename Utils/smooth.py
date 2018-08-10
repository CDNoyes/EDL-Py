import numpy as np
from scipy.interpolate import interp1d, BSpline, make_interp_spline
import matplotlib.pyplot as plt


def smooth(x, y, N=1, tau=0.3):
    """ Smooths a function y(x) N times using tau as the smoothing factor
    
    Larger values of tau produce more smoothing

    Returns an interpolation function
    """
    for _ in range(N):
        b = make_interp_spline(x, y, k=3)

        y = b(x)
        # tau = 0.5
        dx = np.diff(x)[0]
        alpha = dx/(tau+dx)
        z = [y[0]]
        for yi in y[1:]:
            z.append(z[-1] + alpha*(yi-z[-1]))
        y = z
    return interp1d(x, z, kind='cubic', axis=-1, copy=True, bounds_error=False, fill_value=(y[0],y[-1]), assume_sorted=True)


def test():
    import sys
    sys.path.append("./EntryGuidance")
    from ParametrizedPlanner import profile

    switches = [40,120,160]
    banks = [0.06,-1,1,-0.1]
    bankProfile = lambda t: profile(t, switch=switches, bank=banks, order=0)

    t1 = np.linspace(0,210,2001)
    bankprof = bankProfile(t1)


    smoothprof = smooth(t1,bankprof,5)
    t = np.linspace(0,200,8001)
    smoothprof = smoothprof(t)
    # smoothprof = smooth(t, smoothprof, 2)(t)
    plt.plot(t1,bankprof,'r--')
    plt.plot(t,smoothprof,label='Smoothed')
    plt.legend()
    plt.show()


def test_1():
    x = np.linspace(0,1,15000)
    y = np.hstack((x,1+0.2*x,1.2-0.8*x))
    t = np.linspace(0,1,y.size)
    
    ys = smooth(t,y,1,tau=0.009)(t)
    ys2 = smooth(t,y,3,tau=0.004)(t)
    
    plt.plot(t,y,'k--')
    plt.plot(t,ys,label="Smoothed Once")
    plt.plot(t,ys2,label="MultiSmoothed")
    plt.legend()
    
    plt.figure()
    n = 2
    dt = np.diff(t)[n-1:]
    t = t[n:]
    plt.plot(t,np.diff(y,n)/dt**n,'k--')
    plt.plot(t,np.diff(ys,n)/dt**n,label="Smoothed Once")
    plt.plot(t,np.diff(ys2,n)/dt**n,label="MultiSmoothed")
    plt.title('First Derivatives')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_1()
    # test()
