import numpy as np
from scipy.interpolate import interp1d, BSpline, make_interp_spline
import matplotlib.pyplot as plt
import sys
sys.path.append("./EntryGuidance")
from ParametrizedPlanner import profile


def smooth(x,y):

    b = make_interp_spline(x, y, k=3)

    y = b(x)
    tau = 0.2
    dx = np.diff(x)[0]
    alpha = dx/(tau+dx)
    z = [y[0]]
    for yi in y[:-1]:
        z.append(z[-1] + alpha*(yi-z[-1]))

    return interp1d(x, z, kind='cubic', axis=-1, copy=True, bounds_error=False, fill_value=(y[0],y[-1]), assume_sorted=True)




def test():
    switches = [40,120,160]
    banks = [0.06,-1,1,-0.1]
    bankProfile = lambda t: profile(t, switch=switches, bank=banks,order=2)

    t1 = np.linspace(0,210,2001)
    bankprof = bankProfile(t1)


    smoothprof = smooth(t1,bankprof)
    t = np.linspace(0,200,801)
    smoothprof = smoothprof(t)
    # smoothprof = smooth(t, smoothprof, 2)(t)
    plt.plot(t1,bankprof,'r--')
    plt.plot(t,smoothprof)
    plt.show()
if __name__ == "__main__":
    test()
