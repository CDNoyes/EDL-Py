"""SDC factorizations of rotational spacecraft dynamics 

"""

import numpy as np 
from SDCBase import SDCBase


class Attitude:
    """
        x = [wx, wy, wz]
        u = [ax, ay, az]

        Thrust acceleration is the control

    """
    @property
    def n(self):
        """ State dimension """
        return 3

    @property
    def m(self):
        """ Control dimension """
        return 3

    def __init__(self, inertia):

        self.J = inertia 
        self.Ji = np.linalg.inv(inertia)

    def A(self, t, w):
        return -self.Ji.dot(np.cross(w, self.J))

    def B(self, t, x):
        return self.Ji

    def C(self, t, x):  
        return np.eye(3)



if __name__ == "__main__":
    import time 
    import pandas as pd 
    import sys 
    import matplotlib.pyplot as plt 

    sys.path.append("./Utils")
    from SDRE import SDRE, SDREC 
    
    # w0 = [-0.4, 0.8, 2]  # high ic
    w0 = [-0.02, 0.1, 0.05]
    x0 = np.array(w0)

    t0 = 0
    tf = 25
    N = 100
    t = np.linspace(t0, tf, N)

    model = Attitude(np.diag([86.24, 85.07, 113.59]))

    # Q = lambda t,x: model.A(t,x)
    Q = lambda t,x: np.zeros((3,3))
    R = lambda t,x,u: np.eye(3)
    F = np.zeros((3, 3))
    z = np.zeros((3, 1)) 

    print("Running point constrained SC detumble ")
    x, u, k = SDREC(x0, tf, model.A, lambda t,x,u: model.B(t,x), model.C, Q, R, F, z, model.m, n_points=N)
    XU = np.concatenate((x, u), axis=1)
    plt.figure()
    plt.plot(t, x)
    plt.figure()
    plt.plot(t, u)
    plt.show()



