"""
    SDC factorizations of a scalar polynomial system 

"""

import numpy as np 
from SDCBase import SDCBase
from replace import replace 

class PolySystem(SDCBase):
    """
        dx = ax + bx^2 + cx^3 + u

    """

    @property
    def n(self):
        return 1
    
    @property
    def m(self):
        return 1

    def __init__(self, a, b, c):
        self.abc = (a, b, c)

    def A(self, t, x):
        a,b,c = self.abc
        return np.asarray([a+b*x+c*x**2])

    def B(self, t, x):
        return np.array([[1]])

    def C(self, t, x):  
        return np.eye(1)

    def D(self, t):
        """ Affine terms """
        return np.zeros(1)


class ConstrainedPolySystem(SDCBase):
    """
        The same 1D system but with hard limit on the control 

        dx = ax + bx^2 + cx^3 + sat(u, umax)
        du = v 
        |u| <= u_max 

    """

    @property
    def n(self):
        return 2
    
    @property
    def m(self):
        return 1

    def __init__(self, a, b, c, umax):
        self.abc = (a, b, c)
        self.max = umax 

    def A(self, t, x):
        a,b,c = self.abc
        u = x[1]
        du = replace(np.clip(u, -self.max, self.max)/u, 1)
        return np.asarray([[a+b*x+c*x**2, du], [0, 0]])

    def B(self, t, x):
        return np.array([0, 1])

    def C(self, t, x):  
        return np.eye(2)

    def D(self, t):
        """ Affine terms """
        return np.zeros(2)


if __name__ == "__main__":
    model = ConstrainedPolySystem(1, 0, -0.1, 3)