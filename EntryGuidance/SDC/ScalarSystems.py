"""
    SDC factorizations of a scalar polynomial system 

"""

import numpy as np 
from SDCBase import SDCBase


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





