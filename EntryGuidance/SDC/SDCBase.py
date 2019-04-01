"""SDC factorizations of rotational spacecraft dynamics 

    In some papers, C(x) defines the output vector
    while in others it defines the terminal constraint. 


"""

import numpy as np 
import abc 


class SDCBase(abc.ABC):
    """
        Base class for SDC factorizations 

    """
    @property
    @abc.abstractmethod
    def n(self):
        """ State dimension """
        return 0

    @property
    @abc.abstractmethod
    def m(self):
        """ Control dimension """
        return 0

    def A(self, t, x):
        raise NotImplementedError

    def B(self, t, x):
        raise NotImplementedError

    def C(self, t, x):  
        return np.eye(self.n)

    def D(self, t):
        """ Affine terms """
        return np.zeros(self.n)

    def __call__(self, t, x):
        return self.A(t, x), self.B(t, x), self.C(t, x), self.D(t)

    def dynamics(self, u):
        def controlled_dynamics(x, t):
            A, B, _, D = self.__call__(t, x)
            return A.dot(x) + B.dot(u) + D
        return controlled_dynamics
         