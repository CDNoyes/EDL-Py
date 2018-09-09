"""  Adaptive Differential Algebra """

import numpy as np 
from abc import ABC

class ADABase(ABC):
    """ Defines an abstract base class for adaptive DA 
    
        In particular, any subclass must provide:
            A method to estimate error
            A mechanism to refine the domain for error control

        Any implementation should also track the correct polynomial for a given initial condition
        and provide a method to determine which polynomial to use for a new initial condition 

        For a dynamical system, the result is a list
        Each element corresponds to a time point
        Each element is itself a list of polynomials

    """

    # def __init__(self):
    #     pass 

    def refine(self):
        raise NotImplementedError

    def error(self):
        raise NotImplementedError

    
class AutomaticDomainSplitting(ADABase):

    def __init__(self):
        pass 

    def error(self):
        pass 

    def refine(self):
        pass 