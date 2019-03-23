""" GNC Simulator 

Integrates dynamics operating on a fixed update cycle 

Interplay between
    Dynamics class(es) - how to concatenate state vectors 
    Controller class
    Data sharing/logging/passing, i.e. what does the controller need, and how does this info get passed?


dynamics have state names, so after each phase, a dataframe of all the states can be created 

"""

import numpy as np 

class Phase:

    def __init__(self):
        pass