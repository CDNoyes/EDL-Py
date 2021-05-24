""" GNC Simulator 

Integrates dynamics operating on a fixed update cycle 

Interplay between
    Dynamics class(es) - how to concatenate state vectors 
    Controller class
    Data sharing/logging/passing, i.e. what does the controller/trigger need, and how does this info get passed?


dynamics have state names, so after each phase, a dataframe of all the states can be created 

Data_map should take the current time, state, and dynamic model are create a dataframe

"""

import numpy as np 


class SinglePhaseSingleSystem:

    def __init__(self, dynamic_model, controller, data_map):

        self.dynamic_model = dynamic_model
        self.controller = controller 
        self.map = data_map

    def run(self, initial_state, trigger):

        X = [np.asarray(x0)]
        U = []
        tc = 0 

        while True:
            xc = X[-1] # Current state 
            u = self.controller(tc, xc)   # What if the controller needs more information? 
            delta = min(dt, t[-1]-tc)
            xi = RK4(model.dynamics(u), xc, np.linspace(tc, tc+delta, 10))  # _ steps per control update 
            X.append(xi[-1])
            U.append(u)
            Nu.append(controller.cache)
            tc += delta


    


# class Phase:

#     def __init__(self):
#         pass