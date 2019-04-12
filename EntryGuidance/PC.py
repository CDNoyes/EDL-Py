""" Predictor Corrector Entry Guidance

Implements the PC method for range control 

 """

import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import odeint 

import sys 
sys.path.append("./Utils")
from RK4 import RK4 
from SDC.Entry import Energy 

class PredictorCorrector:

    def __init__(self, target_range, final_energy, model):
        self.sf = target_range
        self.ef = final_energy 
        self.filters = [None, None] # First order fading memory filter values 
        self.previous = None  # Record of past control value for secant update 
        self.model = model # model that provides the longitudinal dynamics d
        

    def __callable__(self, state, energy):

        # 
        if self.previous is None:
            u0 = np.radians(45)
        else:
            u0 = self.previous[0] 

        du = 0.1 # bank angle delta for finite diff 
        ulim = np.pi/2 
        if np.abs(u0 - np.pi/2) < du: # basically, if too close to the control limit, perturb in the opposite direction 
            du *= -1 


        u1 = max(0, u0 + du) # Guaranteed to be in the interval [0, pi/2]
        z0 = self.predict(state, energy, u0)
        z1 = self.predict(state, energy, u1)

        dzdu = (z1-z0)/du 
        unew = correct()


    def correct(self, u0, z0, dzdu):
        
        for i in range(6):
            u = u0 + 0.5**i * z0/dzdu 
            u = np.clip(u, 0, np.pi/2) 
            

    def predict(self, state, energy, bank):
        # Integrate current state to final energy 
        x = odeint(self.model.dynamics(bank), state, [energy, self.ef])

        sf = x[-1][1] # TODO make sure this index matches the range state for whatever model I use 
        z = sf - self.sf # range error

        # self.previous = [bank, z]
        return z 


