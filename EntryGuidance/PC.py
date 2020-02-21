""" Predictor Corrector Entry Guidance

Implements the PC method for range control 

 """

import pickle, os 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d 

# import sys 
from VMC import VMC 
from Target import Target 
from SRPData import SRPData 


def constant_controller(banks):
    def _control(e,x,l,d):
        return banks
    return _control


class PredictorCorrector:

    def __init__(self, target_range, final_energy):
        self.sf = target_range
        self.lonf = target_range/3397
        self.target = Target(0, self.lonf, 0)
        self.ef = final_energy 
        self.filters = [None, None] # First order fading memory filter values 
        self.previous = None  # Record of past control value for secant update 
        # self.model = model # model that provides the longitudinal dynamics d

        N = 30
        self.model = VMC()
        self.model.null_sample(N)
        self.banks = np.linspace(0, np.pi/2, N)
        self.model.control = constant_controller(self.banks)
        # self.srpdata = pickle.load(open(os.path.join(os.getcwd(), "data\\FuelOptimal\\srp_27k_5d.pkl"),'rb'))

    def __call__(self, current_state, **kwargs):

        self.model.run(current_state, Ef=self.ef, stepsize=[5, 0.1, 10], save=False)
        xf = self.model.xf.T # 7 x N 
        lon = xf[1].squeeze()
        Sf = lon*3397

        sigma = interp1d(Sf, self.banks, )(self.sf)
        
        
        # self.model.srp_trim(self.srpdata, self.target)
        # fuel = self.model.mc_srp['fuel']
        # fuel_opt = interp1d()  # Note we arent looking for the trajectory with minimal use 

        # plt.figure()
        # plt.plot(np.degrees(self.banks).squeeze(), Sf)
        # plt.hlines(self.sf, 0, 90, label="Target Range")
        # plt.vlines(np.degrees(sigma), np.min(Sf), np.max(Sf), 'r', label="Optimum")
        # plt.xlabel("Bank Angle (deg)")
        # plt.ylabel("Range (km)")
        # plt.show()

        return sigma 



if __name__ == "__main__":
    from InitialState import InitialState
    x0 = InitialState(vehicle="heavy", fpa=np.radians(-16.9))

    ef = 0.5*500**2
    pc = PredictorCorrector(753.7, ef, )
    # pc(x0)

    Vf = np.linspace(0, 800, 40)
    Sigma = []
    for vf in Vf:
        ef = 0.5*vf**2
        pc.ef =  ef 
        sigma = pc(x0)
        
        Sigma.append(sigma)

    Ef = 0.5*Vf**2
    plt.figure()
    plt.plot(Vf, np.degrees(Sigma))
    plt.xlabel("Terminal Velocity used to determine terminal energy (m/s)")
    plt.ylabel("Optimal Bank Angle to Achieve Target Range (deg)")

    plt.figure()
    plt.plot(Ef, np.degrees(Sigma))
    plt.xlabel("Terminal energy (m^2/s^2)")
    plt.ylabel("Optimal Bank Angle to Achieve Target Range (deg)")
    plt.show()