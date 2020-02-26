import pickle
import numpy as np
import chaospy as cp 
import matplotlib.pyplot as plt 

from scipy.interpolate import interp1d
from scipy.io import loadmat, savemat

import sys, os
sys.path.append("./")

from Utils.RK4 import RK4
from Utils.boxgrid import boxgrid 

from EntryGuidance.EntryEquations import Entry, EDL
from EntryGuidance.Simulation import Simulation, Cycle, EntrySim
from EntryGuidance.InitialState import InitialState
from EntryGuidance.Planet import Planet 
from EntryGuidance.SRPData import SRPData 
from EntryGuidance.SRPUtils import range_from_entry, srp_from_entry
from EntryGuidance.Target import Target 
from EntryGuidance.VMC import VMC, velocity_trigger


def reversal_controller(bank, v_reverse, vectorized):
    """ This version requires v_reverse to be the same length as the state
    This is so that we can quickly determine where a reversal should go 
    
    """
    if vectorized:
        def _control(e,x,l,d):
            sigma = np.ones_like(x[0])*bank
            sigma[np.less_equal(x[3], v_reverse)] *= -1 
            return sigma
        return _control
    else:
        def _control(v):
            sigma = bank
            if v < v_reverse:
                sigma = -bank 
            return sigma
        return _control

def switch_controller(v_reverse, vectorized):
    b1 = 90
    b2 = 15
    if vectorized:
        def _control(e,x,l,d):
            v = x[3]
            sigma = np.radians(b1)*np.ones_like(e)
            sigma[np.less_equal(v, v_reverse[0])] = -np.radians(b2)
            sigma[np.less_equal(v, v_reverse[1])] = np.radians(b2)
            return sigma
    else:
        def _control(v):
            sigma = np.radians(b1)
            if v <= v_reverse[0]:
                sigma = -np.radians(b2)
            if v <= v_reverse[1]:
                sigma = np.radians(b2)
            return sigma
    return _control


class SRPController:
    
    """ From current state, integrate a profile forward in a vectorized manner, 
    select the best one -  requires both determining the best/correct endpoint of a given trajectory, 
    as well as the best trajectories from the set and interpolating 
    
    Given that we will have a target altitude at the end of the deceleration phase (site altitude + margin)
    An altitude trigger set to that altitude makes sense, as the ignition point is guaranteed to be earlier 
    
    """
    
    def __init__(self, N, target, srpdata=None, update_function=None, debug=False, constant_controller=True, time_constant=2):
        
        self.profile = None 
        self.constant_controller = constant_controller
        self.time_constant = time_constant
        
        self.banksign = 1 # Used as a record of the previous commanded bank sign 
        self.debug = debug 
        self.N = N
        self.target = target 
        self.update = update_function
        self.vmc = VMC()
        self.vmc.null_sample(np.product(N)) 
        

        self.srpdata = srpdata 
            
        self.vmc.set_trigger(velocity_trigger(500))
        self.control_params = []  # These are actual the whole space of control params over which we are searching for the optimum       
        
        # Construct a simulation here for use in predictions
        Vf = 400 
        self.sim = Simulation(cycle=Cycle(0.1), output=False, use_da=False, **EntrySim(Vf=Vf), )
        
        self.history = {"n_updates": 0,  "params": [], "n_reversals": 0 ,"fuel": [], "state": [], "velocity": []}


    def plot_history(self):
        h = self.history
        update = list(range(h['n_updates']+1))
        
        plt.figure(figsize=(8,14))
        plt.subplot(2, 1, 1)
        plt.plot(update, h['fuel'], 'o-')
        plt.ylabel("Estimated Optimal Fuel Required (kg)")
        plt.subplot(2, 1, 2)
        plt.plot(update, h['params'], 'o-')
        plt.xlabel("Update #")
        plt.ylabel("Optimal Parameters")
        
        
    def predict(self, state):
        # A method that integrates forward from the current state and gets the predicted fuel performance
        # The update method can utilize this to check if an update should occur (if the discrepancy is large etc)
        
        def ref_profile(velocity, **args):
            sigma = self.profile(velocity)
            return sigma 
        self.sim.run(state, [ref_profile], TimeConstant=self.time_constant) # Make sure the time constant matches what we use in the VMC in __call__
        mf = self.srp_trim(self.sim.history)
        return mf 
        
    def srp_trim(self, traj):
        return self.srpdata.srp_trim(traj, self.target)


    def set_profile(self, params):
        if self.constant_controller:
            self.profile = reversal_controller(*params, vectorized=False)
        else:
            self.profile = switch_controller(params, False)
        

    def set_control_params(self, velocity,):
        """ This function sets the current search space for the vectorized monte carlo """

        if self.constant_controller: # single reversal controller 
            
            current_bank, current_reverse = self.history['params'][-1]
            
#             bank_range = np.radians([10, 50]) ## 33 to 47 should cover the majority of EFPA variations 
            if 1:
                # Tight 
                bank_range = current_bank + np.radians([-1.0, 1.0]) # should check for bounds like 0 though 
                reversal_range = current_reverse + np.array([-100, 100])
            else:
                # Loose
                bank_range = np.radians([10, 30])
                # bank_range = current_bank + np.radians([-15, 15]) # should check for bounds like 0 though 
                reversal_range = current_reverse + np.array([-500, 500])


            bank_range[0] = max(0, bank_range[0])
            
#             reversal_range = [min(velocity-100, 2000), min(velocity, 4000)] # The current velocity is the upper bound on when a reversal could occur, the lower bound is fixed until below the lower bound 
            B,Vr = boxgrid([bank_range, reversal_range], self.N, interior=True).T

            self.control_params = (B,Vr)
            self.vmc.control = reversal_controller(B, Vr, vectorized=True)
        else:
            if velocity >= self.history['params'][-1][0]:
#                 V1,V2 = boxgrid([[min(4600, velocity-400), min(5000, velocity+2)], [500, 1200]], self.N, interior=True).T  # Allowing 0 set of points above the current velocity means no 90 deg arc 
#                 V1,V2 = boxgrid([[min(4600, velocity-400), min(4800, velocity+2)], [750, 950]], self.N, interior=True).T  # A tight set for plotting, not for actual guidance 
                V1,V2 = boxgrid([[min(4400, velocity-400), min(4900, velocity+2)], [750, 1100]], self.N, interior=True).T
            else: # After we've done the first reversal, only check the second one 
                V0 = self.history['params'][-1][0]
                V1,V2 = boxgrid([[V0, V0], [500, 1200]], [1, np.product(self.N)], interior=True).T

            self.control_params = (V1,V2)
            self.vmc.control = switch_controller([V1,V2], True)
        

    def __call__(self, current_state, **kwargs):
        """Maintain a bank angle profile
            Replan either at fixed intervals or at certain variables 
        
        """
        
        # Initialize on the fly if needed
        if self.profile is None:
            if self.constant_controller:
                banks = [0.5070640774215105, 0.7186126052854204, 0.6159175070853673, 0.808363021976321, 0.7670262765343501, 0.66598089878731, 0.7909097294563778, 0.8993038619486571,0.1974977837783057, 0.3812166524092878, 0.08, 1.028]
                efpa = np.radians([-16.3-0.5, -16.3, -16.3-0.25,-16.3+0.25, -16.3+0.125, -16.3-0.125, -16.1, -16.3+0.5, -16.3-1, -16.3-0.75, -17.4, -15.4])
                vr = [2847.3684210526317, 2935, 2895, 3021, 2957.89,  2910.526, 2989.47, 3126.315789473684, 2757.8947368421054, 2789.4736842105262, 2650,  3332.95]
                azi = [-0.5, -0.25, 0, 0.25, 0.5] # currently just for -16.9 deg 
#                 vra = [3310, 3069, 2824, 2624, 2393] # "gradient" is roughly -950 m/s per deg  

                k = np.argsort(efpa)
                efpa = np.array(efpa)[k]
                banks = np.array(banks)[k]
                vr = np.array(vr)[k]
                
                b = interp1d(efpa, banks, fill_value='extrapolate')(current_state[4])
                v0 = interp1d(efpa, vr, fill_value='extrapolate')(current_state[4])
                dv0 = np.degrees(current_state[5])*-950    # correction for azimuth 
                v0 += dv0
                if 1 or self.debug:
                    print("Initialized to {:.2f} deg, {:.1f} m/s".format(np.degrees(b), v0))
                self.set_profile((b,v0))
                self.history['params'].append((b,v0))
            else:
                assert False, "Reversal controller params must be set before calling "
        
        # Determine the current bank angle from the profile
        v = current_state[3]
        
        if self.update(self.history, current_state, **kwargs):
#             if self.debug:
            print("Update triggered...")
            if 0: # Just used for debugging to turn the actual replanning off 
                print(self.history)
                self.history['n_updates'] += 1
                self.history['params'].append(self.history['params'][-1])
                self.history['fuel'].append(self.history['fuel'][-1])
#                 self.history['state'].append(current_state)
                self.set_control_params(current_state[3],)

            else:
                self.set_control_params(current_state[3],)
                self.vmc.run(current_state, save=False, stepsize=[1, 0.05, 10], time_constant=self.time_constant)
                if self.debug:
                    self.vmc.plot()
                    # self.vmc.plot_trajectories()

                if self.target is not None:
                    self.vmc.srp_trim(self.srpdata, self.target, vmax=790, hmin=2000, optimize=False)
                    if self.debug:
                        self.vmc.plot_srp(max_fuel_use=3000)
                    fuel = np.array(self.vmc.mc_srp['fuel'])
                    keep =  fuel < np.mean(fuel) # keep anything under the average 
                    if not np.any(keep):
                        print("No viable solution under {:.1f} kg found in current search space".format(np.min(fuel)*2))
                    if np.any(keep):
                        opt = np.argmin(fuel)
                        params = self.control_params[0][opt], self.control_params[1][opt]
                        traj = self.vmc.mc[opt]
#                         print(np.shape(traj))
                        self.set_profile(params)
#                         mf = self.predict(current_state)   
#                         print(np.shape(self.sim.history))
                
                        # plot the single integration trajectory and the optimal one from the vectorized to determine why they are different... 
#                         for x,label in zip([traj, self.sim.history],['Single','Vector']):
#                             r,th,ph,V,fpa,psi,m = x.T 
#                             h = (r-3397e3)/1000
#                             for i,state in enumerate([h, np.degrees(th), np.degrees(ph), np.degrees(fpa), np.degrees(psi)]):
                            
#                                 plt.figure(60+i)
#                                 plt.plot(V, state, label=label)
#                                 plt.xlabel('Velocity')
#                                 plt.legend()

                        self.history['n_updates'] += 1 
                        self.history['params'].append(params)
                        self.history['fuel'].append(fuel[opt]) # should we use the single prediction? why don't they match better
                        self.history['velocity'].append(np.linalg.norm(self.vmc.mc_srp['ignition_state'][opt][3:]))
                        self.history['state'].append(current_state)

                        if 1: # diagnostics 
                            # print("Target DR = {:.1f} km".format(self.target.longitude*3397))
                            print("- Optimum: {:.1f} kg at {:.1f} deg, reversal at {:.1f}".format(fuel[opt], np.degrees(self.control_params[0][opt]), self.control_params[1][opt]))
                            print("- Predicted Ignition velocity = {:.1f} m/s".format(np.linalg.norm(self.vmc.mc_srp['ignition_state'][opt][3:])))
#                             print("- Optimum: {:.1f} kg, {}".format(fuel[opt], params))
#                             print("Predicted fuel consumption from single integration: {:.1f} kg".format(mf))
                            print("Ignition State:")
                            for state in self.vmc.mc_srp['ignition_state'][opt]:
                                print("{:.1f}".format(state))

                        if self.debug:
                            figsize = (10, 10)
                            cmap = 'inferno'
                            if self.constant_controller:
                                plt.figure(figsize=figsize)
#                                 plt.scatter(np.degrees(self.control_params[0][keep]), self.control_params[1][keep], c=fuel[keep])
                                plt.tricontourf(np.degrees(self.control_params[0][keep]), self.control_params[1][keep], fuel[keep], 15, cmap=cmap)
                                plt.plot(np.degrees(self.control_params[0][opt]), self.control_params[1][opt], 'rx', markersize=8)
                                plt.xlabel("Bank Angle (deg)")
                                plt.ylabel("Reversal Velocity (m/s)")
                                plt.title("Fuel Usage (kg)")
                                plt.colorbar()
                            else:
                                plt.figure(figsize=figsize)
#                                 plt.scatter(self.control_params[0][keep], self.control_params[1][keep], c=fuel[keep])
                                plt.tricontourf(self.control_params[0][keep], self.control_params[1][keep], fuel[keep], 15, cmap=cmap)
                                plt.plot(self.control_params[0][opt], self.control_params[1][opt], 'rx', markersize=8)
                                plt.xlabel("Reversal 1 Velocity (m/s)")
                                plt.ylabel("Reversal 2 Velocity (m/s)")
                                plt.title("Fuel Usage (kg)")
                                plt.colorbar()

                        
        # Whether or not we just updated the profile, call it!
        bank = self.profile(v)

        # Capture the number of reversals 
        s = np.sign(bank)
        if s == 0:
            s = 1
        if not s == self.banksign:
            self.history['n_reversals'] += 1
            self.banksign =  s
        return bank 



def update_rule(history, state, **kwargs):
    r,th,ph,v,gamma,psi,m = state
    
    Vr = [6500, 3000, 1500] # The velocities at which to update 
    # Put a high velocity to trigger an initial planning, but in general we should start with a nominal stored already, and not update till later 
    
    return np.any([(history['n_updates'] == i and v <= vr) for i, vr in enumerate(Vr)])


def test_single():
    x0 = InitialState(vehicle='heavy', fpa=np.radians(-16.9))    
    target = Target(0, 753.7/3397, 0)  # This is what all the studies have been done with so far 

    srpdata = pickle.load(open(os.path.join(os.getcwd(), "data\\FuelOptimal\\srp_27k_5d.pkl"),'rb'))

    mcc = SRPController(N=[10, 300], target=target, srpdata=srpdata, update_function=update_rule, debug=True, time_constant=2)
    mcc(x0)
    plt.show()


def test_sim():
    """ Runs a scalar simulation with the SRP Controller called multiple times """
    x0 = InitialState(vehicle='heavy', fpa=np.radians(-16.3))    
    target = Target(0, 753.7/3397, 0)   # This is what all the studies have been done with so far 
    TC = 2      # time constant 
    srpdata = pickle.load(open(os.path.join(os.getcwd(), "data\\FuelOptimal\\srp_27k_5d.pkl"),'rb'))

    mcc = SRPController(N=[20, 20], target=target, srpdata=srpdata, update_function=update_rule, debug=True, time_constant=TC)
    Vf = 500     # Anything lower than the optimal trigger point is fine 
    sim = Simulation(cycle=Cycle(1), output=True, use_da=False, **EntrySim(Vf=Vf), )
    sim.run(x0, [mcc], TimeConstant=TC) 

    mf = mcc.srp_trim(sim.history)
    print("Fuel consumed: {:.1f} kg".format(mf))
    sim.plot()
    mcc.plot_history()


def test_sweep():
 
    target = Target(0, 753.7/3397, 0)  # This is what all the studies have been done with so far 
    srpdata = pickle.load(open(os.path.join(os.getcwd(), "data\\FuelOptimal\\srp_27k_5d.pkl"),'rb'))

    # FPAs = np.linspace(-17.2, -15.8, 6)
    # FPAs = [-17.39]
    # AZIs = [-0.5, -0.25, 0, 0.25, 0.5]
    # AZIs = [0.]
    ic = boxgrid([(-17.2, -15.8), (-0.5, 0.5)], [15, 9], interior=True)
    # ic = boxgrid([(-15, -13.5), (0,0)], [4,1], interior=True)

    data = []
    for efpa, azi in ic:
        print("\n Entry FPA/AZI:")
        print(efpa, azi)
        x0 = InitialState(vehicle='heavy', fpa=np.radians(efpa), psi=np.radians(azi)) 
        for boolean in [1,]:
            mcc = SRPController(N=[30, 100], target=target, srpdata=srpdata, update_function=update_rule, debug=False, constant_controller=boolean)
            # mcc = SRPController(N=[3, 5], target=target, srpdata=srpdata, update_function=update_rule, debug=False, constant_controller=boolean)
            mcc(x0)
            data.append([mcc.history['fuel'][-1], *mcc.history['params'][-1]])

    data = np.array(data).T
    data[1] = np.degrees(data[1])

    plt.figure()
    plt.tricontourf(ic.T[0], ic.T[1], data[0])
    plt.xlabel("EFPA (deg)")
    plt.ylabel("Azimuth Error (deg)")
    plt.colorbar(label="Fuel Required (kg)")

    plt.figure()
    plt.tricontourf(ic.T[0], ic.T[1], data[1])
    plt.xlabel("EFPA (deg)")
    plt.ylabel("Azimuth Error (deg)")
    plt.colorbar(label="Bank Angle (deg)")

    plt.figure()
    plt.tricontourf(ic.T[0], ic.T[1], data[2])
    plt.xlabel("EFPA (deg)")
    plt.ylabel("Azimuth Error (deg)")
    plt.colorbar(label="Reversal Velocity (m/s)")

    # optimal parameters and fuel use vs efpa 
    # for d in data:
    #     plt.figure()
    #     plt.plot(ic.T[0], d)
    #     plt.xlabel("EFPA (deg)")

    # for d in data:
    #     plt.figure()
    #     plt.plot(AZIs, d)
    #     plt.xlabel("EAZI (deg)")

    plt.show()

    import pandas as pd 
    df = pd.DataFrame(data.T, columns=["fuel","p1","p2"])
    df.to_csv("temp.csv")



# def __objective(p, sim):


def optimize():
    """ Compares sequential univariate optimization to the vectorized solution """
    Vf = 500 
    sim = Simulation(cycle=Cycle(0.25), output=False, use_da=False, **EntrySim(Vf=Vf), )


if __name__ == "__main__":
    test_single()
    # test_sweep()