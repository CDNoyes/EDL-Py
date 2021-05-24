import pickle
import time 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

from scipy.interpolate import interp1d, Rbf
from scipy.io import loadmat, savemat

import sys, os
sys.path.append("./")

from Utils.RK4 import RK4
from Utils.boxgrid import boxgrid 
from Utils.compare import compare 

from EntryGuidance.EntryEquations import Entry, EDL
from EntryGuidance.Simulation import Simulation, Cycle, EntrySim
from EntryGuidance.InitialState import InitialState
from EntryGuidance.Planet import Planet 
from EntryGuidance.SRPData import SRPData 
from EntryGuidance.SRPUtils import range_from_entry, srp_from_entry
from EntryGuidance.Target import Target 
from EntryGuidance.VMC import VMC, velocity_trigger
from EntryGuidance.Triggers import Trigger, VelocityTrigger, AltitudeTrigger

SRPFILE = os.path.join(os.getcwd(), "data\\FuelOptimal\\srp_7200kg.pkl")

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
            sigma[np.less_equal(v, v_reverse[0])] = -np.radians(b1)
            sigma[np.less_equal(v, v_reverse[1])] = np.radians(b2)
            return sigma
    else:
        def _control(v):
            sigma = np.radians(b1)
            if v <= v_reverse[0]:
                sigma = -np.radians(b1)
            if v <= v_reverse[1]:
                sigma = np.radians(b2)
            return sigma
    return _control


def _objective(bank, reverse_velocity, x0, srp_trim, aero, time_constant):
    """ This auxiliary function is used by SRPController.optimize method """
    Vf = 450
    states = ['Bank1','Bank2']
    trigger = [VelocityTrigger(reverse_velocity), VelocityTrigger(Vf)]
    data = {'states':states, 'conditions': trigger}
    DT = 1
    sim = Simulation(cycle=Cycle(DT), output=False, use_da=False, **data)
    # control = [reversal_controller(bank, reverse_velocity, vectorized=False)]*2

    control = [lambda **d: bank, lambda **d: -bank]
    sim.run(x0, control, AeroRatios=aero, TimeConstant=time_constant, StepsPerCycle=10)
    # sim.plot()
    # plt.show()
    m0 = srp_trim(sim.history)
    return m0


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
        
        self.aeroscale = np.array([1, 1]) # Lift and drag modifiers in the prediction 

        self.srpdata = srpdata 
            
        self.vmc.set_trigger(velocity_trigger(500))
        self.control_params = []  # These are actual the whole space of control params over which we are searching for the optimum       
        
        # Construct a simulation here for use in predictions
        Vf = 450 
        self.sim = Simulation(cycle=Cycle(1), output=False, use_da=False, **EntrySim(Vf=Vf), )
        self.sim.run([2000+3397e3, self.target.longitude*0.99, self.target.latitude, Vf + 10, np.radians(-10), 0, 7200], [lambda **d: 0])
        # self.predict() # This is just to initialize an edlModel
        self.history = {"n_updates": 0,  "params": [], "n_reversals": 0 ,"fuel": [], "state": [], "velocity": [], 'entry_state': [], 'ignition_state': []}     


    def plot_history(self):
        h = self.history
        update = list(range(1, 1+h['n_updates']))
        param = np.array(h['params'])

        N = 4

        plt.figure(figsize=(8,14))
        plt.subplot(N, 1, 1)
        plt.plot(update, h['fuel'], 'o-')
        plt.ylabel("Estimated \nFuel Required (kg)")
        if self.constant_controller:
            plt.subplot(N, 1, 2)
            plt.plot(update, np.degrees(param.T[0][1:]), 'o-')
            plt.xlabel("Update #")
            plt.ylabel("Optimal Bank (deg)")

            plt.subplot(N, 1, 3)
            plt.plot(update, param.T[1][1:], 'o-')
            plt.xlabel("Update #")
            plt.ylabel("Optimal Reversal (m/s)")
        else:
            plt.subplot(N, 1, 2)
            plt.plot(update, (param.T[0]), 'o-')
            plt.xlabel("Update #")
            plt.ylabel("Optimal V1 (m/s)")

            plt.subplot(N, 1, 3)
            plt.plot(update, param.T[1], 'o-')
            plt.xlabel("Update #")
            plt.ylabel("Optimal V2 (m/s)")

        plt.subplot(N, 1, 4)
        plt.plot(update, h['velocity'], 'o-')
        plt.xlabel("Update #")
        plt.ylabel("Optimal Ignition Velocity (m/s)")

        
    def predict(self, state):
        # A method that integrates forward from the current state and gets the predicted fuel performance
        # The update method can utilize this to check if an update should occur (if the discrepancy is large etc)
        
        # def ref_profile(velocity, **args):
        #     sigma = self.profile(velocity)
        #     return sigma 
        # self.sim.run(state, [ref_profile], TimeConstant=self.time_constant, StepsPerCycle=10) # Make sure the time constant matches what we use in the VMC in __call__
        
        Vf = 450
        states = ['Bank1','Bank2']
        bank = self.history['params'][-1][0]
        reverse_velocity = self.history['params'][-1][1]
        trigger = [VelocityTrigger(reverse_velocity), VelocityTrigger(Vf)]
        data = {'states':states, 'conditions': trigger}
        DT = 1
        self.sim = Simulation(cycle=Cycle(DT), output=False, use_da=False, **data)
        # control = [reversal_controller(bank, reverse_velocity, vectorized=False)]*2

        control = [lambda **d: bank, lambda **d: -bank]
        self.sim.run(state, control, AeroRatios=self.aeroscale, TimeConstant=self.time_constant, StepsPerCycle=10)


        data = self.srp_trim(self.sim.history, full_return=True)
        return data 
        

    def srp_trim(self, traj, *args, **kwargs):
        """Utility to allow direct calls to srp_trim without passing the target"""
        return self.srpdata.srp_trim(traj, self.target, *args, **kwargs)


    def set_profile(self, params):
        if self.constant_controller:
            self.profile = reversal_controller(*params, vectorized=False)
        else:
            self.profile = switch_controller(params, False)
        

    def set_control_params(self, velocity,):
        """ This function sets the current search space for the vectorized monte carlo """

        # set the originally null sampled points to the current aero scale factors for a better prediction
        # if self.debug:
        if self.vmc.samples is not None:
            print("...using aero scale factors: {:.2f} (lift), {:.2f} (drag)".format(*self.aeroscale))
            self.vmc.samples[0] = self.aeroscale[1] - 1
            self.vmc.samples[1] = self.aeroscale[0] - 1

        if self.constant_controller: # single reversal controller 
            
            current_bank, current_reverse = self.history['params'][-1]
            
            if velocity >= self.history['params'][-1][1]:
                N = self.N 
    #             bank_range = np.radians([10, 50]) ## 33 to 47 should cover the majority of EFPA variations 
                if 0:
                    # Tight 
                    bank_range = current_bank + np.radians([-3.0, 3.0]) # should check for bounds like 0 though 
                    reversal_range = current_reverse + np.array([-100, 100])

                else:
                    # Loose
                    bank_range = current_bank + np.radians([-10, 30])
                    # bank_range = np.radians([5, 60])
                    # reversal_range = current_reverse + np.array([-2500, 2500])
                    reversal_range = np.array([500, 2200]) #current_reverse + np.array([-250, 250])
                    self.vmc.null_sample(np.product(N)) 

            else: # After reversal, really hone in on the correct bank for the remainder 
                bank_range = current_bank + np.radians([-10, 10])
                reversal_range = [current_reverse, current_reverse]
                N = [np.product(self.N),1]
                self.vmc.null_sample(N[0]) 


            bank_range[0] = max(0, bank_range[0])
            
#             reversal_range = [min(velocity-100, 2000), min(velocity, 4000)] # The current velocity is the upper bound on when a reversal could occur, the lower bound is fixed until below the lower bound 
            B,Vr = boxgrid([bank_range, reversal_range], N, interior=True).T

            self.control_params = (B,Vr)
            self.vmc.control = reversal_controller(B, Vr, vectorized=True)

        else:
            if not self.history['params'] or velocity >= self.history['params'][-1][1]:
                # V1,V2 = boxgrid([[min(4600, velocity-400), min(5000, velocity+2)], [500, 1200]], self.N, interior=True).T  # Allowing 0 set of points above the current velocity means no 90 deg arc 
#                 V1,V2 = boxgrid([[min(4600, velocity-400), min(4800, velocity+2)], [750, 950]], self.N, interior=True).T  # A tight set for plotting, not for actual guidance 
                V1,V2 = boxgrid([[min(4400, velocity-400), min(5400, velocity+2)], [2000, 4200]], self.N, interior=True).T
                # V1,V2 = boxgrid([[4600, 5480], [3000, 4600]], self.N, interior=True).T

            else: # After we've done the first reversal, only check the second one 
                if velocity >= self.history['params'][-1][1]:
                    V0 = self.history['params'][-1][0]
                    V1,V2 = boxgrid([[V0, V0], [2000, 4200]], [1, 10*self.N[1]], interior=True).T
                    self.vmc.null_sample(V2.size) 
                else: # No more optimization - in reality switch to the constant parametrization
                    self.constant_controller = True 
                    # self.history['params'][-1] = list(self.history['params'][-1])
                    self.history['params'].append(list(self.history['params'][-1]))
                    self.history['params'][-1][0] = np.radians(15)
                    self.history['params'][-1][1] = (velocity - 500)/2.0
                    self.set_control_params(velocity)
                    return 
                    # V0 = self.history['params'][-1][0]
                    # V1 = self.history['params'][-1][1]
                    # V1,V2 = boxgrid([[V0, V0], [V1, V1]], [1, 1], interior=True).T
                    # self.vmc.null_sample(1)
            self.control_params = (V1,V2)
            self.vmc.control = switch_controller([V1,V2], True)
        

    def __call__(self, current_state, lift=None, drag=None, **kwargs):
        """ The primary controller method
        Returns the value of the bank profile based on the current velocity
        Bank profile is updated according to the update rule provided

        Possible update rules included at fixed state variables such as velocity, or a measured drag value
        Adaptive updates, such as when the predicted fuel required has grown (or changed) by more than a threshold value 

        
        """

        r,th,ph,v,fpa,azi,m = current_state

        # Aero ratio update - no filter, just uses the current ratio to predict later 
        if lift is not None:
            self.sim.edlModel.update_ratios(1,1) # this ensures we get nominal values 
            Lm, Dm = self.sim.edlModel.aeroforces(r, v, m)  # Nominal model values
            if v > 4500:
                self.aeroscale = np.clip([lift/Lm, drag/Dm], 0.9, 1.1) # basically, never let the control freak out over massive estimates due to scale height 
            else:
                self.aeroscale = np.clip([lift/Lm, drag/Dm], 0.75, 1.25) # basically, never let the control freak out over massive estimates due to scale height 
        
        # TODO: make this an init method or something 
        # Initialize on the fly if needed
        if self.profile is None:
            if self.constant_controller:
                df = pd.read_csv("./data/FuelOptimal/srp_params.csv")
    
                data = df.values.T[1:] # remove the index column 
                bank_data = data[1]
                v0_data = data[2]
                ic = np.radians(data[3:5]) # TODO: check if the inputs are inside the box defined by these points. The RBFs will extrapolate, often with poor results.
                bank_model = Rbf(*ic, data[1], function="Linear")
                b = np.radians(bank_model(fpa, azi))
                rev_model = Rbf(*ic, data[2], function="Linear")
                v0 = rev_model(fpa, azi)
                if lift is not None:
                    b -= np.radians(np.clip(80 * (1-self.aeroscale[0]), -10, 10))  # update initial bank guess for lift factor 
                    b += np.radians(np.clip(100 * (1-self.aeroscale[1]), -12, 12))  # update initial bank guess for drag factor 
                    b = np.clip(b, 0, np.pi/2) # ensure the above do not push the bank angle into negative territory 
                # if  1:
                #     print("FPA outside data region, using a linear fit")
                #     b = np.radians(58 + 12*(np.degrees(fpa)+14.7))
                # v0 = 2702.8 - 500 * np.degrees(azi)
                if 1 or self.debug:
                    print("SRP Controller initialized to {:.2f} deg, {:.1f} m/s".format(np.degrees(b), v0))
                self.set_profile((b,v0))
                self.history['params'].append((b,v0))
            else:
                self.set_profile((3500, 1500))
                # assert False, "Reversal controller params must be set before calling "
        
       
        if self.update(self.history, current_state, **kwargs):
            if 'time' in kwargs:
                print("Update triggered... (sim time = {:.0f} s, current velocity = {:.1f} m/s)".format(kwargs['time'], v))
            # print("Current aero ratios: {}".format(self.aeroscale))
            # if self.debug:
            if 0: # Just used for debugging to turn the actual replanning off 
                print(self.history)
                self.history['n_updates'] += 1
                self.history['params'].append(self.history['params'][-1])
                self.history['fuel'].append(self.history['fuel'][-1])
#                 self.history['state'].append(current_state)
                self.set_control_params(current_state[3],)

            else:

                if 0:  # Sequential optimization based parameter updates
                    # sol = self.optimize_mc(current_state, max_iters=3)    # Repeated 1-D VMC based 
                    if 0 and current_state[3] <= self.history['params'][-1][1]: # past the reversal velocity 
                        sol = self.optimize_nonlinear(current_state, method='SLSQP', scalar=True)           # Nonlinear optimization-based 
                        # sol = self.optimize(current_state, verbose=True)      # 1-D optimization based 
                    else:
                        sol = self.optimize_nonlinear(current_state, )           # Nonlinear optimization-based 'Powell' or 'Nelder-Mead'

                    params = sol['params']
                    fuel = sol['fuel']
                    self.set_profile(params)
                    self.history['params'].append(params)

                    if 1: # diagnostics 
                        print("- Optimum: {:.1f} kg at {:.1f} deg, reversal at {:.1f}".format(fuel, np.degrees(params[0]), params[1]))
                        # print("Ignition State:")
                        # for state in data['ignition_state']:
                        #     print("{:.1f}".format(state))

                    # Don't actually have these with these method, have to integrate one more time to get state/vel
                    try:
                        data = self.predict(current_state)

                        if self.debug:
                            print("Optimization: {:.2f} kg\nIntegration: {:.2f} kg".format(fuel, data['fuel']))

                        self.history['n_updates'] += 1 
                        self.history['fuel'].append(fuel) 
                        self.history['state'].append(current_state)
                        self.history['entry_state'].append(data['terminal_state'])
                        self.history['ignition_state'].append(data['ignition_state'])
                        self.history['velocity'].append(np.linalg.norm(data['ignition_state'][3:]))
                        if 1:
                            print("- Predicted Ignition velocity = {:.1f} m/s".format(self.history['velocity'][-1]))

                    except IndexError: # no solution was found 
                        print("No solution found") 

                else: # Vectorized monte carlo to brute force a solution, but also good because of the plots it generates 
                    self.set_control_params(current_state[3],)
                    # self.vmc.run(current_state, save=False, stepsize=[1, 0.05, 10], time_constant=self.time_constant)
                    self.vmc.run(current_state, save=False, stepsize=0.5, time_constant=self.time_constant)
                    # if self.debug:
                        # self.vmc.plot()
                    self.vmc.plot_trajectories()

                    self.vmc.srp_trim(self.srpdata, self.target, vmax=700, hmin=3000, optimize=False)
                    # if self.debug:
                    #     self.vmc.plot_srp(max_fuel_use=3000)
                    fuel = np.array(self.vmc.mc_srp['fuel'])
                    # keep =  fuel < self.srpdata.mmax # keep anything under the max in the table  
                    keep =  fuel < np.min(fuel)*1.5 # keep anything under the max in the table  
                    if not np.any(keep):
                        print("No viable solution under {:.1f} kg found in current search space".format(np.min(fuel)*2))
                        self.history['n_updates'] += 1 # should add the previous history as the new one or the plot_history will fail 

                    else:
                        opt = np.argmin(fuel)
                        params = self.control_params[0][opt], self.control_params[1][opt]
                        traj = self.vmc.mc[opt]
    #                         print(np.shape(traj))
                        self.set_profile(params)

                        self.history['n_updates'] += 1 
                        self.history['params'].append(params)
                        self.history['fuel'].append(fuel[opt]) # should we use the single prediction? why don't they match better
                        self.history['velocity'].append(np.linalg.norm(self.vmc.mc_srp['ignition_state'][opt][3:]))
                        self.history['state'].append(current_state)
                        self.history['entry_state'].append(self.vmc.mc_srp['terminal_state'][opt])
                        self.history['ignition_state'].append(self.vmc.mc_srp['ignition_state'][opt])

                        if 1: # diagnostics 
                            # print("Target DR = {:.1f} km".format(self.target.longitude*3397))
                            if self.constant_controller:
                                print("- Optimum: {:.1f} kg at {:.1f} deg, reversal at {:.1f}".format(fuel[opt], np.degrees(self.control_params[0][opt]), self.control_params[1][opt]))
                            else:
                                print("- Optimum: {:.1f} kg at v1 = {:.1f} m/s, v2 = {:.1f} m/s".format(fuel[opt], self.control_params[0][opt], self.control_params[1][opt]))
                            print("- Predicted Ignition velocity = {:.1f} m/s".format(np.linalg.norm(self.vmc.mc_srp['ignition_state'][opt][3:])))
    #                             print("- Optimum: {:.1f} kg, {}".format(fuel[opt], params))
    #                             print("Predicted fuel consumption from single integration: {:.1f} kg".format(mf))
                            print("Ignition State:")
                            for state in self.vmc.mc_srp['ignition_state'][opt]:
                                print("{:.1f}".format(state))

                        if self.debug:
                            figsize = (7, 4)
                            fontsize=14
                            ticksize=fontsize-2

                            cmap = 'inferno'
                            if self.constant_controller:
                                plt.figure(figsize=figsize)
    #                                 plt.scatter(np.degrees(self.control_params[0][keep]), self.control_params[1][keep], c=fuel[keep])
                                plt.tricontourf(np.degrees(self.control_params[0][keep]), self.control_params[1][keep], 100*fuel[keep]/7200, 15, cmap=cmap)
                                # plt.plot(np.degrees(self.control_params[0][opt]), self.control_params[1][opt], 'rx', markersize=8)
                                plt.xlabel(r"Bank Angle Magnitude, $\sigma_c$ (deg)", fontsize=fontsize)
                                plt.ylabel(r"Reversal Velocity, $v_r$ (m/s)", fontsize=fontsize)
                                cbar = plt.colorbar()
                                cbar.set_label("PMF for Pinpoint Landing (%)")
                                cbar.ax.tick_params(labelsize=ticksize)
                                cbar.ax.yaxis.label.set_size(fontsize)
                                plt.tick_params(labelsize=ticksize)

                            else:
                                plt.figure(figsize=figsize)
    #                                 plt.scatter(self.control_params[0][keep], self.control_params[1][keep], c=fuel[keep])
                                plt.tricontourf(self.control_params[0][keep], self.control_params[1][keep], 100*fuel[keep]/7200, 15, cmap=cmap)
                                # plt.plot(self.control_params[0][opt], self.control_params[1][opt], 'rx', markersize=8)
                                plt.xlabel(r"Reversal 1 Velocity, $v_1$  (m/s)", fontsize=fontsize)
                                plt.ylabel(r"Reversal 2 Velocity, $v_2$  (m/s)", fontsize=fontsize)
                                # plt.title("Propellant Usage (kg)")
                                cbar = plt.colorbar()
                                cbar.set_label("PMF for Pinpoint Landing (%)")
                                cbar.ax.tick_params(labelsize=ticksize)
                                cbar.ax.yaxis.label.set_size(fontsize)
                                plt.tick_params(labelsize=ticksize)

                        
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

    def optimize_mc(self, x0, max_iters=5):
        """ Rather than do full factorial Monte Carlos of parameters, do 1-D searches with fewer points """

        velocity = x0[3]
        improvement_tolerance = 10 # kg 
        current_bank, current_reverse = self.history['params'][-1]

        if len(self.N) == 2:
            N1,N2 = self.N
        else:
            N1 = N2 = self.N

        fuel_best = 1e6 
        for iteration in range(max_iters):
                
            # Fix the bank angle, optimize reversal          
            if not iteration or velocity <= current_reverse:  # After reversal, don't bother updating it 
                pass 

            else: 
                bank_range = [current_bank, current_bank]
                reversal_range = current_reverse + np.array([-100, 100])
                N = [1, N2]
                B,Vr = boxgrid([bank_range, reversal_range], N, interior=True).T
                self.control_params = (B,Vr)
                self.vmc.null_sample(N2) 
                self.vmc.samples[0] = self.aeroscale[0] - 1
                self.vmc.samples[1] = self.aeroscale[1] - 1
                self.vmc.control = reversal_controller(B, Vr, vectorized=True)
                self.vmc.run(x0, save=False, stepsize=[1, 0.05, 10], time_constant=self.time_constant)
                # if self.debug:
                #     self.vmc.plot()
                    # self.vmc.plot_trajectories()

                self.vmc.srp_trim(self.srpdata, self.target, vmax=690, hmin=2000, optimize=False)
                fuel = np.array(self.vmc.mc_srp['fuel'])
                opt = np.argmin(fuel)
                if fuel[opt] <= fuel_best:
                    current_bank, current_reverse = self.control_params[0][opt], self.control_params[1][opt]
                    improvement = fuel_best - fuel[opt]
                    fuel_best = min(fuel[opt], fuel_best)
                    traj = self.vmc.mc[opt]
                    print("  Solution: {:.2f} deg, {:.1f} m/s, {:.1f} kg".format(np.degrees(current_bank), current_reverse, fuel_best))
                else:
                    print("Reversal opt failed")
                    # self.set_profile(params)

            if iteration and improvement <= improvement_tolerance:
                print("Solution is only mildly improving, terminating optimization")
                break
    

            # Fix the reversal, optimize bank angle
            bank_range = current_bank + np.radians([-5, 5])
            reversal_range = [current_reverse, current_reverse]
            N = [N1, 1]
            B,Vr = boxgrid([bank_range, reversal_range], N, interior=True).T
            self.control_params = (B,Vr)
            self.vmc.null_sample(N1) 
            self.vmc.samples[0] = self.aeroscale[0] - 1
            self.vmc.samples[1] = self.aeroscale[1] - 1
            self.vmc.control = reversal_controller(B, Vr, vectorized=True)
            self.vmc.run(x0, save=False, stepsize=[1, 0.05, 10], time_constant=self.time_constant)
            # if self.debug:
            #     self.vmc.plot()
                # self.vmc.plot_trajectories()

            self.vmc.srp_trim(self.srpdata, self.target, vmax=690, hmin=2000, optimize=False)
            fuel = np.array(self.vmc.mc_srp['fuel'])
            opt = np.argmin(fuel)
            if fuel[opt] <= fuel_best:
                current_bank, current_reverse = self.control_params[0][opt], self.control_params[1][opt]
                improvement = fuel_best - fuel[opt]
                fuel_best = min(fuel_best,fuel[opt])
                traj = self.vmc.mc[opt] 
                print("  Solution: {:.2f} deg, {:.1f} m/s, {:.1f} kg".format(np.degrees(current_bank), current_reverse, fuel_best))
            else:
                print("Bank opt failed")

            if improvement <= improvement_tolerance:
                print("Solution is only mildly improving, terminating optimization")
                break


        self.set_profile((current_bank, current_reverse))
        return {'params': (current_bank, current_reverse), 'fuel': fuel_best, 'traj': traj} 

    def optimize_nonlinear(self, x0, method='Nelder-Mead', scalar=False):
        """ Optimization based approach """
        from scipy.optimize import minimize 

        aero = self.aeroscale
        target = self.target
        srpdata = self.srpdata
        optimize_trim = False 

        obj =  lambda p: _objective(*p, x0=x0, srp_trim=lambda x: srpdata.srp_trim(x, target, vmax=700, optimize=optimize_trim), aero=aero, time_constant=self.time_constant)
        p0 = self.history['params'][-1]
        vr = p0[1]
        if x0[3] < vr:
            p0[1] = 600 #max(460, vr/2) #500 
            p0[0] = np.abs(p0[0])*self.banksign
        # In theory we could do something like, check if the current is a valid solution, and if so, use tight bounds. 
        # If not, use looser bounds? 
        # t0 = time.time()
        # print(obj(p0))
        # t1 = time.time()
        # print("Took {:.2f} s".format(t1-t0))
        # bounds = [p0[0] + np.radians([-20, 20]), p0[1] + np.array([-100, 100]) ]
        bounds = [np.radians([5, 70]), vr + np.array([-100, 100]) ]


        if scalar: # scalar opt on just bank 
            obj = lambda p: _objective(p[0], reverse_velocity=vr, x0=x0, srp_trim=lambda x: srpdata.srp_trim(x, target, vmax=700, optimize=optimize_trim), aero=aero, time_constant=self.time_constant)
            p0 = p0[0]
            bounds = [bounds[0]]
            # print(vr)
            # print(bounds)
            # print(obj(p0))

        t0 = time.time()
        if method in ["Nelder-Mead", "Powell"]: # These work exactly the same every single time for some reason 
            sol = minimize(obj, p0, method='Nelder-Mead') # Works well, but fairly slow ~ 2 minutes from a good guess 
        else:
            sol = minimize(obj, p0, method=method, bounds=bounds) # Fails without "good" bounds, but fast otherwise! 
        # sol = minimize(obj, p0, method='L-BFGS-B', bounds=bounds) # Worked fairly well, but slower and less optimal than SLSQP typically 
        # sol = minimize(obj, p0, method='TNC', bounds=bounds)  # Similar to LBFGS, but slower. 
        # sol = minimize(obj, p0, method='trust-constr', bounds=bounds)  # Didnt work well even with good bounds/guess 
        t1 = time.time()

        # Here we could/should check if the solution is on the boundary, indicating further improvement is possible with different bounds. 
        # if np.any(bounds[0] == sol.x[0]) or np.any(bounds[1] == sol.x[1]):
        #     print("Solution on boundary")

        if scalar:
            print("{} Optimization: \n\tBank* = {:.2f} deg\n\tVr*    = {:.2f} m/s\n\tFuel  = {:.1f} kg\n\tFound via {} calls over {:.1f} s".format(method, np.degrees(sol.x[0]), vr, sol.fun, sol.nfev, t1-t0))
            output = {'params': (sol.x[0],vr), 'fuel': sol.fun,} 
        else:
            if self.debug:
                print("{} Optimization: \n\t Bank* = {:.2f} deg\n\tVr* =   {:.2f} m/s\n\tFuel =  {:.1f} kg\n\tFound via {} calls over {:.1f} s".format(method, np.degrees(sol.x[0]), sol.x[1], sol.fun, sol.nfev, t1-t0))
            output = {'params': sol.x, 'fuel': sol.fun,} 

        return output

    def optimize(self, x0, verbose=True):
        """ Uses sequential univariate optimization to determine optimal parameters """
        from functools import partial 
        from scipy.optimize import minimize_scalar, minimize

        # Problem data 
        target = self.target
        srpdata = self.srpdata
        aero = self.aeroscale 


        # Optimization tolerances
        fuel_tol = 5 # kg 
        bank_tol = np.radians(0.25) # 0.5 degree tolerance 
        vr_tol = 10  # m/s
        max_iters = 3

        # Initialize our search parameters 
        if self.history['params']: #stuff is not None:
            bank_best, vr_best = self.history['params'][-1]
            fuel_best = 6000 # should we set this to the previous value? 

            # Optimization bounds
            bank_range = np.radians([-5, 5])
            bank_lims = bank_best + bank_range 
            vr_range = np.array([-150, 150])
            vr_lims = vr_best + vr_range

        else:
            bank_best = np.radians(0) # doesnt actually get used except in the first delta computation
            vr_best = 3100
            fuel_best = 0 
            # Optimization bounds
            bank_lims = np.radians([10, 65])
            vr_lims = [2000, 4000]

        # TODO: Handle failure scenarios where no good solution is found for the initial reversal (or bank)

        BANK_ANGLE_OPTIMIZATION_FAILED = False
        REVERSAL_ANGLE_OPTIMIZATION_FAILED = False

        T0 = time.time()
        for i in range(max_iters):
            if verbose:
                print("\nIteration {}".format(i+1))

            # fix the reversal, optimize bank 
            t0 = time.time()
            bank_fun = partial(_objective, x0=x0, srp_trim=lambda x: srpdata.srp_trim(x, target), reverse_velocity=vr_best, aero=aero, time_constant=self.time_constant)
            bank_sol = minimize_scalar(bank_fun, bounds=bank_lims, method='Bounded')
            # bank_sol = minimize(bank_fun, x0=bank_best, bounds=[bank_lims], method='SLSQP')
            t1 = time.time()

            if bank_sol.fun <= fuel_best:
                BANK_ANGLE_OPTIMIZATION_FAILED = False
                if verbose:
                    print("    Bank optimization: \n\tu* = {:.2f} deg\n\tFuel = {:.1f} kg\n\tFound via {} calls over {:.1f} s".format(np.degrees(bank_sol.x), bank_sol.fun, bank_sol.nfev, t1-t0))
                bank_delta = np.abs(bank_best - bank_sol.x)
                bank_best = bank_sol.x 
                fuel_delta = np.abs(fuel_best-bank_sol.fun)
                fuel_best = bank_sol.fun  
                bank_lims = bank_best + bank_range # Update bounds for the next iteration

                if i and fuel_delta < fuel_tol:
                    break
            else:
                print("Bank angle optimization with bounds [{:.2f} {:.2f}] deg could not improve the current solution".format(*np.degrees(bank_lims)))
                BANK_ANGLE_OPTIMIZATION_FAILED = True

            # fix the bank, optimize reversal 
            t0 = time.time()
            vr_fun = partial(_objective, bank=bank_best, x0=x0, srp_trim=lambda x: srpdata.srp_trim(x, target), aero=aero, time_constant=self.time_constant)
            vr_sol = minimize_scalar(lambda x: vr_fun(reverse_velocity=x), bounds=vr_lims, method='Bounded')
            t1 = time.time()
            if vr_sol.fun <= fuel_best:
                if verbose:
                    print("Reversal optimization: \n\tVr* = {:.2f} m/s\n\tFuel = {:.1f} kg\n\tFound via {} calls over {:.1f} s".format(vr_sol.x, vr_sol.fun, vr_sol.nfev, t1-t0))

                vr_delta = np.abs(vr_best - vr_sol.x)
                vr_best = vr_sol.x
                fuel_delta = np.abs(fuel_best-vr_sol.fun)
                fuel_best = vr_sol.fun
                vr_lims = vr_best + vr_range
            else:
                if verbose:
                    print("    Reversal optimization with bounds [{:.1f} {:.1f}] deg could not improve the current solution".format(*vr_lims))
                REVERSAL_ANGLE_OPTIMIZATION_FAILED = True

            if BANK_ANGLE_OPTIMIZATION_FAILED and REVERSAL_ANGLE_OPTIMIZATION_FAILED:
                print("Both optimization steps failed...")
                if fuel_best > 5000:
                    print("changing bounds and trying again") 
                    # Change the center point, the range, or both? If nominally we use a center point and small range, maybe try a big range 
                    bank_lims = np.radians([10, 65])
                    vr_lims = [2000, 4000]
                else:
                    print("...but solution is acceptable, optimization terminating.")
                    break 
            # Basically, only break if both optimizations succeeded and the solutions didnt change by much
            if not REVERSAL_ANGLE_OPTIMIZATION_FAILED and vr_delta < vr_tol and (fuel_delta < fuel_tol): # and not BANK_ANGLE_OPTIMIZATION_FAILED and bank_delta < bank_tol:
                break 

        TF = time.time()
        if verbose:
            print("\nTotal optimization time: {:.1f} s".format(TF-T0))
            print("Solution = {:.2f} deg, reversal at V = {:.2f} using {:.1f} kg of fuel".format(np.degrees(bank_best), vr_best, fuel_best))
        
        return {'params': (bank_best, vr_best), 'fuel': fuel_best,} 



def update_rule_maker(Vr):
    """ Utility to generate an update rule with any set of reversal velocities """
    def rule(history, state, **kwargs):
        r,th,ph,v,gamma,psi,m = state
        return np.any([(history['n_updates'] == i and v <= vr) for i, vr in enumerate(Vr)])
    return rule 

def update_rule(history, state, **kwargs):
    r,th,ph,v,gamma,psi,m = state
    
    Vr = [6000, 5400, 4500, 2500, 1500] # The velocities at which to update 
    # Put a high velocity to trigger an initial planning, but in general we should start with a nominal stored already, and not update till later 
    
    return np.any([(history['n_updates'] == i and v <= vr) for i, vr in enumerate(Vr)])

class SRPControllerTrigger(Trigger):
    """ Essentially a velocity trigger, but using the most recent predicted trigger condition """
    def __Trigger(self, velocity, **kwargs):
        if self.controller.history['velocity']:
            self.update_info("V <= {:.1f} m/s (predicted optimal ignition vel with {} offset)".format(self.controller.history['velocity'][-1]+self.offset, self.offset ))
            return velocity <= self.controller.history['velocity'][-1] + self.offset 
        return velocity <= 400 # If the controller has never been called and thus doesnt have a predicted terminal velocity, it's definitely False 

    def __init__(self, controller, offset=0):
        self.controller = controller
        self.offset = offset 
        super(SRPControllerTrigger, self).__init__(self.__Trigger, 'SRP Controller ignition point reached')

    
def test_single():
    x0 = InitialState(vehicle='heavy', fpa=np.radians(-15.75))    
    # use 700 for constant controller, 625 for two switch
    target = Target(0, 625/3397, 0)  # This is what all the studies have been done with so far 

    srpdata = pickle.load(open(SRPFILE,'rb'))

    mcc = SRPController(N=[30, 30], target=target, srpdata=srpdata, update_function=lambda *p, **d: True, debug=True, time_constant=2, constant_controller=False)
    # mcc = SRPController(N=[200, 200], target=target, srpdata=srpdata, update_function=lambda *p, **d: True, debug=True, time_constant=2)
    mcc(x0)
    # print(mcc.history['ignition_state'])
    # mcc.sim.plot()
    plt.show()


def test_sim(InputSample, EFPA, plot=False):
    """ Runs a scalar simulation with the SRP Controller called multiple times """
    x0 = InitialState(vehicle='heavy', fpa=np.radians(EFPA))    
    target = Target(0, 625/3397, 0)   
    TC = 2      # time constant 
    srpdata = pickle.load(open(SRPFILE,'rb'))

    # mcc = SRPController(N=[30, 30], target=target, srpdata=srpdata, update_function=update_rule_maker([5490, 5050, 4500, 3000, 2000, 1000]), debug=False, time_constant=TC, constant_controller=False)
    mcc = SRPController(N=[30, 30], target=target, srpdata=srpdata, update_function=update_rule_maker([5480]), debug=False, time_constant=TC, constant_controller=False)
    Vf = 450     # Anything lower than the optimal trigger point is fine 
    
    states = ['Entry']
    trigger = [SRPControllerTrigger(mcc, -10)] # Go 10 m/s lower than the true trigger point says to
    # trigger = [AltitudeTrigger(3)] # Go 10 m/s lower than the true trigger point says to
    sim_inputs = {'states': states, 'conditions': trigger}
    # sim = Simulation(cycle=Cycle(1), output=True, use_da=False, **EntrySim(Vf=Vf), )
    sim = Simulation(cycle=Cycle(0.1), output=False, use_da=False, **sim_inputs)
    sim.run(x0, [mcc], TimeConstant=TC, InputSample=InputSample, StepsPerCycle=10) 

    data = mcc.srp_trim(sim.history, full_return=True)
    mf = data['fuel']
    print("Ignition state:")
    for state in data['ignition_state']:
        print("{:.1f}".format(state))
    # mcc.plot_history()
    print("Final fuel consumed: {:.1f} kg".format(mf))
    if plot:
        sim.plot()    # The trajectory resulting from integrating the controller commands 
        # mcc.plot_history()
        # mcc.sim.plot() # The (last) trajectory predicted by the controller
        # v1 = sim.df['velocity']
        # v2 = mcc.sim.df['velocity']
        # for var in ['lift']:
        #     y1 = sim.df[var]
        #     y2 = mcc.sim.df[var]
        #     compare(v1, y1, v2, y2, N=None, plot=True)
        #     plt.suptitle(var.capitalize())


    # plt.show()
    return mf, sim.df, mcc.history


def parametric_sensitivity():
    """
        Test the guidance on Cd, Cl, rho0, and hs errors

        TODO: Save the trajectories and more for plotting 
        and investigation of optimal ignition points, stability of predicted solutions, etc 
    """
    totals = [] # [-/= fuel cases]
    m0 = []  # nominal fuel for each efpa 
    trajectories = []
    histories = []
    efpas = [-15.75]
    data = []
    for efpa in efpas:

        mnom, traj, hist = test_sim([0,0,0,0], efpa)
        m0.append(mnom)
        inps = np.eye(4) * 0.1 

        for sample in inps:
            print("\nSAMPLE {}".format(sample))
            neg, trajn, histn = test_sim(-sample, efpa)
            pos, trajp, histp = test_sim(sample, efpa)
            data.append([neg, pos])
            trajectories.append([trajn, trajp])
            histories.append([histn, histp])
        totals.append(data)
        data = np.array(data)
        delta = data - mnom 
        print("\nRelative to {:.1f} kg nominal: ".format(mnom))
        print(delta)


    output = {'efpa': efpas, "samples": inps, 'prop_nominal': m0, 'traj_nominal':traj, "history_nominal": hist, "prop": data, "traj": trajectories, "histories": histories}
    pickle.dump(output, open("./data/FuelOptimal/parametric_data_two_switch.pkl", 'wb'))

def plot_parametric(datafile):
    """ 
    """
    from Utils.smooth import smooth 

    # output = {'efpa': efpas, "samples": inps, 'prop_nominal': m0, 'traj_nominal':traj, "prop": data, "traj": trajectories, "histories": histories}
    data = pickle.load(open(datafile, 'rb'))

    def print_dicts(d, tabs=' '):
        if isinstance(d, dict):
            for key, val in d.items():
                print(tabs + key)
                try:
                    print(tabs + "{}".format(np.shape(val)))
                except:
                    print(tabs + "No shape - lists of lists/arrays with uneven sizes")
                try:
                    # print(type(val[0]))
                    print_dicts(val[0], tabs=tabs+'  --')
                    print_dicts(val[0][0], tabs=tabs+'  --')
                except:
                    pass

    print_dicts(data)
    data['prop'] = np.array(data['prop'])
    Vf = [data['traj_nominal']['velocity'].iloc[-1]+10] + [h['velocity'][-1] for h in np.array(data['histories']).flatten()]
    # print(Vf)
    prop = np.array([data['prop_nominal'][0], *data['prop'].flatten()])
    # print(prop)
    delta = prop - prop[0]
    per = delta/prop[0]*100
    ignition = [np.zeros((5,))] + [h['ignition_state'][-1] for h in np.array(data['histories']).flatten()]

    try:
        histories = [data['history_nominal']] + [h for h in np.array(data['histories']).flatten()]
    except:
        histories = [0] + [h for h in np.array(data['histories']).flatten()] # no nominal history...

    # create a table 
    states = ['DR','CR','Alt','Vx', 'Vz']
    table_data = {'pmf': prop/7200*100, 'delta pmf': delta/7200*100, 'ignition velocity': Vf}
    for state, values in zip(states, np.array(ignition).T):
        table_data[state] = values
    df = pd.DataFrame(table_data)
    df.to_csv("./data/FuelOptimal/parametric_sensitivity_table.csv")

    savedir = "./Documents/FuelOptimal/1d parametric/"

    # Plots GALORE
    figsize=(7,5)
    fontsize=14
    ticksize=12 

    # plt.figure(figsize=figsize)
    # plt.scatter(vf, prop)
    # plt.xlabel('Ignition Velocity', fontsize=fontsize)
    # plt.ylabel('Propellant (kg)', fontsize=fontsize)

    traj = [data['traj_nominal']]
    for d in data['traj']:
        traj.extend(d)
    # print(traj)
    labels = ['nominal']
    labels += [a+b for b in [r'$C_D$',r'$C_L$',r'$\rho$', r'$h_s$'] for a in ['- 10% ','+10% ']]
    # print(labels)

    # plt.figure(5, figsize=figsize)
    # plt.scatter(table_data['DR'][1:], table_data['Alt'][1:], s=200)
    # plt.hlines(3000, 0, 16000, 'r', label="Minimum altitude")
    # plt.quiver(table_data['DR'], table_data['Alt'], table_data['Vx'], table_data['Vz'], scale=5e3)
    # plt.axis('equal')
    # plt.tick_params(labelsize=ticksize)
    # plt.xlabel('Downrange to Target (m)', fontsize=fontsize)
    # plt.ylabel('Altitude (m)', fontsize=fontsize)
    # plt.grid()
    # plt.legend()
    # skip = [r'$C_D$',r'$C_L$',r'$h_s$']
    keeplist = [r'$C_D$',r'$C_L$',r'$\rho$', r'$h_s$', [r'$C_D$',r'$C_L$',r'$\rho$', r'$h_s$']]
    # keeplist = [[r'$C_D$',r'$C_L$',r'$\rho$', r'$h_s$']]
    names = ['drag','lift','rho','hs','all']
    figs = ['h_v','dr_cr','bank','lod','h_v_zoomed','prop_hist']
    fignum_mult = 0 
    nplots = len(figs)

    for keep in keeplist:
        if not isinstance(keep, list):
            keep = [keep]
        keep = ['nominal'] + keep
        fignum_mult += 1
        for tr, vf, hist, label in zip(traj, Vf, histories, labels):

            if not np.any([s in label for s in keep]):
                # print(label)
                continue

            v = tr['velocity'].values 
            tr = tr[v>=vf]
            v = tr['velocity'].values 

            plt.figure(nplots*(fignum_mult-1) + 1, figsize=figsize)
            plt.plot(tr['velocity'], tr['altitude'], label=label)
            plt.xlabel('Velocity', fontsize=fontsize)
            plt.ylabel('Altitude (km)', fontsize=fontsize)
            plt.tick_params(labelsize=ticksize)
            plt.grid()
            plt.legend()

            plt.figure(nplots*(fignum_mult-1) + 2, figsize=figsize)
            plt.plot(tr['crossrange'], tr['downrange'], label=label)
            plt.ylabel('Downrange (km)', fontsize=fontsize)
            plt.xlabel('Crossrange (km)', fontsize=fontsize)
            # plt.axis('equal')
            plt.tick_params(labelsize=ticksize)
            plt.grid()
            plt.legend()

            t = tr['time'].values
            k = np.diff(t) == 0
            t = t[:-1][k]

            bank_smooth = smooth(t, tr['bank'].values[:-1][k], N=1, tau=1.01)(tr['time'])
            bank = tr['bank'].values

            beta = 0.5
            # bank_smooth = beta * bank[1:] + (1-beta)*bank[:-1]
            bank_smooth = beta * bank + (1-beta)*bank_smooth
            plt.figure(nplots*(fignum_mult-1) + 3, figsize=figsize)
            # plt.plot(tr['velocity'], bank_smooth, label=label)
            plt.plot(tr['velocity'], tr['bank'], label=label)
            plt.tick_params(labelsize=ticksize)
            plt.xlabel('Velocity', fontsize=fontsize)
            plt.ylabel('Bank Angle (deg)', fontsize=fontsize)
            plt.grid()
            plt.legend()

            lodv = tr['lift']*np.cos(np.radians(tr['bank']))/tr['drag']
            plt.figure(nplots*(fignum_mult-1) + 4, figsize=figsize)
            plt.plot(tr['velocity'], lodv, label=label)
            # plt.plot(tr['velocity'], tr['bank'], label=label)
            plt.tick_params(labelsize=ticksize)
            plt.xlabel('Velocity', fontsize=fontsize)
            plt.ylabel('Vertical L/D ', fontsize=fontsize)
            plt.grid()
            plt.legend()

            plt.figure(nplots*(fignum_mult-1) + 5, figsize=figsize)
            plt.plot(tr['velocity'][v<=1200], tr['altitude'][v<1200], label=label)
            plt.xlabel('Velocity', fontsize=fontsize)
            plt.ylabel('Altitude (km)', fontsize=fontsize)
            plt.tick_params(labelsize=ticksize)
            plt.grid()
            plt.legend()

            if 'nominal' in label: # no nominal history saved 
                continue
            plt.figure(nplots*(fignum_mult-1) + 6, figsize=figsize)
            updates = list(range(1,1+len(hist['fuel'])))
            plt.plot(updates, np.array(hist['fuel'])/7200, "o--", label=label)
            plt.xlabel('Parameter Update', fontsize=fontsize)
            plt.ylabel('PMF (%)', fontsize=fontsize)
            plt.xticks([0] + updates)
            plt.tick_params(labelsize=ticksize)
            plt.grid()
            plt.legend()

        for i in range(nplots):
            plt.figure(nplots*(fignum_mult-1) + i+1)
            plt.savefig(os.path.join(savedir, "{}_{}".format(figs[i], names[fignum_mult-1])))



    # plt.show()

def test_sweep():
    """ Sweeps over a variety of EFPA/EAZI angles to generate a table of initial params."""

    target = Target(0, 700/3397, 0)  
    srpdata = pickle.load(open(SRPFILE,'rb'))

    ic = boxgrid([(-16.3, -15.3), (-0.5, 0.5)], [11, 9], interior=True)
    df = pd.read_csv("./data/FuelOptimal/srp_params - Copy.csv")
    data = df.values.T[1:]
    ic_existing = data[3:5].T
    # ic = [[-16.5, 0.5]]

    data = []
    for efpa, azi in ic:
        skip = False 
        print("\n Entry FPA/AZI:")
        print(efpa, azi)
        # for pair in ic_existing:
        #     if np.abs(efpa - pair[0]) < 0.05 and azi==pair[1]:
        #         print("Already run a sufficiently close sample, skipping")
        #         skip = True 
        # if skip:
        #     continue 
        x0 = InitialState(vehicle='heavy', fpa=np.radians(efpa), psi=np.radians(azi)) 
        for boolean in [1,]:
            mcc = SRPController(N=[30, 100], target=target, srpdata=srpdata, update_function=update_rule, debug=False, constant_controller=boolean)
            # mcc = SRPController(N=[3, 5], target=target, srpdata=srpdata, update_function=update_rule, debug=False, constant_controller=boolean)
            try:
                mcc(x0)
                data.append([mcc.history['fuel'][-1], *mcc.history['params'][-1], efpa, azi, *mcc.history['ignition_state'][-1], *mcc.history['entry_state'][-1]])
            except IndexError:  # Occurs when the optimization ends without finding a feasible solution 
                print("Could not find viable solution ")
                pass

    data = np.array(data).T
    data[1] = np.degrees(data[1])

    df = pd.DataFrame(data.T, columns=["fuel", "p1"," p2", "efpa", "eazi", 'x','y','z','vx','vz', 'r', 'lon','lat', 'v', 'fpa','azi','m'])
    
    # try-except because one time my dumb@$$ had temp.csv open and I lost 2+ hours worth of computations 
    try:
        df.to_csv("./data/FuelOptimal/temp.csv")
    except:
        df.to_csv("./data/FuelOptimal/temp128371287391.csv")

    plt.show()

def plot_sweep(datafile):
    import matplotlib as mpl
    mpl.rc('image', cmap='inferno')
    df = pd.read_csv(datafile)
    
    data = df.values.T[1:]
    ic = data[3:].T
    print(data.shape)

    pmf = data[0]/df['m'] * 100


    plt.figure()
    plt.tricontourf(ic.T[0], ic.T[1], pmf)
    plt.xlabel("EFPA (deg)")
    plt.ylabel("Entry Azimuth Error (deg)")
    plt.colorbar(label="Propellant Mass Fraction (%)")

    plt.figure()
    plt.tricontourf(ic.T[0], ic.T[1], data[1])
    plt.xlabel("EFPA (deg)")
    plt.ylabel("Entry Azimuth Error (deg)")
    plt.colorbar(label="Bank Angle (deg)")

    plt.figure()
    plt.scatter(ic.T[0], data[1])
    plt.xlabel("EFPA (deg)")
    plt.ylabel("Bank Angle (deg)")

    plt.figure()
    plt.tricontourf(ic.T[0], ic.T[1], data[2])
    plt.xlabel("EFPA (deg)")
    plt.ylabel("Entry Azimuth Error (deg)")
    plt.colorbar(label="Reversal Velocity (m/s)")

    # Auto histograms 
    # for var in ['x','y','z','vx','vz']:
    #     plt.figure()
    #     plt.tricontourf(ic.T[0], ic.T[1], df[var])
    #     plt.xlabel("EFPA (deg)")
    #     plt.ylabel("Entry Azimuth Error (deg)")
    #     plt.colorbar(label='Ignition State {}'.format(var.capitalize()))

        # plt.figure()
        # plt.hist(df[var])
        # plt.xlabel(var.capitalize())

    
    plt.figure()
    plt.scatter(df['vx'], df['x'], c=pmf)
    plt.xlabel('Downrange Velocity')
    plt.ylabel('Downrange distance')
    plt.colorbar(label="Propellant Mass Fraction (%)")

    plt.figure()
    plt.scatter(df['vz'], df['z'], c=pmf)
    plt.xlabel('Vertical Velocity')
    plt.ylabel('altitude')
    plt.colorbar(label="Propellant Mass Fraction (%)")
    
    plt.show()


def dataframe_merge(file_to_append, file_to_add):
    # performs a special merge: appends data from second file to first file
    # but checks to see if two solutions exist for the same initial conditions, 
    # and takes the better of the two based on propellant cost 
    df1 = pd.read_csv(file_to_append)[1:]
    df2 = pd.read_csv(file_to_add)[1:]
    df = pd.concat([df1,df2])
    df.to_csv("./data/FuelOptimal/merged.csv", index=False)
    # first, create one dataframe
    remove_duplicates("./data/FuelOptimal/merged.csv")
    # 

def remove_duplicates(datafile):
    # checks for repeated datapoints and uses the one with lower propellant cost 
    outfile = datafile.replace('.csv','_duplicates_removed.csv')
    print(outfile)

    df = pd.read_csv(datafile)
    efpa = df['efpa']
    eazi = df['eazi']
    prop = df['fuel']

    pairs = np.array([pair for pair in zip(efpa,eazi)])
    unique = prop > 0 # start with all 1s
    append = []
    dup_list = []

    for pair in pairs:
        # print(pair)
        dup = np.all(pairs == pair, axis=1)
        try:
            repeat = np.any(np.all(pair == np.array(dup_list), axis=1))
        except:
            repeat = False 

        if np.sum(dup) > 1 and not repeat:
            dup_list.append(pair)
            unique = np.logical_and(unique, np.invert(dup))
            print("Duplicate(s) found = {}".format(pairs[dup][0]))
            i = np.argmin(prop[dup].values)
            append.append(df[dup].iloc[i])

    append = pd.DataFrame(append)
    df_out = pd.concat([df[unique], append])
    df_out.to_csv(outfile, index=False)


if __name__ == "__main__":
    
    # test_single()
    # test_sweep()
    # dataframe_merge("./data/fuelOptimal/srp_params - Copy.csv","./data/fuelOptimal/data_to_merge.csv")
    # remove_duplicates("./data/fuelOptimal/test.csv")
    # plot_sweep("./data/FuelOptimal/merged_duplicates_removed.csv")
    # plot_sweep("./data/FuelOptimal/data_to_merge.csv")
    # test_sim([0., 0., 0., 0.01], -15.75, True)
    # test_sim([-0.1, 0., 0., 0.0], -15.75, True)
    parametric_sensitivity()
    plot_parametric("./data/FuelOptimal/parametric_data_two_switch.pkl")

    
    # v = np.linspace(600, 5000, 1000)

    # b1 = reversal_controller(np.radians(30), 3500, False)
    # b2 =  switch_controller([4000, 2500], False)

    # B1 = np.degrees([b1(vi) for vi in v])
    # B2 = np.degrees([b2(vi) for vi in v])

    # fontsize=14
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1,2,1)
    # plt.plot(v, B1, label=r"($\sigma_c$, $v_r$)",linewidth=4)
    # plt.gca().set_ylim([-95, 95])
    # plt.xlabel('Velocity (m/s)', fontsize=fontsize)
    # plt.ylabel('Bank angle (deg)', fontsize=fontsize)
    # plt.grid(True)
    # plt.legend()

    # plt.subplot(1,2,2)
    # plt.plot(v, B2, label=r"($v_1$, $v_2$)",linewidth=4)
    # plt.gca().set_ylim([-95, 95])
    # plt.xlabel('Velocity (m/s)', fontsize=fontsize)
    # plt.ylabel('Bank angle (deg)', fontsize=fontsize)
    # plt.grid(True)
    # plt.legend()
    # plt.show()