""" Defines a class for conducting vectorized Monte Carlo simulations """

import os
import sys
sys.path.append("./Utils")  # path appending is relative to where the file is run 

import time
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.io import savemat, loadmat

from EntryEquations import EDL
from InitialState import InitialState
# from .Uncertainty import getUncertainty
from RK4 import RK4
import Parachute


class MonteCarlo(object):
    """ Monte carlo class """

    def __init__(self):

        self.samples    = None
        self.controls   = None
        self.mc         = None

    def set_controls(self, controls):
        self.controls = controls
        return

    def reference_data(self, ref_profile, Vf=470, plot=False):
        """ In closed loop simulations, generate reference data once and use it
            each simulation
        """
        print("Generating reference data...")
        from Simulation import Simulation, Cycle, EntrySim
        from InitialState import InitialState
        # from ParametrizedPlanner import profile
        # from Utils.submatrix import submatrix
        from scipy.interpolate import interp1d

        x0 = InitialState()
        sim = Simulation(cycle=Cycle(0.2), output=False, **EntrySim(Vf=Vf))
        res = sim.run(x0, [ref_profile])
        self.ref_sim = sim
        # s0 = sim.history[0,6]-sim.history[-1,6] # This ensures the range to go is 0 at the target for the real simulation
        if plot:
            sim.plot(compare=False)
            plt.show()
        # self.x0_nom = InitialState(1, range=s0, bank=bank[0])
        print("...done. ")
        print(sim.history.shape)

    def sample(self, N, sample_type='S', parametric=True, initial=False, knowledge=False):
        """ Generates samples for use in Monte Carlo simulations """
        from Uncertainty import getUncertainty
        uncertainty = getUncertainty(parametric=parametric, initial=initial, knowledge=knowledge)

        self.samples = uncertainty['parametric'].sample(N, sample_type)
        self.psamples = uncertainty['parametric'].pdf(self.samples)
        print(" ")
        print("Generating {} samples...".format(N))

        str2name = {'S': 'Sobol', 'R': 'Random', 'L': 'Latin hypercube'}

        print("     {} sampling method".format(str2name[sample_type]))
        print("     Parametric uncertainty only")

    def run(self, x0=None, save=None):

        if self.controls is None:
            print(" ")
            print("Warning: The controls must be set prior to running.")
            print("Exiting.")
            return

        if self.samples is None:
            print(" ")
            print("Warning: The number and type of sampling must be set prior to running.")
            print("Exiting.")
            return

        if x0 is None:
            x0 = InitialState()

        print(" ")
        print("Running Monte Carlo...")
        t_start = time.time()
        self._run(x0)
        print("Monte Carlo complete, time elapsed = {} s.".format(time.time()-t_start))

        if save is None:
            saveDir = './data/'
            filename = getFileName('MC_{}'.format(len(self.samples.T)), saveDir)
            fullfilename = saveDir + filename
            print("Saving {} to default location {}".format(filename, saveDir))
        elif save is False:
            return
        else:
            print("Saving data to user location {}".format(save))
            fullfilename = save

        savemat(fullfilename, {'xf': self.xf, 'states': self.mc, 'samples': self.samples, 'pdf': self.psamples})

    def _run(self, x, stepsize=0.5):
        edl = EDL(self.samples, Energy=True)
        self.model = edl
        optSize = self.samples.shape[1]
        if x.ndim == 1:  # Allows a single initial condition or an array
            x = np.tile(x, (optSize, 1)).T
        X = [x]
        energy = np.mean(edl.energy(x[0], x[3], False))
        print("E0 {}".format(energy))
        energyf = edl.energy(edl.planet.radius+1000, 350, False)  # go down to low energy then parse afterward
        print("Ef {}".format(energyf))

        E = [energy]
        while True:
            # print("E: {:.1f}".format(energy))
            Xc = X[-1]
            energys = edl.energy(Xc[0], Xc[3], False)
            lift, drag = edl.aeroforces(Xc[0], Xc[3], Xc[7])

            # Range control
            u = self.controls[0](energys)
            # Shape the control
            u.shape = (1, optSize)
            u = np.vstack((u, np.zeros((2, optSize))))
            de = -np.mean(drag)*np.mean(Xc[3]) * stepsize
            # if (energy + de) < energyf:
            #     de = energyf - energy
            eom = edl.dynamics(u)
            X.append(RK4(eom, X[-1], np.linspace(energy, energy+de, 10))[-1])
            energy += de
            E.append(energy)

            if np.isnan(energy):
                print("Something wrong")
                break

            if energy <= energyf:
                print("energy depleted ")
                break 

            if len(E) > 6000:
                print("Breaking ")
                break
        X = np.array(X)
        print("MC shape")
        print(X.shape)
        self.mc_full = X
        self.trigger()  # This should be moved to a method that the user controls after running the MC
        # This will provide flexibility to use different triggers 

    def set_trigger(self, trigger_function):
        self._trigger = trigger_function 

    def trigger(self):
        xfi = [self._trigger(traj) for traj in np.transpose(self.mc_full, (2,0,1))]
        xf = [traj[i] for i, traj in zip(xfi, np.transpose(self.mc_full, (2,0,1)))], 
        self.xf = np.array(xf).squeeze()
        print("Terminal state shape: ")
        print(np.shape(self.xf))
        self.mc = [traj[:i] for i, traj in zip(xfi, np.transpose(self.mc_full, (2,0,1)))]

    def plot(self, figsize=(10, 6), fontsize=16):
        try:
            self.mc 
            self.xf 
        except AttributeError:
            if hasattr(self, 'mc_full'):
                print("MonteCarlo has been run but trigger has not been called to determine final states.")
            else:
                print("MonteCarlo must be run before plotting.")
            return 

        plt.figure(1, figsize=figsize)
        Parachute.Draw(figure=1)
        Parachute.Draw(figure=3)
        for X in self.mc: # each is n_samples x n_points now
            r,lon,lat,v,fpa,psi,s,m = X.T
            dr = self.model.planet.radius*lon/1000
            cr = -self.model.planet.radius*lat/1000
            h = self.model.altitude(r, km=True)
            plt.figure(1)
            plt.plot(v, h, 'o')

            plt.figure(2, figsize=figsize)
            plt.plot(cr, dr, 'o')

        r,lon,lat,v,fpa,psi,s,m = self.xf.T
        dr = self.model.planet.radius*lon/1000
        cr = -self.model.planet.radius*lat/1000
        plt.figure(3)
        plt.plot(v, (r-self.model.planet.radius)/1000,'o')

        plt.figure(4)
        plt.plot(cr, dr, 'o')

        try:
            xf = self.ref_sim.history[-1]
            hor_err = np.sqrt((lon - xf[1])**2 + xf[2]**2)*3397
            
            plt.figure(5, figsize=figsize)
            plt.hist(hor_err, cumulative=True, histtype='step', bins='auto', linewidth=4, density=True)
            plt.xlabel("Horizontal Error (km)")
        except AttributeError:  # No ref sim, perhaps because loaded from file, or because no reference is needed 
            pass 

        # plt.figure(6, figsize=figsize)
        # plt.hist((r-self.model.planet.radius)/1000., cumulative=True, histtype='step', bins='auto', linewidth=4, density=True)
        # plt.xlabel("Final Altitude (km)")

    def load(self, mat_file):
        data = loadmat(mat_file)
        print(data['states'].shape)
        try:
            self.xf = data['xf']
        except KeyError:
            self.xf = data['states'][-1]

        self.mc = np.transpose(data['states'], (0,2,1))
        self.samples = data['samples']
        self.psamples = data['pdf']
        edl = EDL(self.samples, Energy=True)
        self.model = edl


def solve_ocp(dr=885., fpa_min=-45, azi_max=5.):
    from Utils.gpops import entry
    from scipy.interpolate import interp1d
    from math import pi

    def rad(num):
        return float(num)*pi/180.
    t0 = time.time()
    traj = entry([float(dr), rad(fpa_min), rad(azi_max)])
    print("Total OCP solution time {} s".format(time.time()-t0))
    sigma = np.squeeze(np.array(traj['state'])[:,-1])

    bankProfile = interp1d(np.squeeze(np.squeeze(traj['energy'])), sigma, fill_value=(sigma[0],sigma[-1]), assume_sorted=False, bounds_error=False, kind='cubic')
    bankP = lambda **d: bankProfile(d['energy'])
    return traj, bankP


def getFileName(name, save_dir):
    """
        Looks in 'save_dir' for files with the pattern name-date-number

        I.e. if name = 'MC' and dir/MC-date-1 exists, MC-date-2 will be returned.
    """
    date = datetime.now().date()

    files = os.listdir(save_dir)
    current = 1
    fname = "{}_{}_{}.mat".format(name, date, current)
    while fname in files:
        current += 1
        fname = "{}_{}_{}.mat".format(name, date, current)

    return fname

def velocity_trigger(traj, Vf=470):
    for idx, state in enumerate(traj):
        if state[3] <= Vf:  
            return idx
        # if self.model.energy(state[0], state[3], False) <= Ef: # Energy trigger
        #     return idx
    return -1

def e_controller(e):
    def controller(energy):
        u = np.ones_like(energy)
        u[energy < e] *= -1 
        return u 
    return controller 

if __name__ == "__main__":
    from itertools import product
    import Apollo 
    
    mc = MonteCarlo()
    # mc.reference_data(lambda **d: 0.6, plot=False)
    mc.sample(250)
    # controls = [lambda x: 1.0*np.ones_like(x)]
    controls = [e_controller(4850e3)]
    mc.set_controls(controls)
    mc.set_trigger(velocity_trigger)
    mc.run(save=False)
    # mc.load("./data/MC_1000_2018-03-29_1.mat")
    mc.plot()
    plt.show()

    # headings = [5., 1.35, 1., 0.5, 0.1, 0.]
    # fpas = [-25., -17., -16., -15., -14]

    # for inputs in product(fpas,headings):
    #     print inputs
    #     traj, bank = solve_ocp(885, *inputs)
    #     controls = [bank]
    #     mc.set_controls(controls)
    #     mc.run()
        # mc.plot()
        # 1/0
