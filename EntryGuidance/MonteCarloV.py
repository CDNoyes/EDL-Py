""" Defines utilities for conducting vectorized Monte Carlo simulations """
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.io import savemat, loadmat

from EntryEquations import EDL
from InitialState import InitialState
from Uncertainty import getUncertainty
from Convex_Entry import LTV
from Utils.RK4 import RK4
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

    def reference_data(self):
        """ In closed loop simulations, generate reference data once and use it
            each simulation
        """
        print "Generating reference data for closed-loop guidance..."
        from Simulation import Simulation, Cycle, EntrySim
        from InitialState import InitialState
        from ParametrizedPlanner import profile
        from Utils.submatrix import submatrix
        from Mesh import Mesh
        from scipy.interpolate import interp1d

        x0 = InitialState()
        switch = [    62.30687581,  116.77385384,  165.94954234]
        bank = [-np.radians(30),np.radians(75),-np.radians(75),np.radians(30)]
        bankProfile = lambda **d: profile(d['time'],switch=switch, bank=bank,order=2)
        # bankProfile = lambda **d: 0.

        sim =  Simulation(cycle=Cycle(1),output=False, **EntrySim(Vf=470))
        res = sim.run(x0,[bankProfile])
        self.ref_sim = sim
        s0 = sim.history[0,6]-sim.history[-1,6] # This ensures the range to go is 0 at the target for the real simulation
        # sim.plot(compare=False)
        # plt.show()
        # self.x0_nom = InitialState(1, range=s0, bank=bank[0])
        print "...done. "
        print sim.history.shape

        print("Generating linearization about reference data...")

        sigma_dot = np.diff(sim.control_history[:,0])/np.diff(sim.times)
        sigma_dot = np.append(sigma_dot, 0) # To make it the same length
        # plt.plot(sim.times,sigma_dot)
        # plt.show()

        # Linearizations for energy or time as the independent variable
        energy = True
        rows = list(range(6))
        cols = rows + [8]
        sim.edlModel.use_energy = energy
        sim.edlModel.DA(True)
        F = np.array([sim.edlModel.dynamics(control)(state, 0)[0:7] for state,control in zip(sim.history, sim.control_history)])
        J = np.array([submatrix(sim.edlModel.jacobian_(state, control), rows, cols) for state,control in zip(sim.history, sim.control_history)])
        J2 = np.array([sim.edlModel.bank_jacobian(state, control, sdot) for state,control,sdot in zip(sim.history, sim.control_history, sigma_dot)])
        B = np.zeros((sim.history.shape[0],7,1))
        B[:,-1,0] = J2[:,0,-1]
        # E_dot = 1/(sim.df['drag']*sim.df['velocity'])
        # plt.plot(sim.history[:,3], (B[:,-1,0]))
        # plt.plot(sim.history[:,3], E_dot, 'k--')
        # plt.show()

        J2 = J2[:,:,cols] # Get rid of the extra terms we dont care about
        A = np.concatenate((J,J2), axis=1)
        F[:,6] = sigma_dot*B[:,-1,0]# Add the bank angle dynamics

        # Define a hypercube trust region around the reference traj
        TR = np.array([5000, np.radians(0.2), np.radians(0.2), 30, np.radians(2), np.radians(2), np.radians(45)])*1

        # Define a new target at exactly 0 crossrange:
        xf = sim.history[-1,:7].copy()
        # xf[2] = xf[2]/2.
        xf[6] = sim.control_history[-1,0]
        x_ref = np.concatenate((sim.history[:,:6], sim.control_history[:,:1]), axis=1)
        # Prepare a mesh
        if energy:
            mesh = Mesh(t0=sim.df['energy'].values[112], tf=sim.df['energy'].values[-1], orders=[4]*40)
            IV = sim.df['energy'].values
            Ei = mesh.times
        else:
            mesh = Mesh(t0=75, tf=sim.times[-1], orders=[4]*25)
            IV = sim.times
            Ei = mesh.times

        # Need to interpolate all of the inputs onto the mesh points
        A = interp1d(IV, A, axis=0)(Ei)
        B = interp1d(IV, B, axis=0)(Ei)
        F = interp1d(IV, F, axis=0)(Ei)
        x_ref = interp1d(IV, x_ref, axis=0)(Ei)
        u_ref = interp1d(IV, sigma_dot, axis=0)(Ei)
        print "...done."
        print("Solving convex optimization problem...")
        # for M in [A,B,F,x_ref,u_ref]:
        #     print(np.shape(M))
        X, U, sol = LTV(x_ref[0], A, B, F, x_ref, u_ref=u_ref, mesh=mesh, trust_region=TR, xf=xf, umax=np.radians(20))
        print "...done."

        return X, U, x_ref, u_ref



    def sample(self, N, sample_type='S', parametric=True, initial=False, knowledge=False):
        """ Generates samples for use in Monte Carlo simulations """
        from Uncertainty import getUncertainty
        uncertainty = getUncertainty(parametric=parametric, initial=initial, knowledge=knowledge)

        self.samples = uncertainty['parametric'].sample(N-1, sample_type)
        self.psamples = uncertainty['parametric'].pdf(self.samples)
        print(" ")
        print("Generating {} samples...".format(N))

        str2name = {'S':'Sobol','R':'Random','L':'Latin hypercube'}

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
            filename = getFileName('MC_{}'.format(len(self.samples.T)),saveDir)
            fullfilename = saveDir + filename
            print("Saving {} to default location {}".format(filename,saveDir))
        elif save is False:
            pass
        else:
            print("Saving data to user location {}".format(save))
            fullfilename = save

        savemat(fullfilename, {'xf':self.xf, 'states':self.mc, 'samples':self.samples, 'pdf':self.psamples})


    def _run(self, x):
        edl = EDL(self.samples, Energy=True)
        self.model = edl
        optSize = self.samples.shape[1]
        if len(x.shape) == 1: # Allows a single initial condition or an array
            x = np.tile(x,(optSize,1)).T
        X = [x]
        energy0 = edl.energy(x[0],x[3],False)[0]
        energyf = edl.energy(edl.planet.radius, 250,False) # go down to low energy then parse afterward

        energy = energy0
        E = [energy]
        while energy > energyf:

            Xc = X[-1]
            energys = edl.energy(Xc[0],Xc[3],False)
            lift,drag = edl.aeroforces(Xc[0],Xc[3],Xc[7])

            # Range control
            u = self.controls[0](energys)
            # Shape the control
            u.shape = (1,optSize)
            u = np.vstack((u,np.zeros((2,optSize))))
            de = -np.mean(drag)*np.mean(Xc[3])
            if (energy + de) <  energyf:
                de = energyf - energy
            eom = edl.dynamics(u)
            X.append(RK4(eom, X[-1], np.linspace(energy,energy+de,10),())[-1])
            energy += de
            E.append(energy)

            if len(E)>600:
                break
        X = np.array(X)
        print X.shape
        self.mc = X
        self.trigger()


    def trigger(self):
        xf = [self._trigger(traj) for traj in np.transpose(self.mc,(2,0,1))]
        print np.shape(xf)
        self.xf = np.array(xf)
        # self.mc

    def _trigger(self, traj):
        for state in traj:
            if state[3] <= 470: # Velocity trigger
                return state
        return traj[-1] # Never reached

    def plot(self):
        plt.figure(1)
        Parachute.Draw(figure=1)
        Parachute.Draw(figure=3)
        for X in np.transpose(self.mc,(2,1,0)): # each is n_samples x n_points now
            r,lon,lat,v,fpa,psi,s,m = X
            dr = self.model.planet.radius*lon/1000
            cr = -self.model.planet.radius*lat/1000

            plt.figure(1)
            plt.plot(v, (r-self.model.planet.radius)/1000.)

            plt.figure(2)
            plt.plot(cr,dr)

        r,lon,lat,v,fpa,psi,s,m = self.xf.T
        dr = self.model.planet.radius*lon/1000
        cr = -self.model.planet.radius*lat/1000
        plt.figure(3)
        plt.plot(v,(r-self.model.planet.radius)/1000.,'o')

        plt.figure(4)
        plt.plot(cr,dr,'o')
        plt.show()


def solve_ocp(dr=885., fpa_min=-45, azi_max=0.):
    from gpops import gpops
    from scipy.interpolate import interp1d
    from math import pi

    def rad(num):
        return float(num)*pi/180.
    t0 = time.time()
    traj = gpops([float(dr), rad(fpa_min), rad(azi_max)])
    print("Total OCP solution time {} s".format(time.time()-t0))
    sigma = np.squeeze(np.array(traj['state'])[:,-1])

    bankProfile = interp1d(np.squeeze(np.squeeze(traj['energy'])), sigma, fill_value=(sigma[0],sigma[-1]), assume_sorted=False, bounds_error=False, kind='cubic')
    return traj, bankProfile

def getFileName(name, save_dir):
    """
        Looks in 'dir' for files with the pattern name-date-number

        I.e. if name = 'MC' and dir/MC-date-1 exists, MC-date-2 will be returned.
    """
    date = datetime.now().date()

    files =  os.listdir(save_dir)
    current = 1
    fname = "{}_{}_{}.mat".format(name,date,current)
    while fname in files:
        current += 1
        fname = "{}_{}_{}.mat".format(name,date,current)

    return fname

def load(mat_file):
    data = loadmat(mat_file)
    self.xf = data['xf']
    self.mc = data['states']
    self.samples = data['samples']
    self.psamples = data['pdf']


if __name__ == "__main__":
    import sys
    sys.path.append(os.getcwd() + '\Utils')
    from itertools import product

    mc = MonteCarlo()
    mc.reference_data()
    # mc.sample(1000)

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
