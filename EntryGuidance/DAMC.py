import unittest

from Simulation import Simulation, Cycle, EntrySim, TimedSim
from Triggers import SRPTrigger, AccelerationTrigger, VelocityTrigger, EnergyTrigger
from InitialState import InitialState
from Uncertainty import getUncertainty
from ParametrizedPlanner import profile
from NMPC import NMPC

from pyaudi import gdual_double as gd
from pyaudi import abs
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from scipy.interpolate import interp1d

import time

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from Utils import DA as da
from Utils import draw
from Utils.submatrix import submatrix

class DAMC(object):
    """ Differential-Algebraic Monte Carlo

        Approximation to Monte Carlo sampling via expansion of the nominal
        trajectory. Expansion order is allowed to vary. Future work includes
        an adaptive version.

    """
    def __init__(self,order=None):

        if order is None:
            print("Using default expansion order 1.")
            self.order = 1
        else:
            self.order=order

    def reference(self):
        # ######################################################
        # Reference data generation
        # ######################################################
        reference_sim = Simulation(cycle=Cycle(1),output=True,**EntrySim())
        if 1:
            switch = [    62.30687581,  116.77385384,  165.94954234]
            banks = [-np.radians(30),np.radians(75),-np.radians(75),np.radians(30)]
        else: # No reversals
            switch = [ 20.09737598,  139.4887652 ]
            banks = [np.radians(30),np.radians(75),np.radians(30)]

        bankProfile = lambda **d: profile(d['time'],switch=switch, bank=banks,order=2)


        x0 = InitialState()
        output_ref = reference_sim.run(x0,[bankProfile],StepsPerCycle=10)
        self.ref_sim = reference_sim
        self.ref_traj=output_ref
        self.ref_xf = output_ref[-1]

        print("Reference data generated.")
        return self


    def integrate(self, dt=2.5, Q=np.array([[.1,0],[0,2]])):
        """ Performs forward integration of the expansion points """
        refs = self.ref_sim.getRef()
        fbl_ref = self.ref_sim.getFBL()
        dragprof = refs['drag']
        altprof = refs['altitude']
        Ef = self.ref_sim.df['energy'].values[-1]

        # Setup DA simulation
        x0 = InitialState(range=self.ref_sim.df['downrange'].values[-1]*1e3)

        states = ['Entry']
        trigger = [EnergyTrigger(Ef)]

        da_sim = Simulation(cycle=Cycle(1), output=False, use_da=True, **{'states':states, 'conditions':trigger})

        xvars = ['r','lon','lat','v','fpa','psi','s','m']
        orders = [self.order*0]*6 + [0]*2
        x0d = [gd(val,name,order) for val,name,order in zip(x0,xvars,orders)]
        params =  ['CD','CL','rho0','sh']
        sample = [gd(0,name,self.order) for name in params]     # Expand around the nominal values
        xvars += params
        self.vars = xvars
        # Setup a controller
        if 0:
            bankProfileEnergy = lambda **d: float(fbl_ref['bank'](da.const(d['energy'])))
            # control = bankProfile # Open loop, time
            control = bankProfileEnergy # Open loop
        else:
            nmpc = NMPC(fbl_ref=fbl_ref,dt=dt,Q=Q)
            control = nmpc.controller

        # Run
        t0 = time.time()
        output = da_sim.run(x0d,[control],StepsPerCycle=10,InputSample=sample,EnergyIV=True)
        t1 = time.time()
        # print("DA Integration time = {} s".format(t1-t0))
        xf = output[-1]
        self.traj = output
        self.xf = xf
        self.offset = -da.const(xf,array=True)+self.ref_xf
        # print "Nominal final states: {}".format(self.ref_xf)
        # print "DA final states: {}".format(da.const(xf))
        # print "Difference = {}".format(self.offset)
        # print output[1]
        return self

    def sample(self,expansion_points):
        """ Evaluates the expansion of the flow """
        xf_new = da.evaluate(self.xf+1*self.offset,self.vars,expansion_points)
        self.MC = xf_new
        return xf_new

    def plot(self):
        label = 'STT({})'.format(self.order)
        h = self.MC[:,3]
        v = self.MC[:,7]
        plt.figure(1)
        plt.plot(v,h,'ko',label=label)

        dr = self.MC[:,10]
        cr = self.MC[:,11]
        plt.figure(2)
        plt.plot(cr,dr,'ko',label='STT({})'.format(self.order))

    def compare(self,truth_data):
        import MCF

        samples = truth_data['samples'].T
        # samples = np.hstack((np.zeros((2000,6)),samples))
        h = [truth_data['states'][0][i][-1,3] for i in range(2000)]
        v = [truth_data['states'][0][i][-1,7] for i in range(2000)]

        he = np.abs(h-self.MC[:,3])
        ve = np.abs(v-self.MC[:,7])
        err = np.vstack((he,ve))
        print err.shape
        Q = np.diag((1/he.max(),1/ve.max()))**2# weighting matrix for norm of (he,ve) to account for units
        err_scalar = np.array([0.5*e.T.dot(Q).dot(e) for e in err.T])
        print err_scalar.shape
        print err_scalar.max()
        print err_scalar.min()
        plt.figure(3)
        plt.hist(err_scalar,bins=40)
        def big_err(x):
            return x > 0.15
        data = MCF.mcsplit(samples.T,err_scalar,big_err)
        MCF.mcfilter(*data, input_names=['CD','CL','p0','hs'], plot=True)
        # Generates error between true MC data and DAMC
        # Applies MCF


def Optimize():
    from scipy.optimize import differential_evolution
    from scipy.io import loadmat

    data = loadmat("E:\Documents\EDL\data\MC_NMPC_2000_EnergyTrigger_default.mat")
    samples = data['samples'].T
    samples = np.hstack((np.zeros((2000,8)),samples))
    damc = DAMC(order=2).reference()

    bounds = [(0,10),(0,10),(1,30)]
    sol = differential_evolution(Cost,args = (damc, samples), bounds=bounds, tol=1e-1, disp=True, polish=False)
    print "Optimized parameters are:"
    print sol.x
    print sol
    return

def Cost(inputs, damc, samples):
    damc.integrate(dt = inputs[2], Q = np.array([[inputs[0],0],[0,inputs[1]]])).sample(samples)

    h   = damc.MC[:,3] # km
    lon = damc.MC[:,5]
    lat = damc.MC[:,6]

    J = np.percentile(lon,99)-np.percentile(lon,1) + np.percentile(lat,99)-np.percentile(lat,1)
    return J

if __name__ == "__main__":

    # Optimize()

    from scipy.io import loadmat
    data = loadmat("E:\Documents\EDL\data\MC_NMPC_2000_EnergyTrigger_optimized.mat")
    samples = data['samples'].T
    samples = np.hstack((np.zeros((2000,8)),samples))
    if 1:
        h = [data['states'][0][i][-1,3] for i in range(2000)]
        v = [data['states'][0][i][-1,7] for i in range(2000)]

        dr = [data['states'][0][i][-1,10] for i in range(2000)]
        cr = [data['states'][0][i][-1,11] for i in range(2000)]
    else: # For data that is all the same length, as in time triggered runs
        h = data['states'][:,-1,3] #[data['states'][i,-1,3] for i in range(2000)]
        v = data['states'][:,-1,7]#[data['states'][i,-1,7] for i in range(2000)]

        dr = data['states'][:,-1,10]#[data['states'][i,-1,10] for i in range(2000)]
        cr = data['states'][:,-1,11]#[data['states'][i,-1,11] for i in range(2000)]


    damc = DAMC(order=1).reference()
    # print Cost([0.1,2,2.5], damc, samples)
    # print Cost([7.1,0.03,4.25], damc, samples) # First optimization run
    damc.integrate(dt = 4.25, Q = np.array([[7.1,0],[0,0.03]]))
    damc.sample(samples)

    # damc.compare(data)

    damc.plot()
    plt.figure(1)
    plt.plot(v,h,'r*',label='MC')
    plt.ylabel('Altitude (km)')
    plt.xlabel('Velocity (m/s)')
    #
    plt.figure(2)
    plt.plot(cr,dr,'r*',label='MC')
    plt.legend()
    plt.ylabel('Downrange (km)')
    plt.xlabel('Crossrange (km)')
    #
    plt.figure()
    eh = damc.MC[:,3]-h
    ev = damc.MC[:,7]-v
    plt.plot(ev,eh,'o')
    plt.ylabel('Altitude error (km)')
    plt.xlabel('Velocity error (m/s)')
    plt.title('STT({})'.format(damc.order))
    print "Mean altitude error = {} km".format(eh.mean())
    print "Altitude std dev = {} km".format(eh.std())
    print "Mean velocity error = {} m/s".format(ev.mean())
    print "Velocity std dev = {} m/s".format(ev.std())
    #
    plt.show()
