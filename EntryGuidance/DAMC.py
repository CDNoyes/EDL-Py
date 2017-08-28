import unittest

from Simulation import Simulation, Cycle, EntrySim, TimedSim
from Triggers import SRPTrigger, AccelerationTrigger, VelocityTrigger
from InitialState import InitialState
from Uncertainty import getUncertainty
from ParametrizedPlanner import profile

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
            print("Using default expansion order 2.")
            self.order = 2
        else:
            self.order=order


    def integrate(self):
        """ Performs forward integration of the expansion points """
        # ######################################################
        # Reference data generation
        # ######################################################
        reference_sim = Simulation(cycle=Cycle(1),output=True,**EntrySim())
        # banks = [-np.radians(30),np.radians(75),-np.radians(75),np.radians(30)]
        # switch = [    62.30687581,  116.77385384,  165.94954234]
        # bank = [-np.radians(30),np.radians(75),-np.radians(75),np.radians(30)]
        switch = [ 20.09737598,  139.4887652 ]
        banks = [np.radians(30),np.radians(75),np.radians(30)]

        bankProfile = lambda **d: profile(d['time'],switch=switch, bank=banks,order=2)



        x0 = InitialState()
        output_ref = reference_sim.run(x0,[bankProfile],StepsPerCycle=10)
        self.ref_traj=output_ref
        self.ref_xf = output_ref[-1]
        refs = reference_sim.getRef()
        dragprof = refs['drag']
        altprof = refs['altitude']
        print("Reference data generated.")
        # Setup DA simulation
        states = ['Entry']
        trigger = [VelocityTrigger(504)]

        da_sim = Simulation(cycle=Cycle(1), output=True, use_da=True, **{'states':states, 'conditions':trigger})

        xvars = ['r','lon','lat','v','fpa','psi','s','m']
        orders = [self.order*0]*6 + [0]*2
        x0d = [gd(val,name,order) for val,name,order in zip(x0,xvars,orders)]
        params =  ['CD','CL','rho0','sh']
        sample = [gd(0,name,self.order) for name in params]     # Expand around the nominal values
        xvars += params
        self.vars = xvars
        # Setup a controller
        control = bankProfile # Open loop

        # Run
        t0 = time.time()
        output = da_sim.run(x0d,[control],StepsPerCycle=10,InputSample=sample)
        t1 = time.time()
        print("DA Integration time = {} s".format(t1-t0))
        xf = output[-1]
        self.traj = output
        self.xf = xf
        self.offset = -da.const(xf,array=True)+self.ref_xf
        print "Nominal final states: {}".format(self.ref_xf)
        print "DA final states: {}".format(da.const(xf))
        print "Difference = {}".format(self.offset)
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


if __name__ == "__main__":
    from scipy.io import loadmat
    data = loadmat("E:\Documents\EDL\data\MC_OpenLoop_2000_TimeTrigger_NoLateral.mat")
    samples = data['samples'].T
    samples = np.hstack((np.zeros((2000,8)),samples))
    h = [data['states'][0][i][-1,3] for i in range(2000)]
    v = [data['states'][0][i][-1,7] for i in range(2000)]

    dr = [data['states'][0][i][-1,10] for i in range(2000)]
    cr = [data['states'][0][i][-1,11] for i in range(2000)]

    damc = DAMC(order=3).integrate()
    damc.sample(samples)

    # damc.compare(data)

    damc.plot()
    plt.figure(1)
    plt.plot(v,h,'r*',label='MC')
    plt.ylabel('Altitude error (km)')
    plt.xlabel('Velocity error (m/s)')

    plt.figure(2)
    plt.plot(cr,dr,'r*',label='MC')
    plt.legend()
    plt.ylabel('Downrange (km)')
    plt.xlabel('Crossrange (km)')

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

    plt.show()
