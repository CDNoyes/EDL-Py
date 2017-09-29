"""
Model Predictive Controllers

    Joel's NMPC - use the taylor expansion of the current state to predict a future state, and determines the optimal control over the interval

"""

from scipy.optimize import minimize_scalar as minimize
from scipy.integrate import quad
from functools import partial
from numpy import pi
import numpy as np

import matplotlib.pyplot as plt

import unittest

from FBL import drag_dynamics

DA_MODE = False
if DA_MODE:
    from pyaudi import exp, acos, cos, sin
    from Utils.smooth_sat import smooth_sat
    import Utils.DA as da
else:
    from numpy import exp, sin, cos, arccos as acos

class NMPC(object):
    """ This is an implementation of the CTNPC for Entry Guidance with an additional drag profile update method """

    def __init__(self, fbl_ref, Q=np.array([[.1,0],[0,2]]), debug=False, dt=2.5):
        self.Q      = np.asarray(Q)
        self.dt     = dt

        self.drag   = fbl_ref['D']
        self.rate   = fbl_ref['D1']
        self.accel  = fbl_ref['D2']
        self.a      = fbl_ref['a']
        self.b      = fbl_ref['b']
        self.bank   = fbl_ref['bank']

        self.debug = debug

    def controller(self, energy, current_state, lift, drag, planet, rangeToGo, **kwargs):
        try:
            energy = energy.constant_cf
        except:
            pass

        r,lon,lat,v,fpa,psi,s,m = current_state

        h = r - planet.radius
        g = planet.mu/r**2
        rho = planet.atmosphere(h)[0]

        # convert dt into approximate change in energy
        dt = self.dt
        de = -dt*v*drag
        if DA_MODE:
            de = de.constant_cf

        a,b = drag_dynamics(drag,None,g,lift,r,v,fpa,rho,planet.scaleHeight)
        V_dot = -drag-g*sin(fpa)
        h_dot = v*sin(fpa)
        rho_dot = -h_dot*rho/planet.scaleHeight
        D_dot = drag*(rho_dot/rho + 2*V_dot/v)

        # h = dt # need time since Ddot and Dddot are time derivatives

        ref_drag = self.drag(energy)
        ref_rate = self.rate(energy)

        try: # vectorized
            b[0]
            W = np.array([0.5*dt**2 * b, dt*b])
            z = np.array([dt*D_dot + 0.5*dt**2 * a, dt*a])
            e = np.array([drag-ref_drag,D_dot-ref_rate])
            d = np.array([self.drag(energy+de)-self.drag(energy),self.rate(energy+de)-self.rate(energy)])
            p = np.diag(W.T.dot(self.Q).dot(W))
            q = np.diag(W.T.dot(self.Q).dot(e+z-d))

        except:
            W = np.array([[0.5*dt**2 * b], [dt*b]])
            z = np.array([[dt*D_dot + 0.5*dt**2 * a], [dt*a]])
            e = np.array([[drag-ref_drag],[D_dot-ref_rate]])
            d = np.array([[self.drag(energy+de)-self.drag(energy)],[self.rate(energy+de)-self.rate(energy)]])

            p = W.T.dot(self.Q).dot(W)
            q = W.T.dot(self.Q).dot(e+z-d)

        if DA_MODE:
            u_unbounded = (-q[0]/p[0])[0]
            if np.abs(u_unbounded.constant_cf) < 10:
                u = smooth_sat(u_unbounded)
            else:
                return float(self.bank(energy+de))
        else:
            u = np.clip(-q/p, 0, 0.96).squeeze()

        return acos(u)*np.sign(self.bank(energy+de))




class test_NMPC(unittest.TestCase):


    def test_nominal(self):
        """ Uses the NMPC controller to reconstruct a bank angle profile """
        from Simulation import Simulation, Cycle, EntrySim
        from Triggers import SRPTrigger, AccelerationTrigger
        from HPC import profile
        from InitialState import InitialState
        from Utils.compare import compare

        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        # ######################################################
        # Reference data generation
        # ######################################################
        reference_sim = Simulation(cycle=Cycle(1),output=False,**EntrySim())
        banks = [-np.radians(30),np.radians(75),-np.radians(75),np.radians(30)]
        bankProfile = lambda **d: profile(d['time'],[62.30687581,  116.77385384,  165.94954234], banks, order=2)

        x0 = InitialState()
        output_ref = reference_sim.run(x0,[bankProfile],StepsPerCycle=10)
        refs = reference_sim.getFBL()

        # ######################################################
        # Closed-loop entry
        # ######################################################
        pre = lambda **kwargs: banks[0]
        Q = np.array([[.1,0],[0,2]])
        nmpc = NMPC(fbl_ref=refs, Q=Q, Ef = reference_sim.df['energy'].values[-1], debug=True)

        states = ['PreEntry','RangeControl']
        conditions = [AccelerationTrigger('drag',.2), SRPTrigger(0,600,1000)]
        input = { 'states' : states,
                  'conditions' : conditions }
        controls = [pre, nmpc.controller]

        # Run the nominal simulation
        sample = None
        # sample = [0.1,-0.1,-0.03,0.003]
        # sample = [-0.15,0.15,-0.03, 0.003]
        # sample = [-0.1,-0.1,-0.03,0.003]

        s0 = reference_sim.history[0,6]-reference_sim.history[-1,6] # This ensures the range to go is 0 at the target for the real simulation
        print s0/1000.
        # x0_full = InitialState(1, range=s0, bank=banks[0], velocity=x0[3], fpa=x0[4])
        # sim = Simulation(cycle=Cycle(1), output=True, **input)
        # reference_sim.plot(plotEnergy=False, legend=False)
        # output = sim.run(x0_full, controls, sample, FullEDL=True)
        # sim.plot(compare=False)
        # compare(reference_sim.df['energy'].values, reference_sim.df['drag'].values, sim.df['energy'].values, sim.df['drag'].values)
        #
        # sim.show()

if __name__ == "__main__":
    unittest.main()
