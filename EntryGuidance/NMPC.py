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

    def __init__(self, Ef, fbl_ref, Q=np.array([[.1,0],[0,2]]), update_type=0, update_tol=5, debug=False, dt=2.5, timeIV=False):
        self.Q      = np.asarray(Q)
        self.dt     = dt
        self.timeIV = timeIV
        if not timeIV:
            self.drag   = fbl_ref['D']
            self.rate   = fbl_ref['D1']
            self.accel  = fbl_ref['D2']
            self.a      = fbl_ref['a']
            self.b      = fbl_ref['b']
            self.bank   = fbl_ref['bank']
        else:
            self.drag   = fbl_ref['D_t']
            self.rate   = fbl_ref['D1_t']
            self.accel  = fbl_ref['D2_t']
            self.a      = fbl_ref['a_t']
            self.b      = fbl_ref['b_t']
            self.bank   = fbl_ref['bank_t']

        # Update parameters
        self.c = 0
        self.c_history = [0]
        self.E_history = []         # This gets initialized the first time the controller is called
        self.D0 = 0
        self.Ef = Ef
        self.type = update_type
        self.tol = update_tol*1e3   # Assumes input is in km
        self.v_update = 0

        self.debug = debug

    def controller(self, energy, current_state, lift, drag, bank, planet, time, rangeToGo, aero_ratios,**kwargs):
        try:
            energy = energy.constant_cf
        except:
            pass

        r,lon,lat,v,fpa,psi,s,m = current_state

        if not len(self.E_history):
            self.E_history.append(energy)
            self.D0 = self.drag(energy) # Use reference, not measured, otherwise the initial profile will be wrong already
            self.v_update = v

        if self.type and v > 1200 and v < 5800 and self.trigger(energy,rangeToGo,drag,fpa) and (self.v_update-v > 100) :
            if self.debug:
                print("Update triggered at V = {} m/s".format(v))
            self.v_update = v
            self.update(energy,rangeToGo,drag,fpa)

        h = r - planet.radius
        g = planet.mu/r**2
        rho = planet.atmosphere(h)[0]
        u = cos(bank)

        # use references at energy value a small dt in the future
        dt = self.dt
        de = -dt*v*drag
        try:
            de = de.constant_cf
        except:
            pass

        # de = -(1/500.)*(self.E_history[0]-self.Ef)
        # dt = de/(-v*drag)
        # if self.debug:
            # print("delta E = {}".format(de))
            # print("delta t = {} s".format(dt))
        a,b = drag_dynamics(drag,None,g,lift,r,v,fpa,rho,planet.scaleHeight)
        V_dot = -drag-g*sin(fpa)
        h_dot = v*sin(fpa)
        rho_dot = -h_dot*rho/planet.scaleHeight
        D_dot = drag*(rho_dot/rho + 2*V_dot/v)

        h = dt # need time since Ddot and Dddot are time derivatives
        if self.timeIV:
            energy = time
            de = dt

        if self.type == 1:
            ref_drag = self.drag(energy)*(1+self.c)
            ref_rate = self.rate(energy)*(1+self.c)

        elif self.type == 2:
            ref_drag = (self.drag(energy)-self.drag(self.E_history[-1]))*(1+self.c) + self.D0
            ref_rate = self.rate(energy)*(1+self.c)

        else:
            ref_drag = self.drag(energy)
            ref_rate = self.rate(energy)

        W = np.array([[0.5*h**2 * b], [h*b]])
        z = np.array([[h*D_dot + 0.5*h**2 * a], [h*a]])
        e = np.array([[drag-ref_drag],[D_dot-ref_rate]])
        d = np.array([[self.drag(energy+de)-self.drag(energy)],[self.rate(energy+de)-self.rate(energy)]])*(1+self.c)

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
        # print u
        # import pdb
        # pdb.set_trace()
        return acos(u)*np.sign(self.bank(energy+de))


    def trigger(self, E, rangeToGo, drag, fpa):
        """ Defines a trigger that determines when the update method is called """

        # Every T seconds

        # When the predicted error > tolerance
        if self.type == 1:
            def inv_drag(energy):
                return -np.cos(fpa)/(self.drag(energy)*(1+self.c))
        else:
            def inv_drag(energy):
                return -np.cos(fpa)/( (self.drag(energy)-self.drag(E))*(1+self.c) + drag)


        rangePredicted = quad(inv_drag, E, self.Ef)[0]
        err = np.abs(rangeToGo - rangePredicted)
        if self.debug:
            print("Predicted range error = {} km (tol = {} km)".format(err/1000,self.tol/1000))

        return err > self.tol

    def update(self, E, rangeToGo, drag, fpa):
        """ Computes a suitable drag profile update of one of the two forms:
            1. Dnew = (1+c)*Dref
            2. Dnew = (1+c)*(Dref-Dref(E0)) + D(E0) # Ensures the initial condition matches the current drag measurement
        """

        if self.type == 1:
            def inv_drag(energy, c):
                return -np.cos(fpa)/(self.drag(energy)*(1+c))
        else:
            def inv_drag(energy, c):
                return -np.cos(fpa)/( (self.drag(energy)-self.drag(E))*(1+c) + drag)

        def cost(c):
            rangePredicted = quad(inv_drag, E, self.Ef, args=(c,))[0]
            return (rangeToGo-rangePredicted)**2

        sol = minimize(cost, method='bounded',bounds=(-0.9,self.c+0.2))
        self.c = np.clip(sol.x, -0.9,0.025+0.025*250/self.v_update)
        # c should be bounded from above to prevent large drag increases which may decrease range error but will also drop altitude significantly
        # larger changes in c may be okay later in the trajectory ?


        # print self.c
        self.D0 = drag
        self.c_history.append(self.c)
        self.E_history.append(E)
        # c = minimize(cost, method='bounded',bounds=(-0.1,0.1)) # bounded version, either works tbh

        if self.debug:
            print("New value of c = {}".format(self.c))
            self.trigger(E,rangeToGo,drag,fpa)




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
        nmpc = NMPC(fbl_ref=refs, Q=Q, Ef = reference_sim.df['energy'].values[-1], update_type=1,update_tol=2,debug=True)

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
    # unittest.main()
    test_sat()
