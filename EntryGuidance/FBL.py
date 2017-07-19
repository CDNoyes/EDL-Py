""" Feedback Linearization-based Entry Guidance """

import numpy as np
from numpy import sin,cos, arccos
from functools import partial

import unittest

from scipy.optimize import minimize_scalar as minimize
from scipy.integrate import quad

class fbl_controller(object):
    """
        Defines a feedback linearizing controller for entry guidance.
        The unique feature of this controller is a drag profile updating scheme
        that adjusts the profile whenever the predicted range error is greater
        than a user-specified tolerance. Two different forms of the update are
        included.
    """
    def __init__(self, Ef, fbl_ref, observer=False, update_type=1):
        self.observer = observer

        self.D0    = 0              # Used in update type 2
        self.drag  = fbl_ref['D']
        self.rate  = fbl_ref['D1']
        self.accel = fbl_ref['D2']

        self.Ef = Ef
        self.type=update_type

        self.a = fbl_ref['a']
        self.b = fbl_ref['b']
        self.bank = fbl_ref['bank'] # only usable until first update
        self.c = 0
        self.c_history = [0]
        self.E_history = [] # This gets initialized the first time the controller is called

    def trigger(self, E, rangeToGo, drag):
        """ Defines a trigger that determines when the update method is called """

        # Every T seconds

        # When the predicted error > tolerance
        if self.type == 1:
            def inv_drag(energy):
                return -1/(self.drag(energy)*(1+self.c))
        else:
            def inv_drag(energy):
                return -1/( (self.drag(energy)-self.drag(E))*(1+self.c) + drag)


        rangePredicted = quad(inv_drag, E, self.Ef)[0]
        err = np.abs(rangeToGo - rangePredicted)
        tol = 1e3 # x km error allowed before replanning
        return err > tol

    def update(self, E, rangeToGo, drag):
        """ Computes a suitable drag profile update of one of the two forms:
            1. Dnew = (1+c)*Dref
            2. Dnew = (1+c)*(Dref-Dref(E0)) + D(E0) # Ensures the initial condition matches the current drag measurement
        """

        if self.type == 1:
            def inv_drag(energy, c):
                return -1/(self.drag(energy)*(1+c))
        else:
            def inv_drag(energy, c):
                return -1/( (self.drag(energy)-self.drag(E))*(1+c) + drag)

        def cost(c):
            rangePredicted = quad(inv_drag, E, self.Ef, args=(c,))[0]
            return (rangeToGo-rangePredicted)**2

        sol = minimize(cost)
        # self.c = np.clip(sol.x, -0.5,0.5)
        if (sol.x - self.c) < 0.1: # disallow large changes? might be needed in some cases
            self.c = sol.x

        # print self.c
        self.D0 = drag
        self.c_history.append(self.c)
        self.E_history.append(E)
        # c = minimize(cost, method='bounded',bounds=(-0.1,0.1)) # bounded version, either works tbh
        # print c


    def controller(self, energy, current_state, lift, drag, bank, planet, time, rangeToGo, **kwargs):
        r,lon,lat,v,fpa,psi,s,m = current_state

        if not len(self.E_history):
            self.E_history.append(energy)
            self.D0 = drag

        if self.type and self.trigger(energy,rangeToGo,drag) and v > 1200 and v < 4800:
            self.update(energy,rangeToGo,drag)


        h = r - planet.radius
        g = planet.mu/r**2
        rho = planet.atmosphere(h)[0]
        u = cos(bank)

        # use references at energy value a small dt in the future
        dt = 3                                                     #2 + mild feedback -> < 0.06 m/s^2 peak drag error
        de = -dt*v*drag

        a,b = drag_dynamics(drag,None,g,lift,r,v,fpa,rho,planet.scaleHeight)
        drag_rate,_ = drag_derivatives(u, lift, drag, g, r, v, fpa, rho, planet.scaleHeight)

        q = 0.5*rho*v**2
        qmax = 0.5*4000**2
        Dmax = 85 # this should come from the nominal drag profile

        # kp = 0.2*q/qmax
        kp = 0.002 * drag/Dmax
        kd = 0.01  * drag/Dmax

        if self.type == 1:
            ref_drag = self.drag(energy+de)*(1+self.c)
            ref_rate = self.rate(energy+de)*(1+self.c)

        elif self.type == 2:
            ref_drag = (self.drag(energy+de)-self.drag(self.E_history[-1]))*(1+self.c) + self.D0
            ref_rate = self.rate(energy+de)*(1+self.c)

        else:
            ref_drag = self.drag(energy+de)
            ref_rate = self.rate(energy+de)

        du = kp*(drag-ref_drag) + kd*(drag_rate-ref_rate)
        # du = 0
        u = cos(self.bank(energy+de)) + du/b

        # u = (fbl_ref['D2'](energy+de)-a)/b #+ du/b
        # u = (fbl_ref['D2'](energy+de)-fbl_ref['a'](energy+de))/fbl_ref['b'](energy+de) #+ du/b

        bank = arccos(np.clip(u,0,1))*np.sign(self.bank(energy+de))
        return bank


def controller(energy, current_state, lift, drag, bank, planet, time, fbl_ref, observer=False, **kwargs):

    r,lon,lat,v,fpa,psi,s,m = current_state
    h = r - planet.radius
    g = planet.mu/r**2
    rho = planet.atmosphere(h)[0]
    u = cos(bank)

    # use references at energy value a small dt in the future
    dt = 2                                                     #2 + mild feedback -> < 0.06 m/s^2 peak drag error
    de = -dt*v*drag
    # energy += de

    # if observer: # Use reference + disturbance estimate
    # u = cos(fbl_ref['bank'](energy)) #- (disturbance)/b_ref(v)

    a,b = drag_dynamics(drag,None,g,lift,r,v,fpa,rho,planet.scaleHeight)
    # print "Control authority: {}".format(np.abs(b/a))
    drag_rate,_ = drag_derivatives(u, lift, drag, g, r, v, fpa, rho, planet.scaleHeight)

    q = 0.5*rho*v**2
    qmax = 0.5*1*4000**2

    kp = 0.05*q/qmax
    kd = 0*0.2*q/qmax

    # From sliding mode paper:
    # qmax = 1e6
    # omega0 = 2*np.pi/80/np.sqrt(1-0.3**2)
    # omega = q/qmax*omega0
    # kp = omega**2
    # kd = 2*0.3*omega

    du = -kp*(drag-fbl_ref['D'](energy+de)) - kd*(drag_rate-fbl_ref['D1'](energy+de))
    # print "du = {}".format(v/b)
    u = cos(fbl_ref['bank'](energy+de)) + du/b

    # u = (fbl_ref['D2'](energy+de)-a)/b #+ du/b
    # u = (fbl_ref['D2'](energy+de)-fbl_ref['a'](energy+de))/fbl_ref['b'](energy+de) #+ du/b

    bank = arccos(np.clip(u,0,1))*np.sign(fbl_ref['bank'](energy+de))

    # import matplotlib.pyplot as plt
    # plt.figure(666)
    # plt.plot(time, (a/b),'ko')

    return bank



def drag_dynamics(D, D_dot, g, L, r, V, gamma, rho, scaleHeight):
    """ Estimates the nonlinear functions a,b such that the second derivative of
        drag with respect to time is given by (a+b*u)
    """

    # CD appears only in terms involving CD_dot which we assume is negligible

    V_dot = -D-g*sin(gamma)
    g_dot = -2*g*V*sin(gamma)/r
    h_dot = V*sin(gamma)
    rho_dot = -h_dot*rho/scaleHeight

    if D_dot is None: # When using an observer, we used the observed estimate, otherwise we use this model estimate
        D_dot = D*(rho_dot/rho + 2*V_dot/V)

    a1 = D_dot*(rho_dot/rho + 2*V_dot/V) - 2*D*(V_dot/V)**2
    a2 = -2*D/V*(D_dot+g_dot*sin(gamma))
    a3 = -2*D*g*cos(gamma)**2 * (1/r - g/V**2)
    a4 = -D/scaleHeight*(-g-D*sin(gamma)+V**2/r*cos(gamma)**2)
    a = a1 + a2 + a3 + a4

    b1 = -2*D*L*g*cos(gamma)/V**2
    # b2 = D*L/h_dot*rho_dot/rho*cos(gamma) # simplifies to
    b2 = -D*L*cos(gamma)/scaleHeight
    b = b1+b2

    return a,b

def drag_derivatives(u, L, D, g, r, V, gamma, rho, scaleHeight):

    V_dot = -D-g*sin(gamma)
    h_dot = V*sin(gamma)
    rho_dot = -h_dot*rho/scaleHeight

    D_dot = D*(rho_dot/rho + 2*V_dot/V)

    a,b = drag_dynamics(D,D_dot,g,L,r,V,gamma,rho,scaleHeight)
    D_ddot = a + b*u

    return D_dot,D_ddot


def alt_dynamics(L,D,g,r,V,gamma):
    a = -g-D*sin(gamma) + V**2/r*cos(gamma)**2
    b = L*cos(gamma)
    return a,b


def alt_derivatives(u, L, D, g, r, V, gamma):
    hdot = V*sin(gamma)
    hddot = -g-D*sin(gamma) + V**2/r*cos(gamma)**2 + L*cos(gamma)*u

    return hdot, hddot

class test_FBL(unittest.TestCase):


    def test_drag_update(self):
        """ Uses a controller implementing drag profile updating """
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
        tf = 358.

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
        fbl_c = fbl_controller(Ef=reference_sim.df['energy'].values[-1],fbl_ref=refs, update_type=1)
        fbl = fbl_c.controller

        states = ['PreEntry','RangeControl']
        conditions = [AccelerationTrigger('drag',.2), SRPTrigger(-0.5,600,100)]
        input = { 'states' : states,
                  'conditions' : conditions }
        controls = [pre, fbl]

        # Run the nominal simulation
        sample = None
        sample = [0.1,-0.1,-0.03,0.003]
        # sample = [-0.1,-0.1,-0.03,0.003]
        s0 = reference_sim.history[0,6]-reference_sim.history[-1,6] # This ensures the range to go is 0 at the target for the real simulation

        x0_full = InitialState(1, range=s0, bank=banks[0], velocity=x0[3], fpa=x0[4])
        sim = Simulation(cycle=Cycle(1), output=True, **input)
        reference_sim.plot(plotEnergy=False, legend=False)
        output = sim.run(x0_full, controls, sample, FullEDL=True)
        sim.plot(compare=False)
        # compare(reference_sim.df['energy'].values, reference_sim.df['drag'].values, sim.df['energy'].values, sim.df['drag'].values)
        import matplotlib.pyplot as plt
        plt.figure(660)
        plt.plot(fbl_c.E_history, fbl_c.c_history)
        sim.show()


    def no_test_profile_reproduction(self):

        """ Integrates a trajectory (truly can be arbitrary), then reruns the trajectory using
            feedback linearization with no model mismatch. It should reproduce the profile with nearly
            no error.
        """
        from Simulation import Simulation, Cycle, TimedSim
        from Triggers import SRPTrigger, AccelerationTrigger
        from HPC import profile
        from InitialState import InitialState
        from Utils.compare import compare

        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        # ######################################################
        # Reference data generation
        # ######################################################
        tf = 358.

        reference_sim = Simulation(cycle=Cycle(1),output=False,**TimedSim(tf))
        banks = [-np.radians(80), np.radians(80),-np.pi/9]
        bankProfile = lambda **d: profile(d['time'],[89.3607, 136.276], banks, order=2)

        x0 = InitialState()
        output_ref = reference_sim.run(x0,[bankProfile],StepsPerCycle=10)
        refs = reference_sim.getFBL()
        # reference_sim.plot(plotEnergy=True, legend=False)
        # reference_sim.viz(ground=False,axesEqual=False)
        # reference_sim.viz(axesEqual=False)
        reference_sim.show()

        # ######################################################
        # Closed-loop entry
        # ######################################################
        pre = lambda **kwargs: banks[0]
        fbl = partial(controller, fbl_ref=refs)

        states = ['PreEntry','RangeControl']
        conditions = [AccelerationTrigger('drag',.2), SRPTrigger(0.5,600,100)]
        input = { 'states' : states,
                  'conditions' : conditions }
        controls = [pre, fbl]

        # Run the nominal simulation
        sample = None

        s0 = reference_sim.history[0,6]-reference_sim.history[-1,6] # This ensures the range to go is 0 at the target for the real simulation

        x0_full = InitialState(1, range=s0, bank=banks[0], velocity=x0[3], fpa=x0[4])
        sim = Simulation(cycle=Cycle(1), output=True, **input)
        reference_sim.plot(plotEnergy=False, legend=False)
        output = sim.run(x0_full, controls, sample, FullEDL=True)
        sim.plot(compare=False)
        compare(reference_sim.df['energy'].values, reference_sim.df['drag'].values, sim.df['energy'].values, sim.df['drag'].values)
        sim.show()

if __name__ == "__main__":
    unittest.main()
