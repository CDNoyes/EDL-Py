""" Sliding Mode Observer """

import numpy as np
from scipy.misc import factorial
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from EntryEquations import Entry
from FBL import drag_dynamics

class SMO(object):
    def __init__(self):

        self.state = [0,0,0] # D, Ddot, disturbance
        self.history = [np.array(self.state)]
        self.K,self.alpha = self.__gains__()

        self.model = Entry() # Instantiate a nominal model


    def __call__(self, timeSteps, liftMeasurements, dragMeasurements, controls, radius, velocity, gamma):
        """ Steps forward by each delta-t in timeSteps, using the array of the same length dragMeasurements providing the measurements at each interval.
            Because measurements are typically taken much more frequently than the bank angle is updated, u===cos(bankAngle) is held constant over the entire span.
        """
        self.E = self.model.energy(radius,velocity)
        self.r = radius
        self.v = velocity
        self.Dmeasured = dragMeasurements
        for dt, L, D, u, r, v, fpa in zip(timeSteps, liftMeasurements, dragMeasurements, controls, radius, velocity, gamma):
            # Get gravity, density, and scale height from model
            h = self.model.altitude(r)
            g = self.model.gravity(r)
            rho = self.model.planet.atmosphere(r)[0]

            a,b = drag_dynamics(self.state[0], self.state[1], g, L, r, v, fpa, rho, self.model.planet.scaleHeight)
            self.state = odeint(self.__dynamics__, self.state, [0,dt], args=(D,a,b,u))[-1]
            # print self.state
            self.history.append(self.state)

    def __dynamics__(self, x, t, D_measured, a, b, u):
        e = D_measured-x[0]
        signe = np.tanh(10*e)                                       # smooth approximation to sign function

        dx = [ x[1] + self.alpha[0]*e + self.K[0]*signe,
               x[2] + self.alpha[1]*e + self.K[1]*signe + a + b*u,
                      self.alpha[2]*e + self.K[2]*signe ]
        return np.array(dx)

    def __gains__(self, poleLocation=1):
        """ Returns nonlinear and linear observer gains such that all three poles of the system are located at -poleLocation """
        x = np.array([1,2,3])
        C3 = (factorial(3)/(factorial(x)*factorial(3-x)))
        alpha = C3*poleLocation**x
        k = [5,0,0]
        k[1:3] = k[0]*np.array([2,1])*poleLocation**x[0:2]

        return k, alpha

    def process(self):
        self.history = np.vstack(self.history)[1:]

    def plot(self, mass=None):
        plt.figure()
        plt.plot(self.E,self.history[:,0],label='Observer')
        plt.plot(self.E,self.Dmeasured,'o',label='Measurements')
        if mass is not None:
            Dmodel = self.model.aeroforces(self.r,self.v,mass)[1]
            plt.plot(self.E,Dmodel,label='Model')
        plt.legend()
        plt.xlabel('Energy')
        plt.ylabel('Drag (m/s^2)')

        plt.figure()
        plt.plot(self.E, self.history[:,2])
        plt.xlabel('Energy')
        plt.ylabel('Disturbance (m/s^4)')
        plt.show()

def test():

    # Define a drag profile based on perturbed models. Store also the state profiles.
    # Propagate the SMO states using the reference data to estimate the disturbance
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
    sim = Simulation(cycle=Cycle(1),output=False,**EntrySim())
    banks = [-np.radians(30),np.radians(75),-np.radians(75),np.radians(30)]
    bankProfile = lambda **d: profile(d['time'],[62.30687581,  116.77385384,  165.94954234], banks, order=2)

    # Run the simulation
    # sample = None
    sample = [0.15,-0.15,0.1, 0.003]

    x0 = InitialState()
    output = sim.run(x0,[bankProfile],StepsPerCycle=10,InputSample=sample)

    # ######################################################
    # SMO Propagation
    # ######################################################
    smo = SMO()
    u = np.cos(sim.control_history[:,0])
    noisyDrag = sim.df['drag'].values *(np.random.normal(1,0.01,u.shape))
    smo(sim.df['time'].values, sim.df['lift'].values, noisyDrag, u, sim.df['radius'].values, sim.df['velocity'].values, sim.df['fpa'].values)
    smo.process()
    smo.plot(mass=sim.df['mass'])

    # compare(reference_sim.df['energy'].values, reference_sim.df['drag'].values, sim.df['energy'].values, sim.df['drag'].values)
    #
    # sim.show()

if __name__ == "__main__":
    test()
