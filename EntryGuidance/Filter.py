''' Defines filters for estimation '''

import numpy as np
from scipy.integrate import odeint


class EKF(object):

    def __init__(self):
        import sys
        from os import path
        sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
        from EntryEquations import Entry
        self.model  = Entry()   # Nominal model

        # Estimator telemetry
        self.state_history  =  []
        self.covariance_history = []
        self.gain_history = []

        # Current estimator information
        self.cov    = None      # Current state covariance
        self.state  = None      # Current estimated state
        self.gain   = None      # Current Kalman gain

        # Current linearizations
        self.A      = None      # Linearized system model
        self.C      = None      # Linearized observation model

    def init(self,state,state_covariance):
        self.state = state
        self.cov = state_covariance

        self.state_history.append(self.state)
        self.covariance_history.append(self.cov)

    def linearize(self, u, measured_drag):
        """ Updates the linearizations of the state and measurement equations """

        J = self.model.jacobian_ad(self.state, u)#[0] # Pyaudi version returns both jacobian and hessian
        # print(J.shape)
        self.A = J[0:6,:][:,0:6] # Don't need the range to go or mass covariances

        self.C = np.array([ [-measured_drag/self.model.planet.scaleHeight],
                            [0],
                            [0],
                            [2*measured_drag/self.state[3]],
                            [0],
                            [0] ]).T

    def update_state(self, dt, u, measurement):
        """ Propagates the state estimate forward one step """
        drag = self.model.aeroforces(np.array([self.state[0]]),np.array([self.state[3]]),np.array([self.state[-1]]))[1]
        err = measurement - drag
        self.state = odeint(self.state_dynamics, self.state, [0,dt],args=(u,err))[-1]
        self.state_history.append(self.state)
        return

    def update_cov(self, dt, Q, R):
        """ Propagates the state covariance estimate forward one step """
        cov_vector = odeint(self.dynamics,self.cov.flatten(), [0,dt],args=(Q,R))[-1]
        cov_vector.shape = self.A.shape
        self.cov = cov_vector
        self.covariance_history.append(self.cov)
        return

    def update(self, dt, measurement, control, Q, R):
        """
        Linearizes the nominal model about the current state estimate
        Updates the Kalman gain
        Updates the state estimate
        Updates the covariance estimate
        """
        self.linearize(control, measurement)
        self.compute_gain(self.cov, R)
        self.update_state(dt, control, measurement)
        self.update_cov(dt, Q, R)
        return

    def state_dynamics(self,x,t,u,err):
        nom_dyn = self.model.dynamics(u)(x,t)
        nom_dyn[0:6] += self.gain.dot(err)
        return nom_dyn

    def dynamics(self,P,t,Q,R):
        """ Covariance dynamics, assuming A,Q, and R are constant on the integration interval
            Inputs
                P   -   Current state covariance matrix
                t   -   Current time
                Q   -   Covariance of the process noise at time t
                R   -   Covariance of the measurement noise at time t
        """
        P.shape = self.A.shape # Vector to matrix

        return (self.A.dot(P) + P.dot(self.A.T) + Q - P.dot(self.C.T).dot(np.linalg.solve(R,self.C.dot(P)))).flatten()

    def compute_gain(self,P,R):
        """ Computes the Kalman gain  """
        # print R
        self.gain = P.dot(self.C.T).dot(np.linalg.inv(R))
        self.gain_history.append(self.gain)
        return

    def process(self):
        self.state_history = np.vstack(self.state_history[:-1])


    def test(self):
        """ The methodology is to use an off-nominal simulation (open or closed loop)
            Initialize the filter with the mean initial condition and a nominal system model
            Pass (possibly noisy) measurements to the filter and estimate the true states
            using only drag? Examine observability? Seems unlikely that flight path and heading
            can be resolved accurately.
        """
        from Simulation import Simulation, Cycle, EntrySim
        from Triggers import SRPTrigger, AccelerationTrigger
        from HPC import profile
        from InitialState import InitialState,Perturb
        from Utils.compare import compare

        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        # ######################################################
        # Reference data generation
        # ######################################################
        reference_sim = Simulation(cycle=Cycle(1),output=False,**EntrySim())
        banks = [-np.radians(30),np.radians(75),-np.radians(75),np.radians(30)]
        bankProfile = lambda **d: profile(d['time'],[62.30687581,  116.77385384,  165.94954234], banks, order=2)

        x0_nav = InitialState()
        x0 = x0_nav[:]
        x0 = Perturb(x0_nav,[10,0,0,1,.001,0])  # initial state perturbations
        # sample = [-.1,.1,.013,-.001]        # parametric perturbations
        sample = None
        output_ref = reference_sim.run(x0,[bankProfile],StepsPerCycle=10,InputSample=sample)

        # ######################################################
        # Estimation via EKF
        # ######################################################
        drag = reference_sim.df['drag'].values                          # Truth data
        bank = np.radians(reference_sim.df['bank'].values)
        R = 1*(.01/3)**2                                                  # Variance, defined as a fraction of the true drag
        measurement_noise = np.random.normal(0,R**0.5,drag.shape)       # Takes standard deviation so 0.01/3 yields a 1-percent deviation in 3-sigma
        measurement_bias = 0
        drag_measured = drag * (1+measurement_noise) + measurement_bias
        measurement_variance = drag**2 * R                              # This is the covariance matrix (scalar in this case) for the kalman filter
        process_noise = np.zeros((6,6))                                 # Assume no process noise
        initial_covariance = np.diag([10**2,0.0001,0.0001,1,0.001,0.0001])
        # Initialize the estimator
        self.init(x0_nav,initial_covariance)
        initialized = False
        # Predict-update until the trajectory is complete
        dt = 1
        for i,group in enumerate(zip(drag_measured,bank,measurement_variance)):
            Dm,banki,Ri = group
            Ri = np.array([[Ri]])
            u = [banki,0,0]
            print("Drag measurement variance = {}".format(Ri))
            self.update(dt,Dm,u,process_noise,Ri)
        self.process()

        # Plot comparisons
        import matplotlib.pyplot as plt
        e = reference_sim.df['energy']
        df = reference_sim.df
        err = self.state_history-reference_sim.history
        # for i in range(0,6):
        #     plt.figure(i+1)
        #
        #     plt.plot(e,reference_sim.history[:,i],label='Truth')
        #     plt.plot(e,self.state_history[:,i],label='EKF')
        plt.figure(1)
        plt.plot(err[:,3],err[:,0]/1000)
        plt.xlabel('Velocity error (m/s)')
        plt.ylabel('Altitude error (km)')
        plt.figure(2)
        plt.plot(np.degrees(err[:,1]),np.degrees(err[:,2]))
        plt.xlabel('Longitude error (deg)')
        plt.ylabel('Latitude error (deg)')
        plt.figure(3)
        plt.plot(np.degrees(err[:,4]),np.degrees(err[:,5]))
        plt.xlabel('Flight Path error (deg)')
        plt.ylabel('Azimuth error (deg)')

        plt.figure(4)
        plt.plot(e,drag)
        plt.plot(e,drag_measured)

        plt.show()

def FadingMemory(currentValue, measuredValue, gain):
    return (1-gain)*(measuredValue-currentValue)

if __name__ == "__main__":
    EKF().test()
