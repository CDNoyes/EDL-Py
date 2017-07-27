''' Defines filters for estimation '''

import numpy as np

class EKF(object):

    def __init__(self):
        self.history = []

        self.cov    = None   # Current state covariance
        self.state  = None   # Current estimated state
        self.gain   = None   # Current Kalman gain 

    def dynamics(P,t,A,C,Q,R):
        """ Covariance dynamics, assuming A,Q, and R are constant on the integration interval
            Inputs
                P   -   Current state covariance matrix
                t   -   Current time
                A   -   Jacobian of the nonlinear (possibly closed-loop) dynamic model
                C   -   Jacobian of the nonlinear measurement model
                Q   -   Covariance of the process noise at time t
                R   -   Covariance of the measurement noise at time t
        """
        P.shape = A.shape # Vector to matrix

        return (A.dot(P) + P.dot(A.T) + Q - P.dot(C.T).dot(np.solve(R,C.dot(P)))).flatten()

    def gain(P,C,R):
        """ Computes the Kalman gain  """
        return P.dot(H.T).dot(np.linalg.inv(R))

def FadingMemory(currentValue, measuredValue, gain):
    return (1-gain)*(measuredValue-currentValue)
