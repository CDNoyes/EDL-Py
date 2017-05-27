""" Sliding Mode Observer """ 

import numpy as np 
from scipy.misc import factorial 
from FBL import drag_dynamics 
from scipy.integrate import odeint 

class SMO(object):
    def __init__(self):
        self.state = [0,0,0] # D, Ddot, disturbance  
        self.history = [np.array(self.state)]
        self.K,self.alpha = self.__gains__()
        
    def __call__(self, timeSteps, dragMeasurements,u, g, L, r, V, gamma, rho, scaleHeight):
        """ Steps forward by each delta-t in timeSteps, using the array of the same length dragMeasurements providing the measurements at each interval. 
            Because measurements are typically taken much more frequently than the bank angle is updated, u===cos(bankAngle) is held constant over the entire span.
        """
        
        for dt,D in zip(timeSteps,dragMeasurements):
            a,b = drag_dynamics(self.state[0], self.state[1], g, L, r, V, gamma, rho, scaleHeight) 
            self.state = odeint(self.__dynamics, self.state, [0,dt], args=(D,a,b,u,self.K,self.alpha))
            self.history.append(self.state)
        
    def __dynamics__(self, x, t, D_measured, a, b, u, k, alpha):

        e = D_measured-x[0]
        signe = np.tanh(10*e) #smooth approximation to sign function
        
        dx = [ x[1] + alpha[0]*e + k[0]*signe,
               x[2] + alpha[1]*e + k[1]*signe + a + b*u,
                      alpha[2]*e + k[2]*signe ]

        return np.array(dx) 
    
    def __gains__(self, poleLocation=10):
        """ Returns nonlinear and linear observer gains such that all three poles of the system are located at -poleLocation """
        x = np.array([1,2,3])
        C3 = (factorial(3)/(factorial(x)*factorial(3-x)))
        alpha = C3*poleLocation**x
        k = [5,0,0]
        k[1:3] = k[0]*np.array([2,1])*poleLocation**x[0:2]
        
        return k, alpha
        
if __name__ == "__main__":
    observer = SMO()
    
    print observer.__gains__()