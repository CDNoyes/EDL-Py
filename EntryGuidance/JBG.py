from numpy import sin, cos, arctan2
from numpy.linalg import norm
import numpy as np

from EntryVehicle import EntryVehicle


# Fuel optimal guidance with a constant throttle, constant thrust angle, and unconstrained downrange

def controller(**d):
''' Wraps the solve function to accept the Simulation input dict '''

    fpa = d['fpa']
    v = d['velocity']
    return 1, solve( m0=d['mass'], u0=v*cos(fpa), v0=v*sin(fpa), vehicle=d['vehicle'] )[1]

    
    
def solve(m0, u0, v0, uf=0, vf=0, throttle=1, vehicle=EntryVehicle(), disp_iter=False):
    ''' Solves for the optimal thrust angle (and final time) to transfer the vehicle from initial velocity to final velocity '''
    mu = np.pi
    g = 3.7
    dv = vf-v0
    du = uf-u0
    
    for i in range(5):
        tf = vehicle.ve*m0/(vehicle.Thrust*throttle)*(1-np.exp(-du/(vehicle.ThrustFactor*cos(mu)*vehicle.ve)))
        mu = arctan2(dv + g*tf, du)
        if disp_iter:
            print "Iteration {}:\ntf = {} s\nmu = {} deg\n".format(i,tf,np.degrees(mu))

    return tf, mu
    
   

def check():
    # Check if using JBG will satisfy:
    
    # hf > hmin ?
   
   # DR > target ?
   return True
   
def test():
    
    
    m0 = 8500
    ToW = 10
    T = ToW*m0*3.7
    
    EV = EntryVehicle(mass = m0, area = 15.8, Thrust = T, Isp = 295, ThrustFactor = 1)
    
    u0 = 550
    v0 = -200
    
    _,_ = solve(m0, u0, v0, vf = -10, vehicle=EV, disp_iter=True)
    
if __name__ == '__main__':
    test()