""" Fuel optimal guidance with a constant acceleration arcs, optimal thrust angle, and downrange targeting """

from numpy import sin, cos, arctan2
from numpy.linalg import norm
import numpy as np


    
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
    
    
    
""" References 

Optimal Space Trajectories, J.P. Marec - page 74 has the definitions of Rtilde and Vtilde, notes that singular only if they have the same direction







"""