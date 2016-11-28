""" Heading Alignment via MPC """

from numpy import sin, cos, arcsin, arccos, sqrt, pi, radians
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.integrate import trapz
from MPC import constant
from functools import partial 

def controller(control_options, control_bounds, get_heading, **kwargs):
    
    sol = optimize(kwargs['current_state'], control_options, control_bounds, kwargs['aero_ratios'], get_heading)
    return sol.x

        
def optimize(current_state, control_options, control_bounds, aero_ratios, get_heading):
    from Simulation import Simulation, NMPCSim
    
    sim = Simulation(output=False, **NMPCSim(control_options))

    guess = [-pi/2]
    sol = minimize_scalar(cost, method='Bounded', bounds=control_bounds, args=(sim, current_state, aero_ratios, get_heading))
    
    return sol
    
def desiredHeading(lon_current, lat_current, lon_target, lat_target):

    # delta = 2*arcsin( sqrt( sin(0.5*(lat_current-lat_target))**2 + cos(lat_current)*cos(lat_target)*sin(0.5*(lon_current-lon_target))**2 ) )
    # heading = pi/2.0 - np.sign(lon_target-lon_current)*arccos( (sin(lat_target)-sin(lat_current)*cos(delta))/(sin(delta)*cos(lat_current)))
    
    if np.abs(lon_target-lon_current) < 1e-5:
        if lat_target-lat_current > 0:
            PHI = 0
        else:
            PHI = pi
    else:
        d = arccos(cos(lat_current)*cos(lat_target)*cos(lon_current-lon_target) + sin(lat_current)*sin(lat_target))
        PHI = np.sign(lon_target-lon_current)*arccos( (sin(lat_target)-sin(lat_current)*cos(d))/(cos(lat_current)*sin(d)) )
    heading = pi/2-PHI    
    return heading
    
    
def cost(u, sim, state, ratios, get_heading):

    controls = [partial(constant,value=u)]
        
    output = sim.run(state, controls, AeroRatios=ratios)
    
    time = output[:,0]
    vel = output[:,7]
    heading = radians(output[:,9])
    
    heading_desired = [get_heading(lon,lat) for lon,lat in radians(output[:,5:7])]
    
    integrand = (heading-heading_desired)**2
    
    return trapz(integrand, time)    
    
    
def test_desiredHeading():
    import matplotlib.pyplot as plt
    
    lon = np.linspace(-1,1,101)
    lat = np.linspace(-1,1,101)
    
    psi = np.array([desiredHeading(Lon,Lat,0,0) for Lon in lon for Lat in lat])

    xx,yy = np.meshgrid(np.degrees(lon),np.degrees(lat))
    psi.shape = (len(lon),len(lat))
    
    # plt.scatter(lon, lat, color=psi, alpha=0.5)
    CT = plt.contour(xx,yy, np.degrees(psi.T),18)
    # CT = plt.contourf(xx,yy, np.degrees(psi.T),18)
    plt.clabel(CT, fontsize=10)
    plt.xlabel('Longitude (deg)')
    plt.ylabel('Latitude (deg)')
    plt.show()
    
if __name__ == "__main__":
    test_desiredHeading()