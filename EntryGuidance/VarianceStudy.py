import unittest

from Simulation import Simulation, Cycle, EntrySim, TimedSim
from Triggers import SRPTrigger, AccelerationTrigger
from InitialState import InitialState
from Uncertainty import getUncertainty
from HPC import profile

from pyaudi import gdual_double as gd
from pyaudi import abs
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from scipy.interpolate import interp1d

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from Utils.RK4 import RK4
from Utils import DA as da
from Utils import draw
from Utils.submatrix import submatrix

# Goals:
# Open loop covariance propagation (using DA) - with or w/o process noise
# Closed-loop covariance with basic drag tracking

# Outputs:
# Draw cov ellipses in h-v, lat-lon, fpa-heading?
# Show evolution of drag covariance, with drag tracking we should see a reduction

def drag_tracker(time,velocity,drag,bankprof,dragprof,**kwargs):
    # return bankprof(time=time) - .02*(drag-dragprof(velocity.constant_cf))*np.sign(bankprof(time=time)) # Constant gain
    return bankprof(time=time) - drag*.02/100*(drag-dragprof(velocity.constant_cf))*np.sign(bankprof(time=time)) # Constant gain

def alt_tracker(time,velocity,altitude,bankprof,altitudeprof,**kwargs):
    return bankprof(time=time) + 1*(altitude/1000-altitudeprof(velocity.constant_cf))*np.sign(bankprof(time=time)) # Constant gain

def first_order():
    tf = 160.

    reference_sim = Simulation(cycle=Cycle(1),output=False,**TimedSim(tf))
    da_sim = Simulation(cycle=Cycle(1), output=True, use_da=True, **TimedSim(tf))
    banks = [-np.pi/2, np.pi/2,-np.pi/9]

    bankProfile = lambda **d: profile(d['time'],[89.3607, 136.276], banks)


    x0 = InitialState()
    output_ref = reference_sim.run(x0,[bankProfile],StepsPerCycle=10)
    refs = reference_sim.getRef()
    dragprof = refs['drag']
    altprof = refs['altitude']
    xvars = ['r','lon','lat','v','fpa','psi','s','m']
    x0d = [gd(val,name,1) for val,name in zip(x0,xvars)]
    params =  ['CD','CL','rho0','sh']
    sample = [gd(0,name,1) for name in params] # Expand around the nominal values
    xvars += params
    
    dragProfile = partial(drag_tracker, bankprof=bankProfile,dragprof=dragprof)
    altProfile = partial(alt_tracker, bankprof=bankProfile,altitudeprof=altprof)
    specs = ['k--','r','b']
    for control,spec in zip([bankProfile],specs):                               # Just open loop
    # for control,spec in zip([bankProfile,dragProfile],specs):                 # open, drag tracking
    # for control,spec in zip([bankProfile,dragProfile,altProfile],specs):      # all 3
    # for control,spec in zip([dragProfile,altProfile],specs):                  # no open loop
    # for control,spec in zip([altProfile],['k--','r','b']):
        output = da_sim.run(x0d,[control],StepsPerCycle=10,InputSample=sample)

        P0 = np.diag([500,0,0,25,.001,0])**2 # Written in terms of std dev, then squared to convert to variance
        # P0 = np.diag([500,0,0,25,0,0])**2 # Written in terms of std dev, then squared to convert to variance
        if not specs.index(spec):
            # alt-vel
            plt.figure(1)
            plt.plot(da.const(output[:,7]),da.const(output[:,3]),'g')
            plt.xlabel('Velocity (m/s)')
            plt.ylabel('Altitude')
            # lat-lon
            plt.figure(2)
            plt.plot(da.const(output[:,5]),da.const(output[:,6]),'g')
            plt.axis('equal')
            plt.xlabel('Longitude (deg)')
            plt.ylabel('Latitude (deg)')
            # drag
            plt.figure(3)
            plt.plot(da.const(output[:,0]),da.const(output[:,13]),'g')
            plt.xlabel('Time (s)')
            plt.ylabel('Drag (m/s^2)')
        # for xf in [output[0],output[int(tf/2)],output[-1]]:
        for xf in output[::1]:
            stm = da.jacobian(xf[4:10],xvars)[:,0:6]

            Pf = stm.dot(P0).dot(stm.T)

            Pfhv = submatrix(Pf,[3,0],[3,0])
            Pfhv[1,1] /= 1000**2
            Pfhv[0,1] /= 1000
            Pfhv[1,0] /= 1000

            Pfll = submatrix(Pf,[1,2],[1,2])

            Dgrad = da.gradient(xf[13],xvars)[0:6]
            Dvar = Dgrad.T.dot(Pf).dot(Dgrad) #Dgrad[0]**2 * Pf[0,0] + Dgrad[3]**2 * Pf[3,3] + 2*Dgrad[0]*Dgrad[3]*Pf[0,3]
            Pd = np.array([[.001, 0],[0, Dvar]])

            draw.cov([xf[7].constant_cf,xf[3].constant_cf],cov=Pfhv,fignum=1,show=False,legtext="t={}".format(xf[0]),legend=False,linespec=spec)
            draw.cov([xf[5].constant_cf,xf[6].constant_cf],cov=Pfll,fignum=2,show=False,legend=False,linespec=spec)
            draw.cov([xf[0],xf[13].constant_cf],cov=Pd,fignum=3,show=False,legend=False,linespec=spec)

    plt.show()

def drag_update():
    tf = 240.

    reference_sim = Simulation(cycle=Cycle(1),output=False,**TimedSim(tf))
    banks = [-np.pi/2, np.pi/2,-np.pi/9]

    bankProfile = lambda **d: profile(d['time'],[89.3607, 136.276], banks)


    x0 = InitialState()
    output_ref = reference_sim.run(x0,[bankProfile],StepsPerCycle=10)
    refs = reference_sim.getRef()
    dragprof = refs['drag_energy']
    E0 = output_ref[0,1]
    Ef = output_ref[-1,1]
    def drag_inv(E):
        return -1/dragprof(E)

    from scipy.integrate import quad
    range_est = quad(drag_inv,E0,Ef)[0]
    print "Nominal drag profile achieves a range of {} km".format(range_est/1000)
    range_desired = range_est + 20*1e3
    c = (-range_desired+range_est)/range_est
    print "Linear update to c = {}".format(c)
    def new_drag_inv(E):
        return -1/((1+c)*dragprof(E))

    range_est_new = quad(new_drag_inv,E0,Ef)[0]
    print "Range error from drag update = {} m".format(range_est_new-range_desired)
    # cq = [0.25 + 0.5*np.sqrt(0.25-2*c),0.25 - 0.5*np.sqrt(0.25-2*c)] # Choose the one with the same sign as c, most likely...
    # cq = np.roots((1,-0.5,c/2))
    d = -range_est
    imax = 2
    poly = [1] + [stm(i,d)/stm(imax,d) for i in range(1,imax)] + [(-range_desired+range_est)/stm(imax,d)]
    cq = np.roots(poly)
    print "Second order update to c = {}".format(cq)
    def new_drag_inv2(E):
        return -1/((1+0.5*(c+cq[1]))*dragprof(E))
        # return -1/((1+cq[1])*dragprof(E))

    range_est_new = quad(new_drag_inv2,E0,Ef)[0]
    print "Range error from 2nd order  drag update = {} m".format(range_est_new-range_desired)

    # print "Simulating with new profile..."
    from FBL import fbl_controller
    fbl = fbl_controller(Ef=Ef,fbl_ref=reference_sim.getFBL())

    fbl.update(E0,range_desired)

def stm(i,delta):
    from scipy.misc import factorial
    return delta*(-1)**(i+1)*factorial(i)

if __name__ == "__main__":
    first_order()
    # drag_update()
