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
    return bankprof(time=time) - .1*(drag-dragprof(velocity.constant_cf))*np.sign(bankprof(time=time))

def first_order():
    tf = 260
    
    reference_sim = Simulation(cycle=Cycle(1),output=False,**TimedSim(tf))
    da_sim = Simulation(cycle=Cycle(1), output=True, use_da=True, **TimedSim(tf))
    banks = [-np.pi/2, np.pi/2,-np.pi/9]

    bankProfile = lambda **d: profile(d['time'],[89.3607, 136.276], banks)
    
                                                
    x0 = InitialState()
    output_ref = reference_sim.run(x0,[bankProfile],StepsPerCycle=10)
    dragprof = reference_sim.getRef()['drag']
    xvars = ['r','lon','lat','v','fpa','psi','s','m']
    x0d = [gd(val,name,1) for val,name in zip(x0,xvars)]
    
    clProfile = partial(drag_tracker, bankprof=bankProfile,dragprof=dragprof)
    
    for control,spec in zip([bankProfile,clProfile],['r--','k--']):
        output = da_sim.run(x0d,[control],StepsPerCycle=10)
        
        # P0 = np.diag([100,0,0,5,.00145,0])**2 # Written in terms of std dev, then squared to convert to variance 
        P0 = np.diag([500,0,0,25,0,0])**2 # Written in terms of std dev, then squared to convert to variance 

        plt.figure(1)
        plt.plot(da.const(output[:,7]),da.const(output[:,3]),'b')
        plt.figure(2)
        plt.plot(da.const(output[:,5]),da.const(output[:,6]),'b')
        for xf in output[::5]:
            stm = da.jacobian(xf[4:10],xvars)[:,0:6]
            
            Pf = stm.dot(P0).dot(stm.T)

            Pfhv = submatrix(Pf,[3,0],[3,0])
            Pfhv[1,1] /= 1000**2 
            Pfhv[0,1] /= 1000 
            Pfhv[1,0] /= 1000 
            
            Pfll = submatrix(Pf,[1,2],[1,2])
            
            draw.cov([xf[7].constant_cf,xf[3].constant_cf],cov=Pfhv,fignum=1,show=False,legtext="t={}".format(xf[0]),legend=False,linespec=spec)
            draw.cov([xf[5].constant_cf,xf[6].constant_cf],cov=Pfll,fignum=2,show=False,legend=False,linespec=spec)
    plt.show()
    
if __name__ == "__main__":
    first_order()