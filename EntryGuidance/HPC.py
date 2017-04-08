""" Differential algebra based predictor corrector using the parametrized planning method.
    aka hybrid predictor corrector 
"""

from ParametrizedPlanner import maneuver
from pyaudi import gdual_double as gd 
import numpy as np 
import matplotlib.pyplot as plt 

import sys
from os import path 
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from Utils.RK4 import RK4
from Utils import DA as da 

def profile(T, switch, bank):
    n = len(bank)
    bank_out = []
    try:
        T[0]
        isScalar = False
    except:
        T = [T]
        isScalar = True
        
    for t in T:    
        if t <= switch[0]:
            bank_out.append(bank[0])
        else:
            for i,s in enumerate(switch[::-1]):
                if t>s:
                    tswitch = s
                    bankcur = bank[n-2-i]
                    banknew = bank[n-1-i]
                    break
            # bank_out.append(maneuver(t,tswitch,bankcur,banknew))
            bank_out.append(banknew)
            
    if isScalar:
        return bank_out[0]
    else:
        return bank_out
        
        
        
def test_profile():
    """ Tests the bank profile for various numbers of switches using standard python variables. """
    t = np.linspace(0,200,5000)
    bank = profile(t, [70,115,150],[-np.pi/2, np.pi/2,-np.pi/9,np.pi/9])
    plt.plot(t,np.array(bank)*180/np.pi)
    plt.show()
    
    
def test_da_profile():
    """ Performs the same tests but utilizing DA variables """
    t = np.linspace(0,200,500)
    bank_inp = [gd(val,'bank{}'.format(i),2) for i,val in enumerate([-np.pi/2, np.pi/2,-np.pi/9,np.pi/9])]

    bank = profile(t, [70,115,150], bank_inp)
    
    # print bank
    
    plt.plot(t, da.const(bank, array=True)*180/np.pi,'k--')
    
    dbank = (-1+2*np.random.random([15,len(bank_inp)]))*np.pi/36
    bank_new = da.evaluate(bank,['bank{}'.format(i) for i,b in enumerate(bank_inp)],dbank)

    for bn in bank_new:    
        plt.plot(t,bn*180/np.pi)
    
    plt.show()

def test_expansion():
    ''' Integrates a trajectory with nominal bank angle
        Then expands around different bank angles
        Then integrates true trajectories using those bank angles 
        And compares
    '''
    from EntryGuidance.Simulation import Simulation, Cycle, EntrySim, TimedSim
    from EntryGuidance.InitialState import InitialState
    import time
    
    tf = 220
    reference_sim = Simulation(cycle=Cycle(1),output=False,**TimedSim(tf))
    da_sim = Simulation(cycle=Cycle(1), output=True, use_da=True, **TimedSim(tf))
    banks = [-np.pi/2, np.pi/2,-np.pi/9]
    bankvars = ['bank{}'.format(i) for i,b in enumerate(banks)]
    bank_inp = [gd(val,'bank{}'.format(i),2) for i,val in enumerate(banks)]
    bankProfile = lambda **d: profile(d['time'],[59.3607, 136.276], bank_inp)
    
                                                
    x0 = InitialState()
    t0 = time.time()
    output = da_sim.run(x0,[bankProfile],StepsPerCycle=10)
    tda = time.time() 
    print "DA integration time {}".format(tda-t0)
    
    xf = output[-1]
    
    # Test some basic optimization: 
    f = (xf[5]+71.5)**2 + (xf[6]+41.4)**2 
    # Relaxed Newton Method:
    dopt = newton_step(f,['bank{}'.format(i) for i,b in enumerate(bank_inp)])
    dopt *= 25*np.pi/180/np.max(np.abs(dopt)) # Restricts the largest step size to 25 degrees
    dopt = line_search(f,dopt,bankvars)       # Estimates the best step size along dopt
    print "delta Bank from single newton step: {}".format(dopt*180/np.pi)
    xf_opt = da.evaluate(xf,['bank{}'.format(i) for i,b in enumerate(bank_inp)],[dopt])[0]
    
    dbank = (-1+2*np.random.random([5,len(bank_inp)]))*np.pi/9
    xf_new = da.evaluate(xf,['bank{}'.format(i) for i,b in enumerate(bank_inp)],dbank)
    teval = time.time()
    print "DA evaluation time {}".format(teval-tda)

    plt.figure(1)
    plt.plot(xf[7].constant_cf,xf[3].constant_cf,'kx')
    plt.plot(xf_opt[7],xf_opt[3],'k^')
    
    for xfn in xf_new:
        plt.plot(xfn[7],xfn[3],'o')
        
    plt.figure(2)
    plt.plot(xf[5].constant_cf,xf[6].constant_cf,'kx')
    plt.plot(xf_opt[5],xf_opt[6],'k^')
    
    for xfn in xf_new:
        plt.plot(xfn[5],xfn[6],'o')    
        
    t0 = time.time()    
    for delta in dbank:
        bankProfile_double = lambda **d: profile(d['time'],[59.3607, 136.276], [a+b for a,b in zip(delta,banks)])
        output = reference_sim.run(x0,[bankProfile_double])
        plt.figure(1)
        plt.plot(output[-1,7], output[-1,3],'*')
        plt.figure(2)
        plt.plot(output[-1,5], output[-1,6],'*')
    tint = time.time()
    print "Integration times for truth comparison {} (includes plotting)".format(tint-t0)
    
    plt.figure(1)
    plt.ylabel('Altitude (km)')
    plt.xlabel('Velocity (m/s)')
    plt.figure(2)
    plt.ylabel('Longitude (deg)')
    plt.xlabel('Latitude (deg)')
    
    plt.show()

def newton_step(f, vars):
    g = da.gradient(f, vars)
    H = da.hessian(f, vars)
    
    return -np.dot(np.linalg.inv(H),g)

def line_search(f, dir, vars):
    # f0 = f.constant_cf 
    dirs = [a*dir for a in np.linspace(0,1,2000)]
    fnew = da.evaluate([f], vars, dirs).flatten()
    i = np.argmin(fnew)
    return dir*np.linspace(0,1,2000)[i]
    
if __name__ == "__main__":
    # test_profile()
    # test_da_profile()
    test_expansion()