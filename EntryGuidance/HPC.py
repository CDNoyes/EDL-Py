""" Differential algebra based replanning using the parametrized planning method.
    aka hybrid predictor corrector
"""
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from EntryGuidance.Simulation import Simulation, Cycle, EntrySim, TimedSim
from EntryGuidance.Triggers import SRPTrigger, AccelerationTrigger
from EntryGuidance.InitialState import InitialState
from EntryGuidance.Uncertainty import getUncertainty
from EntryGuidance.ParametrizedPlanner import profile,profile2
import EntryGuidance.Apollo

from pyaudi import gdual_double as gd
from pyaudi import abs, sqrt
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from scipy.interpolate import interp1d

from Utils.RK4 import RK4
from Utils import DA as da

fs = 14 # Global fontsize for plotting

class controllerState(object):
    def __init__(self):
        self.tReplan = 0
        self.nReplan = 0

    def reset(self):
        self.tReplan = 0
        self.nReplan = 0

def controller(time, current_state, switch, bank0, reference, lon_target, lat_target, **kwargs):
    if not hasattr(controller, 'bank'): #or time < tReplan (we've gone back in time so its a new sim) # Perform a number of initializations
        print( "Initializing HPC controller.")
        controller.bankvars = ['bank{}'.format(i) for i,b in enumerate(bank0)]
        controller.bank = np.array([gd(val,var,2) for val,var in zip(bank0,controller.bankvars)])
        controller.ref = reference # Also need to update the reference each time we replan
        states = ['RangeControl']
        conditions = [SRPTrigger(0.0,700,100)]
        input = { 'states' : states,
              'conditions' : conditions }
        # controller.sim = Simulation(cycle=Cycle(1), output=True, use_da=True, **EntrySim()) # So that the simulation doesn't get rebuilt unnecessarily
        controller.sim = Simulation(cycle=Cycle(1), output=False, use_da=True, **input) # So that the simulation doesn't get rebuilt unnecessarily
        controller.nReplan = 0
        controller.tReplan = 0

    r,theta,phi,v,gamma,psi,s,m=current_state
    # Determine if a replanning is needed
    nReplanMax = 5
    tReplanMin = 30 # Minimum time between replannings
    # print "In HPC controller: Range Error {} km".format(np.abs(s-controller.ref(v))/1000)
    if v < 5300 and np.abs(s-controller.ref(v)) > 4e3 and controller.nReplan < nReplanMax and (time-controller.tReplan) > tReplanMin:
    # Don't check until velocity is monotonic, less than max replans, hasn't passed the last switch
        print( "Replanning triggered")
        controller.nReplan += 1
        controller.tReplan = time
        # One option is just to compare estimated range to go with the reference value at the same (velocity/energy/time)
        # Another option is to track using Apollo between plannings, and use the predicted range to go as the replanning tool
        # However, we should also be taking into account crossrange objective, so maybe a single quick integration to check sometimes?
        traj = predict(controller.sim, current_state, bankProfile = lambda **d: profile(d['time'], [sw-time for sw in switch], controller.bank), AR=kwargs['aero_ratios']) # Need to pass DA bank variables
        nInactive = np.sum(np.asarray(switch)<time) # The number of switches already passed (and therefore removed from optimization), but never remove the last element
        dbank = optimize(traj, lon_target, lat_target, controller.bankvars, nInactive)
        controller.bank += dbank

        # Need to evaluate traj at dbank, then use those states to update the reference
        vel = da.evaluate(traj[:,7], controller.bankvars, [dbank]).flatten()
        vel = np.flipud(vel) # Flipped to be increasing for interp1d limitation
        range = da.const(traj[:,10],array=True)
        range = da.evaluate(traj[:,10], controller.bankvars, [dbank]).flatten()
        rtg = np.flipud(range[-1]*1e3-range*1e3) # Range to go

        # plt.figure()
        # plt.plot(vel,controller.ref(vel))
        try:
            controller.ref = interp1d(vel, rtg, fill_value=(rtg[0],rtg[-1]), assume_sorted=True, bounds_error=False, kind='linear')
        except:
            print("Updating controller reference failed, using previous reference")

        # plt.plot(da.const(traj[:,7]),(traj[-1,10].constant_cf - da.const(traj[:,10],array=True))[::-1]*1e3)
        # plt.plot(v,s,'r*')
        # plt.plot(vel, controller.ref(vel),'k--')
        # plt.show()

    # If not, or if replanning is done:
    # Simply evaluate profile
    bank = profile(time, switch, da.const(controller.bank),order=1)

    return bank

def predict(sim, x0, bankProfile, AR):

    output = sim.run(x0,[bankProfile],StepsPerCycle=10,AeroRatios=AR)
    return output

def optimize(DA_traj, longitude_target, latitude_target, bankvars, nInactive):
    # NOTICE: targets must be given in degrees!!
    xf = DA_traj[-1]
    print( "Predicted final state: {}".format(da.const(xf)[4:10]))
    # Test some basic optimization:
    f = (xf[5]-longitude_target)**2 + (xf[6]-latitude_target)**2  # Lat/lon target  - works well
    # f = ((xf[3]-6.4)**2 + (1/10000.)*(xf[7]-800.0)**2)    # Alt/vel target - combination doesn't work well
    # f = (xf[3]-6.9)**2     # Alt target - works well
    # f = (xf[7]-840.0)**2    # Vel target - works well
    # f = -xf[3] # Maximizing altitude

    # Relaxed Newton Method:
    dopt = newton_step(f, bankvars, nInactive)
    dopt *= 15*np.pi/180/np.max(np.abs(dopt)) # Restricts the largest step size
    dopt = line_search(f, dopt, bankvars)       # Estimates the best step size along dopt
    print( "delta Bank: {}".format(dopt*180/np.pi))

    xf_opt = da.evaluate(xf,bankvars,[dopt])[0]
    print( "New final state: {}".format(xf_opt[4:10]))
    return np.asarray(dopt)


def newton_step(f, vars, nInactive):
    """ Returns the step direction based on gradient and hessian info """
    g = da.gradient(f, vars)
    H = da.hessian(f, vars)
    nu = len(g)
    #TODO watch out for non-invertible hessians
    res = [0]*nInactive # Pad with zeros for the inactive segments
    d = -np.dot(np.linalg.inv(H[nInactive:nu,nInactive:nu]),g[nInactive:nu])
    res.extend(d)
    return np.array(res)

def line_search(f, dir, vars):
    """ Performs a "brute force" line search along a given direction by simply evaluating
        the function at a large number of points and taking the lowest value found.
        No further refinement is done.
    """
    dirs = [a*dir for a in np.linspace(0,1,2000)]
    fnew = da.evaluate([f], vars, dirs).flatten()
    i = np.argmin(fnew)
    return dir*np.linspace(0,1,2000)[i]

def grid_search(f,vars):
    from scipy.optimize import differential_evolution as DE
    # dopt = DE(__grid_search_fun, args=(f,vars), popsize=500, bounds=((0,np.pi/4),(-np.pi/4,0),(-np.pi/4,np.pi/18)),disp=True,tol=1e-5) # True reasonable bounds
    dopt = DE(__grid_search_fun, args=(f,vars), popsize=100, bounds=((-np.pi/18,np.pi/4.5),(-np.pi/4.5,np.pi/18),(-np.pi/4,np.pi/18)),disp=False,tol=1e-2) # Bigger bounds
    return dopt.x

def __grid_search_fun(x, f, vars):
    return da.evaluate([f], vars, [x])[0,0]

# def optimize_profile():

# #################################################
# ################ Test Functions #################
# #################################################

def test_profile():
    """ Tests the bank profile for various numbers of switches using standard python variables. """
    lw = 2
    t = np.linspace(0,220,5000)
    label = ['Discontinuous','Continuous','Once Differentiable']
    for order in range(3):
        bank = profile(t, [70,115,150],[-np.pi/2, np.pi/2,-np.pi/9,np.pi/9],order=order)
        plt.plot(t,np.array(bank)*180/np.pi,label=label[order],lineWidth=lw)

    plt.xlabel('Time (s)',fontsize=fs)
    plt.ylabel('Bank angle (deg)',fontsize=fs)
    plt.legend()
    plt.axis([0,220,-95,95])
    plt.show()

def test_da_profile2():
    """ Performs the same tests but utilizing DA variables with profile2 """

    t = np.linspace(0,200,500)
    order = 1
    bank_inp = [gd(val,'bank{}'.format(i),order) for i,val in enumerate([-np.pi/2, np.pi/2,-np.pi/9,np.pi/9])]
    switch = [-10,70,115,] + [gd(val,'s{}'.format(i),order) for i,val in enumerate([150])] + [250]
    bank = np.array([profile2(ti, switch, bank_inp) for ti in t ])

    plt.plot(t, da.const(bank, array=True)*180/np.pi,'k--')

    dbank = (-1+2*np.random.random([5,len(bank_inp)]))*np.pi/9
    dswitch =  (-1+2*np.random.random([5,len(bank_inp)-3]))*10.
    eval_pt = np.concatenate((dbank,dswitch),axis=1)
    vars = ['bank{}'.format(i) for i in range(4)] + ['s{}'.format(i) for i in range(1)]
    bank_new = da.evaluate(bank,vars,eval_pt)

    for bn in bank_new:
        plt.plot(t,bn*180/np.pi)

    plt.show()


def test_da_profile():
    """ Performs the same tests but utilizing DA variables """
    t = np.linspace(0,200,500)
    bank_inp = [gd(val,'bank{}'.format(i),2) for i,val in enumerate([-np.pi/2, np.pi/2,-np.pi/9,np.pi/9])]

    bank = profile(t, [70,115,150], bank_inp)

    plt.plot(t, da.const(bank, array=True)*180/np.pi,'k--')

    dbank = (-1+2*np.random.random([5,len(bank_inp)]))*np.pi/9
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
    import time

    tf = 220
    reference_sim = Simulation(cycle=Cycle(1),output=False,**TimedSim(tf))
    da_sim = Simulation(cycle=Cycle(1), output=True, use_da=True, **TimedSim(tf))
    banks = [-np.pi/2, np.pi/2,-np.pi/9]
    bankvars = ['bank{}'.format(i) for i,b in enumerate(banks)]
    bank_inp = [gd(val,'bank{}'.format(i),1) for i,val in enumerate(banks)]
    bankProfile = lambda **d: profile(d['time'],[89.3607, 136.276], bank_inp)


    x0 = InitialState()
    t0 = time.time()
    output = da_sim.run(x0,[bankProfile],StepsPerCycle=10)
    tda = time.time()
    print( "DA integration time {}".format(tda-t0))

    xf = output[-1]

    # Test some basic optimization:
    # f = (xf[5]+71.5)**2 + (xf[6]+41.4)**2  # Lat/lon target  - works well
    # f = ((xf[3]-6.4)**2 + (1/10000.)*(xf[7]-800.0)**2)    # Alt/vel target - combination doesn't work well
    # f = (xf[3]-6.9)**2     # Alt target - works well
    # f = (xf[7]-840.0)**2    # Vel target - works well
    f = -xf[3] # Maximizing altitude
    # Relaxed Newton Method:
    # dopt = newton_step(f, bankvars)
    # dopt *= 15*np.pi/180/np.max(np.abs(dopt)) # Restricts the largest step size
    # dopt = line_search(f, dopt, bankvars)       # Estimates the best step size along dopt
    # print "delta Bank from single newton step: {}".format(dopt*180/np.pi)
    dopt = np.zeros_like(banks)

    # dopt = grid_search(f, bankvars) # Brute force, could work well since evaluating is so fast
    # print "delta Bank from DE: {}".format(dopt*180/np.pi)


    xf_opt = da.evaluate(xf,bankvars,[dopt])[0]

    dbank = (-1+2*np.random.random([500,len(bank_inp)]))*np.pi/9
    xf_new = da.evaluate(xf,bankvars,dbank)
    teval = time.time()
    print( "DA evaluation time {}".format(teval-tda))

    plt.figure(1)
    plt.plot(xf[7].constant_cf,xf[3].constant_cf,'kx')
    # plt.plot(xf_opt[7],xf_opt[3],'k^')

    for xfn in xf_new:
        plt.plot(xfn[7],xfn[3],'o',fillstyle='none')

    plt.figure(2)
    plt.plot(xf[5].constant_cf,xf[6].constant_cf,'kx')
    # plt.plot(xf_opt[5],xf_opt[6],'k^')

    for xfn in xf_new:
        plt.plot(xfn[5],xfn[6],'o',fillstyle='none')

    plt.figure(3)
    plt.plot(xf[8].constant_cf,xf[9].constant_cf,'kx')
    # plt.plot(xf_opt[8],xf_opt[9],'k^')

    for xfn in xf_new:
        plt.plot(xfn[8],xfn[9],'o',fillstyle='none')

    if True:
        xf_new_true = []
        t0 = time.time()
        for delta in dbank:
            bankProfile_double = lambda **d: profile(d['time'],[89.3607, 136.276], [a+b for a,b in zip(delta,banks)])
            output = reference_sim.run(x0,[bankProfile_double])
            xf_new_true.append(output[-1])
            plt.figure(1)
            plt.plot(output[-1,7], output[-1,3],'x')
            plt.figure(2)
            plt.plot(output[-1,5], output[-1,6],'x')
            plt.figure(3)
            plt.plot(output[-1,8], output[-1,9],'x')
        tint = time.time()
        # print "Integration times for truth comparison {} (includes plotting)".format(tint-t0)
        xf_new_true = np.array(xf_new_true)
        err = np.abs(xf_new-xf_new_true)
        dbanknorm = np.linalg.norm(dbank,axis=1)
        label =['Altitude error (m)','Longitude error (deg)','Latitude error (deg)','Velocity error (m/s)', 'Flight path error (deg)', 'Heading error (deg)']
        # conversion = [1,np.pi/180,np.pi/180,1,np.pi/180,np.pi/180]
        for i in range(6):
            plt.figure(4+i)
            plt.plot(dbanknorm*180/np.pi,err[:,i+4],'ko')
            plt.xlabel('Norm of Bank Deviations (deg)',fontsize=fs)
            plt.ylabel(label[i],fontsize=fs)

    # Add labels to each plot
    plt.figure(1)
    plt.ylabel('Altitude (km)')
    plt.xlabel('Velocity (m/s)')
    plt.figure(2)
    plt.ylabel('Longitude (deg)')
    plt.xlabel('Latitude (deg)')
    plt.figure(3)
    plt.ylabel('Flight path angle (deg)')
    plt.xlabel('Heading angle (deg)')
    plt.show()

def test_controller():
    import MPC as mpc
     # Plan the nominal profile:
    reference_sim = Simulation(cycle=Cycle(1),output=False,**EntrySim())
    switch = [129, 190]
    bank = [-np.radians(80), np.radians(80), -np.radians(30)]
    bankProfile = lambda **d: profile(d['time'],switch=switch, bank=bank,order=2)

    x0 = InitialState()
    output = reference_sim.run(x0,[bankProfile])

    references = reference_sim.getRef()
    drag_ref = references['drag']

    # Create the simulation model:

    states = ['PreEntry','RangeControl']
    conditions = [AccelerationTrigger('drag',2), SRPTrigger(0,700,10)] # TODO: The final trigger should be an input
    input = { 'states' : states,
              'conditions' : conditions }

    sim = Simulation(cycle=Cycle(1), output=True, **input)

    # Create guidance laws
    pre = partial(mpc.constant, value=bankProfile(time=0))
    hpc = partial(controller, switch=switch, bank0=bank, reference=references['rangeToGo'], lon_target=output[-1,5], lat_target=output[-1,6])
    controls = [pre, hpc]

    # Run the off-nominal simulation
    perturb = getUncertainty()['parametric']
    sample = None
    # sample = perturb.sample()
    # print sample
    # sample = [.1,-.1,-.05,0]
    sample = [.133,-.133,.0368,.0014] # Worst case sample from Apollo runs
    # sample = [-0.12,0.0, 0,0]
    s0 = reference_sim.history[0,6]-reference_sim.history[-1,6] # This ensures the range to go is 0 at the target for the real simulation
    x0_full = InitialState(1, range=s0, bank=np.radians(-80))

    # Single trajectory
    reference_sim.plot(plotEnergy=True, legend=False)
    output = sim.run(x0_full, controls, sample, FullEDL=True)
    sim.plot(compare=False)
    sim.show()


if __name__ == "__main__":
    # test_da_profile()
    test_da_profile2()
    # test_expansion()
    # test_controller()
