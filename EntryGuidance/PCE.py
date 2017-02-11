''' Generic functionality to build a PCE model out of simulation outputs '''

import chaospy as cp
import numpy as np
from scipy.io import savemat, loadmat
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from functools import partial
import time

from Simulation import Simulation, Cycle, EntrySim
from ParametrizedPlanner import HEPBankSmooth
import HeadingAlignment as headAlign
from Triggers import AccelerationTrigger, SRPTrigger
from Uncertainty import getUncertainty
from InitialState import InitialState
import MPC as mpc
import Apollo


def build(p, sim, pdf, x0, fun, **kwargs):


    polynomials = cp.orth_ttr(order=1, dist=pdf)
    samples,weights = cp.generate_quadrature(order=1, domain=pdf, rule="Gaussian")
    print "Running {} trajectories to build PCE model...".format(samples.shape[1])
    print "...using {}".format(p)
    stateTensor = [fun(p, sim, x0, s, **kwargs) for s in samples.T]
    PCE = cp.fit_quadrature(polynomials,samples,weights,stateTensor)
    E = cp.E(poly=PCE,dist=pdf)
    V = cp.Var(poly=PCE,dist=pdf)
    print "PCE Expectation: {} ".format(E)
    print "PCE Variance: {} ".format(V)
    return E + V**0.5
    
  
def optimize():
    reference_sim = Simulation(cycle=Cycle(1),output=False,**EntrySim())
    bankProfile = lambda **d: HEPBankSmooth(d['time'],*[99.67614316,  117.36691891,  146.49573609], minBank=np.radians(30))
                                                
    x0 = InitialState()
    output = reference_sim.run(x0,[bankProfile])
    use_energy=True
    aeg_gains = Apollo.gains(reference_sim,use_energy=use_energy)
    states = ['PreEntry','RangeControl']
    conditions = [AccelerationTrigger('drag',2), SRPTrigger(0.5,700,500)]
    input = { 'states' : states, 'conditions' : conditions }   
    sim = Simulation(cycle=Cycle(1), output=False, **input)
    
    perturb = getUncertainty()['parametric']

    get_heading = partial(headAlign.desiredHeading, lat_target=np.radians(output[-1,6]), lon_target=np.radians(output[-1,5]))
    x0_full = InitialState(full_state=True)
    bounds = [(0,25), (0.0,np.radians(45))]
    t0 = time.time()
    sol = differential_evolution(partial(build,gains=aeg_gains,get_heading=get_heading), args=(sim, perturb, x0_full, miss_distance), bounds=bounds, tol=1e-1, disp=True, polish=False)
    t_build = time.time()-t0
    print "Optimization took {} s".format(t_build)
    print "Optimal overcontrol: {}".format(sol.x[0])
    print "Optimal corridor: {} deg".format(np.degrees(sol.x[1]))
    
def miss_distance(p, sim, x0, sample, gains, get_heading):
    ''' Computes the miss distance based on Apollo guidance. 
        The design parameters in p are the overcontrol gain K, and the azimuth-error corridor limit. 
    '''
       
    gains['K'] = p[0]
       
       
    pre = partial(mpc.constant, value=np.radians(-30))
    aeg = partial(Apollo.controller, reference=gains, bounds=(np.pi/9., np.pi/1.25), get_heading=get_heading, use_energy=True, heading_error=p[1])
    output = sim.run(x0, [pre,aeg], sample, FullEDL=True)

    Xf = output[-1,:]
    hf = Xf[3]
    dr = Xf[10] #in km
    cr = Xf[11]    
    
    # miss_dist = np.sqrt(cr**2 + (dr-x0[6]/1000)**2)
    miss_dist = np.sqrt(cr**2 + (dr-905.2)**2)
    return miss_dist
    
    
def test():
    reference_sim = Simulation(cycle=Cycle(1),output=False,**EntrySim())
    bankProfile = lambda **d: HEPBankSmooth(d['time'],*[99.67614316,  117.36691891,  146.49573609], minBank=np.radians(30))
    # bankProfile = lambda **d: np.sin(d['time']/20)
                                                
    x0 = InitialState()
    output = reference_sim.run(x0,[bankProfile])
    use_energy=True
    aeg_gains = Apollo.gains(reference_sim,use_energy=use_energy)
    states = ['PreEntry','RangeControl']
    conditions = [AccelerationTrigger('drag',2), SRPTrigger(0.5,700,500)]
    input = { 'states' : states, 'conditions' : conditions }   
    sim = Simulation(cycle=Cycle(1), output=False, **input)
    
    perturb = getUncertainty()['parametric']

    get_heading = partial(headAlign.desiredHeading, lat_target=np.radians(output[-1,6]), lon_target=np.radians(output[-1,5]))
    x0_full = InitialState(full_state=True)
    t0 = time.time()
    build([1,0.06], sim, perturb, x0_full, miss_distance, gains=aeg_gains, get_heading=get_heading)
    t_build = time.time()-t0
    print "PCE build took {} s".format(t_build)
    
if __name__ == "__main__":
    optimize()
    # test()
