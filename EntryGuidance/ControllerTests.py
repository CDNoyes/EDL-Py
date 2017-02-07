# A script for testing various EG controllers




from functools import partial
from numpy import pi
import numpy as np

def test_controller():
    from scipy.io import savemat, loadmat
    import matplotlib.pyplot as plt
    
    from Simulation import Simulation, Cycle, EntrySim, SRP
    from ParametrizedPlanner import HEPBank,HEPBankSmooth
    import HeadingAlignment as headAlign
    from Triggers import AccelerationTrigger, VelocityTrigger, RangeToGoTrigger, SRPTrigger
    from Uncertainty import getUncertainty
    from InitialState import InitialState
    import MPC as mpc
    # from Riccati import controller as SDRE
    # from SDC import time as sdc
    import Apollo
    
    # Plan the nominal profile:
    reference_sim = Simulation(cycle=Cycle(1),output=False,**EntrySim())
    bankProfile = lambda **d: HEPBankSmooth(d['time'],*[99.67614316,  117.36691891,  146.49573609], minBank=np.radians(30))
    # bankProfile = lambda **d: np.sin(d['time']/20)
                                                
    x0 = InitialState()
    output = reference_sim.run(x0,[bankProfile])

    references = reference_sim.getRef()
    drag_ref = references['drag']
    # reference_sim.plot(plotEnergy=True)
    # plt.show()
    use_energy=True
    aeg_gains = Apollo.gains(reference_sim,use_energy=use_energy)
    
    if 1:
        # Create the simulation model:
        if 0:    
            states = ['PreEntry','RangeControl','HeadingAlign']
            # conditions = [AccelerationTrigger('drag',4), VelocityTrigger(1300), VelocityTrigger(500)]
            # conditions = [AccelerationTrigger('drag',4), VelocityTrigger(1300), RangeToGoTrigger(0)]
            conditions = [AccelerationTrigger('drag',2), VelocityTrigger(1400), SRPTrigger(0.5,700,500)]
        else:
            states = ['PreEntry','RangeControl']
            conditions = [AccelerationTrigger('drag',2), SRPTrigger(0.5,700,500)]
        input = { 'states' : states,
                  'conditions' : conditions }
                  
        sim = Simulation(cycle=Cycle(1), output=True, **input)

        # Create some guidance laws
        
        option_dict = mpc.options(N=1,T=None)
        # option_dict = mpc.options(N=1,T=10)
        option_dict_heading = mpc.options(N=1,T=20)
        get_heading = partial(headAlign.desiredHeading, lat_target=np.radians(output[-1,6]), lon_target=np.radians(output[-1,5]))
        
        mpc_heading = partial(headAlign.controller, control_options=option_dict_heading, control_bounds=(-pi/2,pi/2), get_heading=get_heading)
        mpc_range = partial(mpc.controller, control_options=option_dict, control_bounds=(0,pi/1.5), references=references, desired_heading=get_heading)
        pre = partial(mpc.constant, value=bankProfile(time=0))
        aeg = partial(Apollo.controller, reference=aeg_gains,bounds=(pi/9.,pi/1.25), get_heading=get_heading,use_energy=use_energy) # This is what the MC have been conducted with
        # aeg = partial(Apollo.controller, reference=aeg_gains,bounds=(np.radians(5),pi/1.25), get_heading=get_heading,use_energy=use_energy)

        # A,B,C=sdc()
        # R = lambda x: np.array([[1e6*x[1]**2]]) # Schedule with dynamic pressure
        # Q = lambda x: np.array([[100000000]])
        # z = lambda v: np.array([[drag_ref(v)]])
        # sdre = partial(SDRE,A=A,B=B,C=C,Q=Q,R=R,z=z)
        
        # controls = [pre, mpc_range, mpc_heading]
        controls = [pre, aeg, mpc_heading]
        
        # Run the off-nominal simulation
        perturb = getUncertainty()['parametric']
        sample = None 
        # sample = perturb.sample()
        # print sample
        # sample = [.1,-.1,-.05,0]
        # sample = [.133,-.133,.0368,.0014] # Worst case sample from Apollo runs
        s0 = reference_sim.history[0,6]-reference_sim.history[-1,6] # This ensures the range to go is 0 at the target for the real simulation
        print s0
        x0_nav = x0 # + Errors in velocity and mass
        x0_full = InitialState(1) #np.array([r0, theta0, phi0, v0, gamma0, psi0, s0, 2804.0] + x0_nav + [1,1] + [np.radians(-15),0])

        if 0:
            output = sim.run(x0, controls, sample, FullEDL=False)
            
            reference_sim.plot()
            sim.plot(compare=False)

        else:
            if 1: # Single trajectory
                reference_sim.plot(plotEnergy=True, legend=False)
                output = sim.run(x0_full, controls, sample, FullEDL=True)
                sim.plot(compare=False)
            
            else: # Multiple
                N = 1000
                sim.set_output(False)
                samples = perturb.sample(N,'S')
                p = perturb.pdf(samples)
                
                if 1: # List comprehension, and save the results
                    stateTensor = [sim.run(x0_full, controls, sample, FullEDL=True) for sample in samples.T]
                    saveDir = './data/'
                    savemat(saveDir+'MC_Apollo_{}_K1_energy'.format(N),{'states':stateTensor, 'samples':samples, 'pdf':p})
                    
                else: # Raw loop, graph each trajectory
                
                    for iter,sample in enumerate(samples.T):
                        output = sim.run(x0_full, controls, sample, FullEDL=True)
                        print "Completed iteration {}".format(iter+1)
                        sim.plot(compare=False, legend=False)
                    sim.show()

        
        sim.show()
        
if __name__ == "__main__":
    import time
    t = time.time()
    test_controller()
    t_new = time.time()
    print "Elapsed time {} s".format(t_new-t)