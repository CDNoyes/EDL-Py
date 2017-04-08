# A script for testing various EG controllers

from functools import partial
from numpy import pi
import numpy as np

from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt

from Simulation import Simulation, Cycle, EntrySim, SRP
from ParametrizedPlanner import HEPBank,HEPBankSmooth, HEPNR
import HeadingAlignment as headAlign
from Triggers import AccelerationTrigger, VelocityTrigger, RangeToGoTrigger, SRPTrigger
from Uncertainty import getUncertainty
from InitialState import InitialState
import MPC as mpc
import Apollo

def describe(sample):

    if sample[0] < -0.1:
        CD = 'Low'
    else:
        CD = 'High'
        
    if sample[1] < -0.1:
        CL = 'low'
    else:
        CL = 'high'    

    if sample[2] < 0:
        rho = 'less dense (in general)'
    else:
        rho = 'more dense (in general)'
        
    if sample[2]*sample[3] > 0:
        conj = 'and'
    else:
        conj = 'but'
        
    if sample[3] < 0:
        sh = 'less dense at high altitude'
    else:
        sh = 'more dense at high altitude'   
    
    return "{} drag, {} lift, {} {} {}".format(CD,CL,rho,conj,sh)    
        
def batch():
    import sys
    sys.path.append('./Misc/')
    from VDP import box_grid
    
    pts = box_grid(((-0.15,0.15),(-0.15,0.15),(-0.15,0.15),(-0.02,0.02)),2)
 
    # sort_column = 2
    # pts = pts[pts[:,sort_column].argsort()] 

    reference_sim = Simulation(cycle=Cycle(1),output=False,**EntrySim())
    bankProfile = lambda **d: HEPBankSmooth(d['time'],*[99.67614316,  117.36691891,  146.49573609], minBank=np.radians(30))

    x0 = InitialState()
    output = reference_sim.run(x0,[bankProfile])
    reference_sim.plot(legend=False,plotEnergy=True)
    plt.show()
    references = reference_sim.getRef()
    drag_ref = references['drag']
    use_energy=True
    use_drag_rate=False
    aeg_gains = Apollo.gains(reference_sim,use_energy=use_energy, use_drag_rate=use_drag_rate)

    states = ['PreEntry','RangeControl']
    conditions = [AccelerationTrigger('drag',2), SRPTrigger(0.5,700,500)]
    input = { 'states' : states,
              'conditions' : conditions }
              
    sim = Simulation(cycle=Cycle(1), output=False, **input)             
    
    get_heading = partial(headAlign.desiredHeading, lat_target=np.radians(output[-1,6]), lon_target=np.radians(output[-1,5]))
    pre = partial(mpc.constant, value=bankProfile(time=0))
    aeg = partial(Apollo.controller, reference=aeg_gains,bounds=(pi/9.,pi/1.25), get_heading=get_heading,use_energy=use_energy, use_drag_rate=use_drag_rate)
    controls = [pre, aeg]     
    
    s0 = reference_sim.history[0,6]-reference_sim.history[-1,6] # This ensures the range to go is 0 at the target for the real simulation    
    x0_full = InitialState(1, range=s0) 
    perturb = getUncertainty()['parametric']  
    p = perturb.pdf(pts.T)  
    stateTensor = []
    fignum = 1    
    for iter,sample in enumerate(pts):
        output = sim.run(x0_full, controls, sample*1/3., FullEDL=True)
        stateTensor.append(output)
        print "Completed iteration {}".format(iter+1)
        if not (iter)%8:
            fignum += 1
        Apollo.plot_rp(output, aeg_gains, use_energy, use_drag_rate=use_drag_rate, fignum=fignum, label=describe(sample),components=False)
                
    saveDir = './data/'
    savemat(saveDir+'Box_1sig_Apollo_K1_energy',{'states':stateTensor, 'samples':pts, 'pdf':p})    
    plt.show()
    
    
def constant_bank():
    reference_sim = Simulation(cycle=Cycle(1),output=False,**EntrySim())
    banks = [0, 5, 15, 30, 45, 60, 75, 90]
    # banks = [30, 45, 60, 75, 90]
    # banks = [20, 62, 90]
    
    # for bank in banks + [-b for b in banks[1:]]:
    coeff = []
    for bank in banks:
        
    
        x0 = InitialState(bank=np.radians(bank))
        bankProfile = lambda **d: np.radians(bank)
        output = reference_sim.run(x0,[bankProfile])
        
        if bank == banks[0]:
            output0 = output
            dref0 = reference_sim.getRef()['drag']

        if not bank in (0,90):
            output1 = output
            
        if bank == banks[-1]:
            output2 = output
            dref1 = reference_sim.getRef()['drag']
            
        if True:
            # polyfit each drag profile 
            order = 12
            mono = output[:,7] < (output[0,7]-10)
            vscale = 500
            pD = np.polyfit(output[mono,7]/vscale,output[mono,13], order)
            
            coeff.append(pD)
            plt.figure(66)
            plt.plot(output[:,7],output[:,13])
            plt.plot(output[mono,7], np.polyval(pD, output[mono,7]/vscale),'--')
    coeff = np.array(coeff)
    print coeff.shape
    cmin = [coeff[:,i].min() for i in range(coeff.shape[1])]
    cmax = [coeff[:,i].max() for i in range(coeff.shape[1])]
    print coeff
    print "Bounding coefficients for order {} expansion:".format(order)
    print cmin 
    print cmax
        # reference_sim.plot(legend=False,plotEnergy=False)
    # v = np.linspace(600,5000,100)
    # d_interp = 0.5*(dref0(v) + dref1(v))
    # plt.figure()
    # plt.plot(output0[:,7],output0[:,13],label = 'bank={}'.format(banks[0]))    
    # plt.plot(output1[:,7],output1[:,13])    
    # plt.plot(output2[:,7],output2[:,13],label = 'bank={}'.format(banks[-1]))  
    # plt.xlabel('Velocity (m/s)')
    # plt.ylabel('Drag (m/s^2)')
    # plt.plot(v,d_interp,'--',label='mean drag profile from 20 and 90 bank')    
    # plt.legend(loc='best')
    plt.show()    
    
    
def test_controller():

    
    # Plan the nominal profile:
    reference_sim = Simulation(cycle=Cycle(1),output=False,**EntrySim())
    # bankProfile = lambda **d: HEPNR(d['time'],*[9.3607, 136.276], minBank=np.radians(30))
    bankProfile = lambda **d: HEPBankSmooth(d['time'],*[99.67614316,  117.36691891,  146.49573609], minBank=np.radians(30))
    # bankProfile = lambda **d: np.radians(-30)
                                                
    x0 = InitialState()
    output = reference_sim.run(x0,[bankProfile])

    references = reference_sim.getRef()
    drag_ref = references['drag']
    # reference_sim.plot(plotEnergy=True)
    # plt.plot(output[:,1],output[:,2])
    # plt.show()
    use_energy=True
    use_drag_rate=False
    use_lateral=True
    aeg_gains = Apollo.gains(reference_sim,use_energy=use_energy, use_drag_rate=use_drag_rate)
    
    if 1:
        # Create the simulation model:
        if 0:    
            states = ['PreEntry','RangeControl','HeadingAlign']
            # conditions = [AccelerationTrigger('drag',4), VelocityTrigger(1300), VelocityTrigger(500)]
            # conditions = [AccelerationTrigger('drag',4), VelocityTrigger(1300), RangeToGoTrigger(0)]
            conditions = [AccelerationTrigger('drag',2), VelocityTrigger(1400), SRPTrigger(0.5,700,500)]
        else:
            states = ['PreEntry','RangeControl']
            conditions = [AccelerationTrigger('drag',2), SRPTrigger(0.0,700,10)]
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
        # pre = partial(mpc.constant, value=np.radians(85))
        aeg = partial(Apollo.controller, reference=aeg_gains,bounds=(0.5*pi/9.,pi/1.25), get_heading=get_heading,
                      use_energy=use_energy, use_drag_rate=use_drag_rate, use_lateral=use_lateral) # This is what the MC have been conducted with


        
        # controls = [pre, mpc_range, mpc_heading]
        controls = [pre, aeg, mpc_heading]
        
        # Run the off-nominal simulation
        perturb = getUncertainty()['parametric']
        sample = None 
        # sample = perturb.sample()
        # print sample
        # sample = [.1,-.1,-.05,0]
        # sample = [.133,-.133,.0368,.0014] # Worst case sample from Apollo runs
        # sample = [0.08,0.0, 0,0]
        s0 = reference_sim.history[0,6]-reference_sim.history[-1,6] # This ensures the range to go is 0 at the target for the real simulation
        
        x0_nav = x0 # + Errors in velocity and mass
        # x0_full = InitialState(1, range=s0, bank=np.radians(-30))
        # x0_full = InitialState(1, range=s0, bank=np.radians(30),fpa=np.radians(-13.65)) # 0.5 deg shallower 
        # x0_full = InitialState(1, range=s0, bank=np.radians(-30),fpa=np.radians(-13.95)) # 0.2 deg shallower 
        # x0_full = InitialState(1, range=s0, bank=np.radians(30),fpa=np.radians(-14.65)) # 0.5 deg steeper
        # x0_full = InitialState(1, range=s0, bank=np.radians(30),fpa=np.radians(-14.30)) # 0.15 deg steeper
        x0_full = InitialState(1, range=s0, bank=np.radians(30),velocity=5505+100) # 50 m/s faster
        # x0_full = InitialState(1, range=s0, bank=np.radians(30),velocity=5505-100) # 50 m/s slower

        if 1: # Single trajectory
            reference_sim.plot(plotEnergy=True, legend=False)
            output = sim.run(x0_full, controls, sample, FullEDL=True)
            Apollo.plot_rp(output, aeg_gains, use_energy, use_drag_rate=use_drag_rate)
            sim.plot(compare=False)
            print sim.x[6]
        else: # Multiple
            N = 1000
            sim.set_output(False)
            samples = perturb.sample(N,'S')
            p = perturb.pdf(samples)
            
            if 1: # List comprehension, and save the results
                stateTensor = [sim.run(x0_full, controls, sample, FullEDL=True) for sample in samples.T]
                saveDir = './data/'
                savemat(saveDir+'MC_Apollo_{}_K1_energy_no_rate'.format(N),{'states':stateTensor, 'samples':samples, 'pdf':p})
                
            else: # Raw loop, graph each trajectory
            
                for iter,sample in enumerate(samples.T):
                    output = sim.run(x0_full, controls, sample, FullEDL=True)
                    print "Completed iteration {}".format(iter+1)
                    sim.plot(compare=False, legend=False)
        
        sim.show()
        
if __name__ == "__main__":
    import time
    t = time.time()
    # test_controller()
    # batch()
    constant_bank()
    t_new = time.time()
    print "Elapsed time {} s".format(t_new-t)