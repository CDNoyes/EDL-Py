""" Modified Apollo Final Phase Hypersonic Guidance for Mars Entry """

import numpy as np
from numpy import sin, cos, tan
from scipy.interpolate import interp1d

# To do: Turn this into a class and use the init method to set the reference and probably also "get_heading". 
# Then, replanning is simply a matter of running the optimizer from HEP, and recomputing the gains needed.

def controller(velocity, lift, drag, fpa, rangeToGo, bank, heading, latitude, longitude, energy, reference, bounds, get_heading, heading_error=0.06, use_energy=False, use_drag_rate=False, use_lateral=True, **kwargs):

    if use_energy:
        IV = energy
        dIV = -drag*velocity
    else:
        IV = velocity
        
        
    alt_rate = velocity*sin(fpa)
    
    if use_drag_rate:
        hs = 9345.5 # Nominal scale height
        g = 3.71    
        drag_rate = drag*(-alt_rate/hs - 2*drag/velocity - 2*g*sin(fpa)/velocity)    
        Rp = predict_range_dr(IV, drag, drag_rate, reference)  
    else:
        Rp = predict_range(IV, drag, alt_rate, reference,b=1)  

        
    LoD_com = LoD_command(IV, rangeToGo/1000., Rp, reference)
    sigma = bank_command(lift/drag, LoD_com)   
    
    # Lateral logic here
    if use_lateral: # Bank corridor logic:
        if rangeToGo < 0:
            sign = np.sign(bank)
        else:
            sign = lateral(np.sign(bank), heading, latitude, longitude, heading_error, get_heading)
    else: # Reverse at same velocity/energy as the reference trajectory
        sign = np.sign(reference['U'](IV+dIV*5.5))
            
    return np.clip(sigma, *bounds)*sign
    
def lateral(bank_sign, heading, latitude, longitude, max_error, compute_heading):

    heading_desired = compute_heading(longitude,latitude)
    # print "Heading error: {} deg".format(np.degrees(heading-heading_desired))
    if np.abs(heading-heading_desired)>max_error:
        return np.sign(heading-heading_desired)
    else:        
        return bank_sign


def gains(sim, use_energy=False, use_drag_rate=False):
    """ Determines the sensitivities based on a reference trajectory. 
    
    """ 
    edl = sim.edlModel   
    w = edl.planet.omega
    
    traj = sim.output
    time = traj[:,0]
    energy = traj[:,1]
    bank = np.radians(traj[:,2])
    radius = traj[:,4]
    lon = np.radians(traj[:,5])
    lat = np.radians(traj[:,6])
    vel = traj[:,7]
    fpa = np.radians(traj[:,8])
    azi = np.radians(traj[:,9])
    rtgo = traj[-1,10]-traj[:,10]
    lift = traj[:,12]
    drag = traj[:,13]
    lod = lift*np.cos(bank)/drag
    altrate = vel*np.sin(fpa)
    
    # Prep for backwards integration of adjoints
    l1 = 1.0                # s
    l2 = 0.0                # v
    l3 = 0.0                # gamma
    l4 = 0*-1.0/tan(fpa[-1])  # h
    l5 = 0.0                # u
    
    L = [[l1,l2,l3,l4,l5]]
    
    hs = edl.planet.scaleHeight
    akm = 1000.0

    tfine = np.linspace(time[-1],time[0],1000) # backwards
    dt = tfine[-2]
    rtogo = interp1d(time, rtgo)(tfine) # Do I want this in meters or km?
    vref  = interp1d(time, vel)(tfine)
    eref  = interp1d(time, energy)(tfine)
    rref  = interp1d(time, radius)(tfine)
    rdtref = interp1d(time, altrate)(tfine)
    dref = interp1d(time, drag)(tfine)
    ddtref = np.diff(dref)/np.diff(tfine)
    lodref = interp1d(time, lod)(tfine)
    
    gamma = interp1d(time, fpa)(tfine)

    liftv = lodref*dref
    sg = sin(gamma)
    cg = cos(gamma)
    c1 = liftv/vref**2 + cg/rref + 3.71*cg/vref**2
    c2 = (vref/rref - 3.71/vref)*sg
    f1 = []
    f2 = []
    f3 = []
    iv = np.argmax(vref)
    uref = interp1d(time, bank)(tfine)[:iv]
    
    for i in range(tfine.shape[0]):
        
        f1.append(-hs/dref[i]*l4/akm) # Divide this gain by 1000 if I use rtogo in km
        if use_drag_rate:
            f2.append(l3/(-cg[i]*dref[i]*(vref[i]/hs + 2*3.71/vref[i]))/akm)
        else:
            f2.append(l3/(vref[i]*cg[i]*akm))
        f3.append(l5/akm)
        
        dl1 = 0
        dl2 = -cg[i]*l1 + 2*dref[i]/vref[i]*l2 - c1[i]*l3 -sg[i]*l4
        dl3 = vref[i]*sg[i]*l1 + 3.71*cg[i]*l2 + c2[i]*l3 - vref[i]*cg[i]*l4
        dl4 = -dref[i]/hs*l2 + l3*(liftv[i]/vref[i]/hs) + vref[i]*cg[i]/rref[i]
        dl5 = -dref[i]/vref[i]*l3
    
        l1 -= dt*dl1
        l2 -= dt*dl2
        l3 -= dt*dl3
        l4 -= dt*dl4
        l5 -= dt*dl5
        L.append([l1,l2,l3,l4,l5]) # This is just to keep them for plotting
        
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(eref[0:iv],f1[0:iv],label='D')
    # plt.plot(eref[0:iv],f2[0:iv],label='h-dot')
    # plt.plot(eref[0:iv],f3[0:iv],label='u')
    # plt.legend(loc='best')
    # plt.show()
    # build the output dictionary
    if use_energy:
        vi = eref[0:iv]
    else:
        vi = vref[0:iv]
        
    f3[f3<0.01] = 0.01
    data = { 'F1'     : interp1d(vi,f1[:iv], fill_value=(f1[0],f1[iv]), assume_sorted=True, bounds_error=False),
             'F2'     : interp1d(vi,f2[:iv], fill_value=(f2[0],f2[iv]), assume_sorted=True, bounds_error=False),
             'F3'     : interp1d(vi,f3[:iv], fill_value=(f3[0],f3[iv]), assume_sorted=True, bounds_error=False),
             'RTOGO'  : interp1d(vi,rtogo[:iv], fill_value=(rtogo[0],rtogo[iv]), assume_sorted=True, bounds_error=False),
             'RDTREF' : interp1d(vi,rdtref[:iv], fill_value=(rdtref[0],rdtref[iv]), assume_sorted=True, bounds_error=False),
             'DREF'   : interp1d(vi,dref[:iv], fill_value=(dref[0],dref[iv]), assume_sorted=True, bounds_error=False),
             'DDTREF' : interp1d(vi,ddtref[:iv], fill_value=(ddtref[0],ddtref[iv]), assume_sorted=True, bounds_error=False),
             'LOD'    : interp1d(vi,lodref[:iv], fill_value=(lodref[0],lodref[iv]), assume_sorted=True, bounds_error=False),
             'U'      : interp1d(vi,uref,fill_value=(uref[0],uref[-1]), assume_sorted=True, bounds_error=False),
             'K'      : 1
            }
             
    return data
 
def plot_rp(output, reference, use_energy, use_drag_rate, fignum=None, title=None, label=None, components=True):
    import matplotlib.pyplot as plt
    
    vel = output[:,7]
    range = output[-1,10]-output[:,10] # Range to go
    drag = output[:,13]
    hdot = vel*np.sin(np.radians(output[:,8]))
    energy = output[:,1]
    bank = np.sign(output[:,2])
    signchange = ((np.roll(bank, 1) - bank) != 0).astype(int)
    pos = hdot >= 0
    iv = np.argmax(vel)

    if use_energy:
        vi = energy[iv:]
    else:
        vi = vel[iv:]
    
    hs = 9345.5 # Nominal scale height
    g = 3.71    
    drag_rate = drag*(-hdot/hs - 2*drag/vel - 2*g*hdot/vel**2)  
    if use_drag_rate:
        rp = predict_range_dr(vi, drag[iv:], drag_rate[iv:], reference)
        rp_range = predict_range_dr(vi, drag[iv:], drag_rate[iv:], reference, a=0,b=0)
        rp_rate = predict_range_dr(vi, drag[iv:], drag_rate[iv:], reference, a=0)-rp_range
        rp_drag = predict_range_dr(vi, drag[iv:], drag_rate[iv:], reference, b=0)-rp_range
        
    else:
        rp = predict_range(vi, drag[iv:], hdot[iv:], reference)
        rp_range = predict_range(vi, drag[iv:], hdot[iv:], reference, a=0, b=0)
        rp_rate = predict_range(vi, drag[iv:], hdot[iv:], reference, a=0) - rp_range
        rp_drag = predict_range(vi, drag[iv:], hdot[iv:], reference, b=0) - rp_range
    
    if fignum is None:
        plt.figure()
    else:
        plt.figure(fignum)
    
    if label is not None:
        total_label = label
    else:
        total_label = 'Total predicted range error'
    
    plt.plot(vi,rp-range[iv:],label=total_label) 
    if components:
        plt.plot(vi,rp_rate,'--',label='rate component') 
        plt.plot(vi,rp_drag,'--',label='drag component') 
        plt.plot(vi,-range[iv:]+rp_range,'--',label='range component') 
    # plt.plot(vi[pos[iv:]],rp[pos[iv:]]-range[iv:][pos[iv:]],'o') 
    plt.xlabel('')
    plt.ylabel('Predicted range error (km)')
    if title is None:
        plt.title('Negative -> undershoot, Positive -> overshoot')
    else:
        plt.title(title)
    plt.legend(loc='best')
    # plt.plot(vi[signchange[iv:]],rp[signchange[iv:]]-range[iv:][signchange[iv:]],'o') 
    # plt.show()

 
def predict_range(V, D, r_dot, ref, a=1, b=1):
    return ref['RTOGO'](V) + a*ref['F1'](V)*(D-ref['DREF'](V)) + b*ref['F2'](V)*(r_dot-ref['RDTREF'](V))
    
def predict_range_dr(V, D, D_dot, ref, a=1, b=1):
    ''' Experimental version using drag rate instead of altitude rate. '''
    return ref['RTOGO'](V) + a*ref['F1'](V)*(D-ref['DREF'](V)) + b*ref['F2'](V)*(D_dot-ref['DDTREF'](V))    
    
def LoD_command(V, R, Rp, ref):
    return ref['LOD'](V) + ref['K']*(R-Rp)/ref['F3'](V)
    
    
def bank_command(LoD, LoD_com):
    return np.arccos(np.clip(LoD_com/LoD,-1,1))
    
    
def C1(x):
    return np.array([[1, 0, 0],
                    [0, np.cos(x), np.sin(x)],
                    [-np.sin(x),np.cos(x),0]])
    
def C2(x):
    return np.array([[np.cos(x), 0, -np.sin(x)],
                    [0, 1, 0],
                    [np.sin(x), 0, np.cos(x)]])
                    
def C3(x):
    return np.array([[np.cos(x), np.sin(x), 0],
                     [-np.sin(x), np.cos(x), 0],                   
                     [0, 0, 1]])
                     
                     
def ctrb():
    """ For a given reference trajectory (and therefore a fixed set of Apollo gains), 
        computes the (vel, fpa) subset of the controllable set to the target downrange.
        Trajectories terminate at the target downrange unless either the maximum altitude 
        or maximum velocity constraints are violated.
    
    """
    from Simulation import Simulation, Cycle, EntrySim, SRP
    from ParametrizedPlanner import HEPBankSmooth
    import HeadingAlignment as headAlign
    from Triggers import SRPTrigger, AccelerationTrigger
    from Uncertainty import getUncertainty
    from InitialState import InitialState
    from MPC import constant
    
    from functools import partial
    from numpy import pi
    from scipy.io import savemat
    
     # Plan the nominal profile:
    reference_sim = Simulation(cycle=Cycle(1),output=False,**EntrySim())
    bankProfile = lambda **d: HEPBankSmooth(d['time'],*[99.67614316,  117.36691891,  146.49573609], minBank=np.radians(30))
                                                
    x0 = InitialState()
    output = reference_sim.run(x0,[bankProfile])

    references = reference_sim.getRef()
    drag_ref = references['drag']
    use_energy=True
    use_drag_rate=False
    use_lateral=True
    aeg_gains = gains(reference_sim,use_energy=use_energy, use_drag_rate=use_drag_rate)
    
    # Create the full simulation model:
    states = ['PreEntry','RangeControl']
    conditions = [AccelerationTrigger('drag',2), SRPTrigger(0.5,700,10)]
    input = { 'states' : states,
              'conditions' : conditions }
              
    sim = Simulation(cycle=Cycle(1), output=False, **input)

    # Create the guidance laws
    get_heading = partial(headAlign.desiredHeading, lat_target=np.radians(output[-1,6]), lon_target=np.radians(output[-1,5]))
    pre = partial(constant, value=bankProfile(time=0))
    aeg = partial(controller, reference=aeg_gains,bounds=(0.5*pi/9.,pi/1.25), get_heading=get_heading,
                  use_energy=use_energy, use_drag_rate=use_drag_rate, use_lateral=use_lateral) 

    controls = [pre, aeg]
    
    # Run the simulations
    perturb = getUncertainty()['parametric']
    sample = None 
    s0 = reference_sim.history[0,6]-reference_sim.history[-1,6] # This ensures the range to go is 0 at the target for the real simulation

    rtg_limit = 1000 # meters
    max_iter = 12
    tol = 0.05
    theta = np.linspace(0, 2*pi*0.98, 100)
    # theta = [3*pi/2]
    ctrb_set = []
    count = 0
    for a,b in zip(np.sin(theta),np.cos(theta)):
        print "Direction {}/{}".format(count,len(theta))
        count += 1
        iter = 0
        scale = 0.5

        # for each direction defined by theta, find the maximum value of s such that the final range to go is less than the defined limit    
        # First we need to bracket it
        rtg_final = rtg_limit-1
        while np.abs(rtg_final) < rtg_limit:
            scale += 0.5
            iter += 1
            dv = 100*a*scale
            dfpa = 0.15*b*scale
            x0_full = InitialState(1, range=s0, bank=np.radians(30),velocity=5505+dv, fpa=np.radians(-14.15 + dfpa)) 
            output = sim.run(x0_full, controls, sample, FullEDL=True)
            rtg_final = sim.x[6]
            
        print "Bracketing completed in {} iterations".format(iter)    
        print "Upper bound in dir [{},{}] = {}".format(a,b,scale)

        
        # Now we refine it until s_max-s_min < tol
        if iter>1:
            scale_min = scale-0.5
        else:
            scale_min = 0
        scale_max = scale
        iter = 0
        diff = scale_max-scale_min
        
        while diff > tol and iter < max_iter:
            scale = 0.5*(scale_min + scale_max)

            iter += 1
            dv = 100*a*scale
            dfpa = 0.15*b*scale
            x0_full = InitialState(1, range=s0, bank=np.radians(30),velocity=5505+dv, fpa=np.radians(-14.15 + dfpa)) 
            output = sim.run(x0_full, controls, sample, FullEDL=True)
            rtg_final = sim.x[6]
            if np.abs(rtg_final) < rtg_limit:
                scale_min = scale
            else:
                scale_max = scale
            
            diff = scale_max-scale_min
        print "Refining terminated in {} iterations".format(iter)    
        print "S max in dir [{},{}] is between [{},{}]".format(a,b,scale_min,scale_max)
        print "delta V0 = {}".format(dv) 
        print "delta efpa = {}".format(dfpa)     
        ctrb_set.append([dv,dfpa])
        savemat('./data/Apollo_CTRB.mat', {'delta':ctrb_set})
        
        
def plot_ctrb(file):
    from scipy.io import loadmat
    import matplotlib.pyplot as plt 
    data = loadmat(file)
    d = np.array(data['delta'])
    plt.figure()
    # plt.scatter(d[:,0],d[:,1])
    plt.plot(d[:,0],d[:,1],'*--')
    plt.xlabel('Change in initial velocity (m/s)')
    plt.ylabel('Change in entry flight path angle (deg)')
    plt.show()
    
if "__main__" == __name__:
    # ctrb()
    # plot_ctrb(file='./data/Apollo_CTRB_50_1km.mat')
    
    from numpy import pi
    theta = np.linspace(0, 2*pi*0.98, 100)
   
    dv = 100*sin(theta)
    dfpa = 0.15*cos(theta)
    import matplotlib.pyplot as plt 
    
    plt.plot(dv,dfpa,'o--')
    plt.plot(dv*1.1,dfpa*1.1,'o--')
    plt.show()