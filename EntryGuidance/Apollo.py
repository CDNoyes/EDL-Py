""" Modified Apollo Final Phase Hypersonic Guidance for Mars Entry """

import numpy as np
from numpy import sin, cos, tan
from scipy.interpolate import interp1d

# To do: Turn this into a class and use the init method to set the reference and probably also "get_heading". 
# Then, replanning is simply a matter of running the optimizer from HEP, and recomputing the gains needed.

def controller(velocity, lift, drag, fpa, rangeToGo, bank, heading, latitude, longitude, energy, reference, bounds, get_heading, heading_error=0.06, use_energy=False, **kwargs):

    if use_energy:
        IV = energy
    else:
        IV = velocity
        
    Rp = predict_range(IV, drag, velocity*sin(fpa), reference)  
        
    LoD_com = LoD_command(IV, rangeToGo/1000., Rp, reference)
    sigma = bank_command(lift/drag, LoD_com)   
    
    # Lateral logic here
    if 1: # Bank corridor logic:
        if rangeToGo < 0:
            sign = np.sign(bank)
        else:
            sign = lateral(np.sign(bank), heading, latitude, longitude, heading_error, get_heading)
    else: # Reverse at same velocity/energy as the reference trajectory
        sign = np.sign(reference['U'](IV))
            
    return np.clip(sigma, *bounds)*sign
    
def lateral(bank_sign, heading, latitude, longitude, max_error, compute_heading):

    heading_desired = compute_heading(longitude,latitude)
    # print "Heading error: {} deg".format(np.degrees(heading-heading_desired))
    if np.abs(heading-heading_desired)>max_error:
        return np.sign(heading-heading_desired)
    else:        
        return bank_sign


def gains(sim, use_energy=False):
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
    
    # Cartesian coords
    # x_rel = radius*sin(lon)*cos(azi)
    # y_rel = radius*sin(lon)*sin(azi)
    # z_rel = radius*cos(lon)
    
    # r_corrected = np.array([np.dot(C3(-t*w),[x,y,z]).T for t,x,y,z in zip(time,x_rel,y_rel,z_rel)]) # Position vector
    
    # Vc = np.array([vel*sin(fpa),vel*cos(fpa)*cos(azi),vel*cos(fpa)*sin(azi)])
    # Vp_rel = np.array([np.dot(C3(-theta), np.dot(C2(phi),vc)) for theta, phi, vc in zip(lon,lat,Vc.T)])

    # omega = np.array([[0, -w, 0],[w, 0, 0], [0,0,0]])
    # Vp = Vp_rel + np.dot(omega,r_corrected.T).T # Velocity vector with planet rotation
    
    # Prep for backwards integration of adjoints
    l1 = 1.0                # s
    l2 = 0.0                # v
    l3 = 0.0                # gamma
    l4 = -1.0/tan(fpa[-1])  # h
    l5 = 0.0                # u
    
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
    
  
    
    # build the output dictionary
    if use_energy:
        vi = eref[0:iv]
    else:
        vi = vref[0:iv]
        
    f3[f3<0.01] = 0.01
    data = { 'F1'    : interp1d(vi,f1[:iv], fill_value=(f1[0],f1[iv]), assume_sorted=True, bounds_error=False),
             'F2'    : interp1d(vi,f2[:iv], fill_value=(f2[0],f2[iv]), assume_sorted=True, bounds_error=False),
             'F3'    : interp1d(vi,f3[:iv], fill_value=(f3[0],f3[iv]), assume_sorted=True, bounds_error=False),
             'RTOGO' : interp1d(vi,rtogo[:iv], fill_value=(rtogo[0],rtogo[iv]), assume_sorted=True, bounds_error=False),
             'RDTREF': interp1d(vi,rdtref[:iv], fill_value=(rdtref[0],rdtref[iv]), assume_sorted=True, bounds_error=False),
             'DREF'  : interp1d(vi,dref[:iv], fill_value=(dref[0],dref[iv]), assume_sorted=True, bounds_error=False),
             'LOD'   : interp1d(vi,lodref[:iv], fill_value=(lodref[0],lodref[iv]), assume_sorted=True, bounds_error=False),
             'U'     : interp1d(vi,uref,fill_value=(uref[0],uref[-1]), assume_sorted=True, bounds_error=False),
             'K'    : 1
             }
             
    return data
 
def plot_rp(output, reference, use_energy=True):
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
    
    plt.figure()
    rp = predict_range(vi, drag[iv:], hdot[iv:], reference)
    
    plt.plot(vi,rp-range[iv:]) 
    # plt.plot(vi[pos[iv:]],rp[pos[iv:]]-range[iv:][pos[iv:]],'o') 
    plt.xlabel('')
    plt.ylabel('Predicted range error (km)')
    plt.title('Negative -> undershoot, Positive -> overshoot')
    plt.plot(vi[signchange[iv:]],rp[signchange[iv:]]-range[iv:][signchange[iv:]],'o') 
    # plt.show()

 
def predict_range(V, D, r_dot, ref):
    return ref['RTOGO'](V) + ref['F1'](V)*(D-ref['DREF'](V)) + ref['F2'](V)*(r_dot-ref['RDTREF'](V))
    
    
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