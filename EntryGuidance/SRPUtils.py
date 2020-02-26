import numpy as np
import sys 

sys.path.append("./")
from EntryGuidance.Planet import Planet 
from EntryGuidance.Target import Target
from Utils.boxgrid import boxgrid 

def srp_from_entry(entry_state, rtg, cr, target_alt=0):
    """ Utility to convert entry state to the cartesian coordinates needed
    
    By assumption, the "good" rtg will be positive (negative indicates an overshoot of the target)
    and we will assume (force) all crossrange values to be positive, and flip the correspond crossrange velocities.
    This allows us to use a smaller (or denser) table since we do not need any negative crossrange cases 
    
    
    """

    
    if len(entry_state) == 7:
        r,th,ph,v,fpa,psi,m = entry_state
    else:
        r,th,ph,v,fpa,psi,m = entry_state.T
           
#     s = np.sign(cr)
    cr = np.abs(cr)
    
    x_srp = np.array([rtg,
    cr,
    r-3397e3-target_alt,
    -v*np.cos(fpa),  # this is negative since rtg is positive 
#     0*v, # these may be positive or negative but crossrange is always taken to be positive 
    v*np.sin(fpa)])
    return x_srp.T


def range_from_entry(entry_state, target):
    """ This only appropriate for use in generating SRP ignition conditions"""

    if len(entry_state) == 7:
        r,th,ph,v,fpa,psi,m = entry_state
    else:
        r,th,ph,v,fpa,psi,m = entry_state.T
        
    planet = Planet('Mars')
    rtogo, crossrange = planet.range(th, ph, psi, *target, km=False)
    
    overshoot = th > target[0]
    rtogo[overshoot] *= -1 
    
    return rtogo, crossrange


def test():
    # Demonstrate that the conversion process is correct 

    m0 = 8500
    r = 3397e3 + 2e3
    v = 500
    fpa = np.radians(-15)
    d = 4/3397
    target = Target(0,0,0)

    for azi in np.radians([45, 0, -45]):
        entry_state = np.array([r, -d, -d, v, fpa, azi,m0], ndmin=2)
        rtg, cr = range_from_entry(entry_state, target.coordinates())
        print("\nRTG: {:.1f}, CR: {:.1f}".format(rtg[0]/1000, cr[0]/1000))
        x_srp = srp_from_entry(entry_state, rtg, cr, target_alt=target.altitude)
        print(x_srp)


def plot_contours():
    import pickle 
    import matplotlib.pyplot as plt 
    import os 

    m0 = 8500
    r = 3397e3 + 2e3
    v = 530
    fpa = np.radians(-10)
    # need to vary lat/lon and azimuth 
    d = 4/3397
    N = 50
    # azi = np.radians([-10,10])
    azi = (0,0) 
    entry_state = boxgrid([(r,r), (-d*1.5, -0.15/3397), (-d,d), (v,v), (fpa,fpa), azi, (m0, m0)], [1,N,N,1,1,1,1], interior=True).T

    psi = np.arctan2(-entry_state[2]*3397, -entry_state[1]*3397)
    entry_state[5] = psi

    # print(entry_state.shape)
    target = Target(0,0,0)
    rtg, cr = range_from_entry(entry_state, target.coordinates())
    # plt.tricontourf(rtg, cr, np.degrees(entry_state[5]), 30, cmap='RdBu')
    # plt.colorbar()

    x_srp = srp_from_entry(entry_state, rtg, cr, target_alt=target.altitude)
    # for x in x_srp.T:
    #     plt.figure()
    #     plt.hist(x)
    srpdata = pickle.load(open(os.path.join(os.getcwd(), "data\\FuelOptimal\\srp_27k_5d.pkl"),'rb'))
    m_srp = srpdata(x_srp)
    # print(x_srp.shape)
    # print(m_srp.shape)

    plt.figure(figsize=(12, 10))
    plt.tricontourf(entry_state[1]*3397, -entry_state[2]*3397, m_srp, cmap='inferno')
    plt.plot(0, 0, 'bo', label="Target")
    plt.colorbar(label="Propellant (kg)")
    plt.xlabel("Range to go (km)")
    plt.ylabel("Cross Track (km)")
    plt.legend(loc="upper left")
    # plt.axis('equal')

    N = 5000
    entry_state = boxgrid([(r,r), (-d*1.5, -0.15/3397), (0,0), (v,v), (fpa,fpa), azi, (m0, m0)], [1,N,1,1,1,1,1], interior=True).T

    psi = np.arctan2(-entry_state[2]*3397, -entry_state[1]*3397)
    entry_state[5] = psi
    rtg, cr = range_from_entry(entry_state, target.coordinates())

    x_srp = srp_from_entry(entry_state, rtg, cr, target_alt=target.altitude)

    m_srp = srpdata(x_srp)

    plt.figure(figsize=(12, 10))
    plt.plot(entry_state[1]*3397, m_srp)
    plt.xlabel("Range to target (km)")
    plt.ylabel("Propellant (kg)")
    # plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    from EntryGuidance.SRPData import SRPData # Unpickling in plot_contours fails without this 
    test()
    plot_contours()