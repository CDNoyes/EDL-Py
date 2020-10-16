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


def sphere_to_cart(x):
    if len(x) == 7:
        r,th,ph,v,fpa,psi,m = x
    else:
        r,th,ph,v,fpa,psi,m = np.asarray(x).T


    ve,vn,vu = v*np.cos(fpa)*np.sin(psi), v*np.cos(fpa)*np.cos(psi), v*np.sin(fpa)  # ENU frame 

    vx,vy,vz = [vu*np.cos(ph)*np.cos(th) - vn*np.sin(ph)*np.cos(th) - ve*np.sin(th),
                vu*np.cos(ph)*np.sin(th) - vn*np.sin(ph)*np.sin(th) + ve*np.cos(th),
                vu*np.sin(ph) + vn*np.cos(ph)]
    print(np.linalg.norm([vx,vy,vz]))
    y = [r*np.cos(ph)*np.cos(th), r*np.cos(ph)*np.sin(th), r*np.sin(ph), vx, vy, vz ]
    return np.array(y).T

def srp_from_entry_new(x, target):
    # Target State is [target.altitude, target.longitude, target.latitude, 0, 0, 0]
    planet = Planet('Mars')
    xc = sphere_to_cart(x)
    
    tc = sphere_to_cart([planet.radius + target.altitude, target.longitude, target.latitude, 0, 0, 0, 8500])

    return tc-xc

def test():
    # Demonstrate that the conversion process is correct 

    m0 = 8500
    r = 3397e3 + 2e3
    v = 500
    fpa = np.radians(-15)
    d = 4/3397
    target = Target(0,0,0)

    for azi in np.radians([45, 0, -45]):
        entry_state = np.array([r, -d, -d, v, fpa, azi, m0], ndmin=2)
        # rtg, cr = range_from_entry(entry_state, target.coordinates())
        # print("\nRTG: {:.1f}, CR: {:.1f}".format(rtg[0]/1000, cr[0]/1000))
        # x_srp = srp_from_entry(entry_state, rtg, cr, target_alt=target.altitude)
        x_srp2 = srp_from_entry_new(entry_state, target)
        # print("\nRTG: {:.1f}, CR: {:.1f}".format(rtg[0]/1000, cr[0]/1000))
        # print(x_srp)
        print(x_srp2)
        print("\n")


def plot_contours():
    import pickle 
    import matplotlib.pyplot as plt 
    import os 
    from Planet import Planet 

    srpdata = pickle.load(open(os.path.join(os.getcwd(), "data\\FuelOptimal\\srp_21k_7200kg.pkl"), 'rb'))

    m0 = 7200
    r = 3397e3 + 5e3
    v = 630
    fpa = np.radians(-10)
    # need to vary lat/lon and azimuth 
    dmax = 25/3397
    dmin = 10/3397
    N = 10  # Use a low number for the quiver plots, ~50 for the contours 
    azi = (0,0) 
    entry_state = boxgrid([(r,r), (-dmax, -dmin), (-dmin,dmin), (v,v), (fpa,fpa), azi, (m0, m0)], [1,N,N,1,1,1,1], interior=True).T

    for head_align in [False, True]:


        # print(entry_state.shape)
        target = Target(0,0,0)
        rtg, cr = range_from_entry(entry_state, target.coordinates())
        print(np.min(rtg))

        if head_align:
            psi = np.arctan2(-cr, rtg)

            entry_state[5] = psi # use this to align heading 
            rtg, cr = range_from_entry(entry_state, target.coordinates())
        
        
        # plt.tricontourf(rtg, cr, np.degrees(entry_state[5]), 30, cmap='RdBu')
        # plt.colorbar()

        x_srp = srp_from_entry(entry_state, rtg, cr, target_alt=target.altitude)
        # for x in x_srp.T:
        #     plt.figure()
        #     plt.hist(x)

        m_srp = srpdata(x_srp)
        m_min = np.min(m_srp)
        keep = m_srp <= 2*m_min # set this to a high value to plot everything 

        print(entry_state.shape)
    #     print(x_srp.shape)
    #     print(m_srp.shape)

        if not head_align:
            plt.figure(figsize=(12, 10))
            plt.tricontourf(rtg[keep]/1000, cr[keep]/1000, m_srp[keep], cmap='inferno')

            plt.plot(0, 0, 'bo', label="Target")
            plt.colorbar(label="Propellant (kg)")
            plt.xlabel("Range to go (km)")
            plt.ylabel("Cross Track (km)")
            plt.legend(loc="upper left")
        
        plt.figure(figsize=(12, 10))
    #     plt.tricontourf(np.degrees(entry_state[1]), np.degrees(entry_state[2]), m_srp, cmap='inferno')
        plt.quiver(np.degrees(entry_state[1]), np.degrees(entry_state[2]), v*np.cos(fpa)*np.cos(entry_state[5]), v*np.cos(fpa)*np.sin(entry_state[5]), scale=5e3)
        plt.plot(0, 0, 'rx', label="Target")
    #     plt.colorbar(label="Propellant (kg)")
        plt.xlabel("Longitude (deg)")
        plt.ylabel("Latitude (deg)")
        plt.legend(loc="bottom right")    
        
    # plt.axis('equal')

    if head_align:
        plt.figure(figsize=(12, 10))
        # plt.plot(entry_state[1]*3397, m_srp)
        plt.plot(rtg/1000, m_srp, 'o')
        plt.xlabel("Range to target (km)")
        plt.ylabel("Propellant (kg)")
        # plt.axis('equal')

    plt.show()


def plot_quivers():
    import pickle 
    import matplotlib.pyplot as plt 
    import os 
    from Planet import Planet 

    savedir = os.path.join(os.getcwd(), "Documents\\FuelOptimal\Subimages")
    srpdata = pickle.load(open(os.path.join(os.getcwd(), "data\\FuelOptimal\\srp_21k_7200kg.pkl"), 'rb'))

    m0 = 7200
    r = 3397e3 + 5e3
    v = 630
    fpa = np.radians(-10)
    # need to vary lat/lon and azimuth 
    dmax = 25/3397
    dmin = 15/3397
    N = 5  # Use a low number for the quiver plots, ~50 for the contours 
    azi = (0,0) 
    entry_state = boxgrid([(r,r), (-dmax, -dmin), (-dmin,dmin), (v,v), (fpa,fpa), azi, (m0, m0)], [1,N,5,1,1,1,1], interior=True).T

    fs = (6, 6)
    font = {'fontsize':16}
    ms = 10
    # plt.subplot(1, 2, 1)


    # print(entry_state.shape)
    target = Target(0,0,0)
    rtg, cr = range_from_entry(entry_state, target.coordinates())
    # print(np.min(rtg))

    psi = np.arctan2(-cr, rtg)
    # psi = Planet().heading(entry_state[1], entry_state[2], 0, 0)

    # x_srp = srp_from_entry(entry_state, rtg, cr, target_alt=target.altitude)

    for heading in [0, psi]:

        if heading is 0:
            text = ''
        else:
            text = '_aligned'

        entry_state[5] = heading
        rtg, cr = range_from_entry(entry_state, target.coordinates())
        if heading is 0:
            d = np.sort(np.random.random(rtg.shape))
        else:
            d = np.linalg.norm([rtg, cr], axis=0)

        plt.figure(figsize=fs)
        if heading is 0:
            plt.scatter(np.degrees(entry_state[1]), np.degrees(entry_state[2]), label='Spacecraft Position')
        else:
            plt.scatter(np.degrees(entry_state[1]), np.degrees(entry_state[2]), label='Spacecraft Position', c=d)
        plt.quiver(np.degrees(entry_state[1]), np.degrees(entry_state[2]), v*np.cos(fpa)*np.cos(heading), v*np.cos(fpa)*np.sin(heading), scale=9e3, label='Velocity')
        plt.plot(0, 0, 'rx', label="Target", markersize=ms)
        plt.xlabel("Longitude (deg)", **font)
        plt.ylabel("Latitude (deg)", **font)
        if heading is 0 or True:
            # plt.legend(loc=(0.7, 0.75))    
            plt.legend(loc='best')    
        ax = plt.gca()
        ax.set_aspect(1/ax.get_data_ratio(), adjustable='box')
        # plt.axis('square')
        plt.savefig(os.path.join(savedir, 'LatLon'+text), bbox_inches='tight')

        plt.figure(figsize=fs)
        # plt.subplot(1, 2, 2)
        if heading is 0:
            plt.scatter(rtg/1000, cr/1000)
            plt.quiver(rtg/1000, cr/1000, v*np.cos(fpa), 0, scale=9e3)
        else:
            plt.scatter(rtg/1000, 0*cr, c=d)
            # plt.quiver(rtg/1000, 0*cr, v*np.cos(fpa), 0, scale=5e3)

        plt.xlabel("Downrange to Target (km)", **font)
        plt.ylabel("Crossrange to Target (km)", **font)
        plt.gca().set_xlim(np.max(rtg/1000)*1.05, -500/1000)
        plt.gca().set_ylim(-17, 17)
        plt.plot(0, 0, 'rx', label="Target", markersize=ms)
        ax = plt.gca()
        ax.set_aspect(1/ax.get_data_ratio(), adjustable='box')
        # plt.axis('square')


        # plt.legend(loc="lower right")    
        plt.savefig(os.path.join(savedir, 'DRCR'+text), bbox_inches='tight')

    plt.show()


if __name__ == "__main__":

    from EntryGuidance.SRPData import SRPData # Unpickling in plot_contours fails without this 
    # test()
    # plot_contours()
    plot_quivers()