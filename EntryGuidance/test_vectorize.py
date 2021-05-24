import numpy as np
import pyaudi as da
from pyaudi import gdual_vdouble as gd
import time

from EntryEquations import EDL
from InitialState import InitialState
from Uncertainty import getUncertainty
from Utils.RK4 import RK4
from ParametrizedPlanner import profile
from NMPC import NMPC
from Converge import Bootstrap

N = 250
applyUncertainty = 1

def test_planet():
    from Planet import Planet
    rho0 = 0.05 * np.random.randn(N) * applyUncertainty
    hs = 0.02 * np.random.randn(N) /3 * applyUncertainty
    mars = Planet(rho0=rho0,scaleHeight=hs)
    h = np.ones_like(hs)*70e3

    rho,Vs = mars.atmosphere(h)
    rho = rho.squeeze()
    if rho.shape == Vs.shape and rho.shape == hs.shape:
        print "Planet is vectorized"
    return mars

def test_vehicle():
    from EntryVehicle import EntryVehicle
    dCL = 0.05 * np.random.randn(N) * applyUncertainty
    dCD = 0.05 * np.random.randn(N) * applyUncertainty

    ev = EntryVehicle(CL=dCL,CD=dCD)
    M = np.ones_like(dCL)*5
    Cd,Cl = ev.aerodynamic_coefficients(M)
    print "Vehicle is vectorized"
    return ev

def test_dynamics():
    mars = test_planet()
    ev = test_vehicle()
    from EntryEquations import Entry
    from InitialState import InitialState
    from Utils.RK4 import RK4

    edl = Entry(PlanetModel=mars,VehicleModel=ev)

    # u = np.zeros((3,N)) # same length as the vectorized components
    x = InitialState()
    x = np.tile(x,(N,1)).T

    print "IC shape = {}".format(x.shape)
    X = [x]
    vprofile = vectorProfile()
    npc = generateController()
    t0 = time.time()
    for t in np.linspace(1,400,400):

        if 0:                   # Open Loop
           u = vprofile(t)

        else:
            Xc = X[-1]
            energy = edl.energy(Xc[0],Xc[3],False)
            lift,drag = edl.aeroforces(Xc[0],Xc[3],Xc[7])
            u = npc.controller(energy=energy, current_state=Xc,lift=lift,drag=drag,rangeToGo=None,planet=edl.planet)
            u.shape = (1,N)
            u = np.vstack((u,np.zeros((2,N))))

        eom = edl.dynamics(u)
        X.append(RK4(eom, X[-1], np.linspace(t,t+1,10),())[-1])
    tMC = time.time() - t0


    print "MC w/ vectorization of {} samples took {} s".format(N,tMC)
    X = np.array(X)
    Xf = X[-1]
    print Xf.shape
    X = np.transpose(X,(2,1,0))

    print X.shape

    # J = -(Xf[0]-3397e3)/1000 + Xf[3]/25#+ np.abs(Xf[2]*3397) # alt maximization
    # iopt = np.argmin(J)
    # print "Optimal switch = {}".format(np.linspace(40,340,N)[iopt])
    import matplotlib.pyplot as plt

    for Xi in X:
        plt.figure(1)
        plt.plot(Xi[1]*3397,Xi[2]*3397)

        plt.figure(2)
        plt.plot(Xi[3],(Xi[0]-3397e3)/1000)

    # X = np.transpose(X,(2,1,0))
    # plt.figure(1)
    # plt.plot(X[iopt][1]*3397,X[iopt][2]*3397,'k')
    # plt.figure(2)
    # plt.plot(X[iopt][3],(X[iopt][0]-3397e3)/1000,'k')

    plt.show()


def vectorProfile():
    from ParametrizedPlanner import profile

    bank = [1.4,0]
    switches = np.linspace(40,340,N)
    return lambda t: np.array([(profile(t,switch=[switch], bank=bank,order=0),0,0) for switch in switches]).T

def generateController():
    from Simulation import Simulation, Cycle, EntrySim
    from Triggers import SRPTrigger, AccelerationTrigger
    from InitialState import InitialState

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    # ######################################################
    # Reference data generation
    # ######################################################
    reference_sim = Simulation(cycle=Cycle(1),output=False,**EntrySim())
    banks = [-np.radians(30),np.radians(75),-np.radians(75),np.radians(30)]
    bankProfile = lambda **d: profile(d['time'],[62.30687581,  116.77385384,  165.94954234], banks, order=2)

    x0 = InitialState()
    output_ref = reference_sim.run(x0,[bankProfile],StepsPerCycle=10)
    refs = reference_sim.getFBL()

    # ######################################################
    # Closed-loop entry
    # ######################################################
    nmpc = NMPC(fbl_ref=refs, debug=False)
    return nmpc

def Optimize():
    """
        Full EDL Optimization including
        - Reference trajectory
            3 bank reversal switch times
            4 constant bank angle segments
        - Controller parameters
            1 Prediction horizon
            2 Gains

        for a total of 10 optimization parameters.

        Or reduced version with 2 banks, 3 angles + controller for 8 parameters.

    """
    from scipy.optimize import differential_evolution, minimize
    from numpy import pi, radians as rad
    from Simulation import Simulation, Cycle, EntrySim
    import Parachute

    sim = Simulation(cycle=Cycle(1),output=False,**EntrySim(Vf=460))
    perturb = getUncertainty()['parametric']
    optSize = 1000
    samples = perturb.sample(optSize,'S')
    edl = EDL(samples,Energy=True)

    heading_alignment = False
    # Differential Evolution Global Optimization
    bounds = [(50,140),(100,165)] + [(rad(70),rad(90)),(rad(70),rad(90)),(0,rad(30))] + [(0,10),(0,10),(0,10)] # general bounds
    # bounds = [(100,120),(135,160)] + [(rad(70),rad(90)),(rad(70),rad(90)),(0,rad(30))] + [(2,3),(0,10),(0,10)] # tightened bounds
    if heading_alignment:
        bounds.extend([(800,1400),(0.1,5)])
    # sol = differential_evolution(Cost,args = (sim,samples,optSize,edl,True,True,heading_alignment), bounds=bounds, tol=1e-2, disp=True, polish=False)
    # print "Optimized parameters (N={}) are:".format(optSize)
    # print sol.x
    # print sol

    if 0:
        # Particle Swarm Optimization
        import pyswarms as ps

        # Initialize swarm
        options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
        # higher c1 -> trajectories following their personal best
        # higher c2 -> trajectories follow the global best

        # Call instance of PSO with bounds argument
        bounds_pso = np.array(bounds).T
        bounds_pso = (bounds_pso[0],bounds_pso[1])
        # print bounds.shape
        pop_size = 200
        optimizer = ps.single.GlobalBestPSO(n_particles=pop_size, dimensions=len(bounds), options=options, bounds=bounds_pso)

        # hack to generate my own initial population
        import chaospy as cp
        U = [cp.Uniform(b[0],b[1]) for b in bounds]
        # replace with a dependent bound for t2:
        U[1] = cp.Uniform(lo=U[0], up=165)
        init_pop = cp.J(*U).sample(size=pop_size-1, rule="S").T
        sol = [  1.12985487e+02,   1.61527467e+02,   1.53352287e+00,   1.02508346e+00,
           4.79475355e-01,   2.47739391e+00,   1.14726959e-01,   6.88822448e+00]
        if heading_alignment:
            sol.extend([800,0.1])
        sol = np.array(sol,ndmin=2)
        init_pop = np.concatenate((init_pop,sol), axis=0)
        optimizer.pos = init_pop
        optimizer.personal_best_pos = init_pop

        # Perform optimization
        cost, sol = optimizer.optimize(lambda x: SwarmCost(x,sim,samples,optSize,edl,True,True,heading_alignment), print_step=1, iters=20, verbose=3)

    # pso sol
    sol = [ 112.98548700000001, 161.527467,
            1.5335228700000001, 1.0250834600000001, 0.47947535499999999,
            2.47739391, 0.114726959, 6.8882244799999999]


    # sol = [ 103.53150718,  127.2118855,    rad(84.95268405), rad(84.95268),   rad(11.97228525), 2.31450607,    4.48346113,    8.30596081]

    # sol =[  1.12985487e+02,   1.61527467e+02,   1.53352287e+00,   1.02508346e+00,
    #    4.79475355e-01,   2.47739391e+00,   1.14726959e-01,   6.88822448e+00]

    # Parachute.Draw(figure=2)
    Cost(sol, sim, samples, optSize, edl, True, True, heading_alignment)
    # new_sol = minimize(Cost, sol, args = (sim,samples,optSize,edl), tol=1e-2, method='Nelder-Mead')

    return

def SwarmCost(inputs, reference_sim, samples, optSize, edl, reduced, msl, align_heading):
    # print inputs.shape
    return np.array([Cost(inp, reference_sim, samples, optSize, edl, reduced=True, msl=msl, align_heading=align_heading) for inp in inputs])

def Cost(inputs, reference_sim, samples, optSize, edl, reduced=True, msl=False, align_heading=False):

    # Reference Trajectory
    if reduced:
        switches = inputs[0:2]
        banks = inputs[2:5]*np.array([1,-1,1])
    else:
        switches = inputs[0:3]
        banks = inputs[3:7]*np.array([-1,1,-1,1])

    if np.any(np.diff(switches) < 0) or np.any(inputs<0):
        return 2e5 # arbitrary high cost for bad switch times or negative gains, prediction horizon

    bankProfile = lambda **d: profile(d['time'], switch=switches, bank=banks,order=2)

    x = InitialState()
    output = reference_sim.run(x,[bankProfile])

    Xf = output[-1,:]
    hf = Xf[3]
    lonTarget = np.radians(Xf[5])
    # fpaf = Xf[8]
    dr = Xf[10]
    cr = Xf[11]
    high_crossrange = np.abs(cr) > 3
    low_altitude = hf <= 9

    if high_crossrange or low_altitude:
        return 300 + 500*np.abs(cr) - 25*hf # arbitrary high cost for bad reference trajectory
    # Otherwise, we have a suitable reference and we can run the QMC

    # Closed loop statistics generation
    refs = reference_sim.getFBL()

    nmpc = NMPC(fbl_ref=refs, debug=False)
    if reduced:
        nmpc.dt = inputs[5]
        nmpc.Q = np.array([[inputs[6],0],[0,inputs[7]]])
    else:
        nmpc.dt = inputs[7]
        nmpc.Q = np.array([[inputs[8],0],[0,inputs[9]]])

    if align_heading:
        V_lim = inputs[-2]
        K = inputs[-1]

    x = np.tile(x,(optSize,1)).T
    X = [x]
    energy0 = edl.energy(x[0],x[3],False)[0]
    energyf = Xf[1]*0.5

    energy = energy0
    E = [energy]
    temp = []
    while energy > energyf:

        Xc = X[-1]
        energys = edl.energy(Xc[0],Xc[3],False)
        lift,drag = edl.aeroforces(Xc[0],Xc[3],Xc[7])

        # Range control
        u = nmpc.controller(energy=energys, current_state=Xc, lift=lift, drag=drag, rangeToGo=None, planet=edl.planet)

        if align_heading:
            # Heading alignment
            rtg = lonTarget - Xc[1]
            crtg = -Xc[2]
            u_heading = np.clip(np.arctan2(crtg,rtg)*K,np.radians(-30),np.radians(30))

            heading_cases = np.where(Xc[3]<V_lim)[0]
            if heading_cases.shape[0]:
                u[heading_cases] = u_heading[heading_cases]

        # Shape the control
        u.shape = (1,optSize)
        u = np.vstack((u,np.zeros((2,optSize))))
        de = -np.mean(drag)*np.mean(Xc[3])
        if (energy + de) <  energyf:
            de = energyf - energy
        eom = edl.dynamics(u)
        X.append(RK4(eom, X[-1], np.linspace(energy,energy+de,10),())[-1])
        energy += de
        E.append(energy)
        if energy < Xf[1]:
            temp.append(energy)

        if len(E)>600:
            break
    X = np.array(X)
    # Xf = X[-1]
    if msl:
        Xf = np.array([Trigger(traj, lonTarget, minAlt=6e3, maxVel=485) for traj in X.transpose((2,0,1))]).T # Parachute deployment
    else:
        Xf = np.array([Trigger(traj,lonTarget) for traj in X.transpose((2,0,1))]).T

    Xf_energy = X[-len(temp)]
    # X = X.transpose((2,1,0))
    # print X.shape
    import matplotlib.pyplot as plt

    # Xi = Xf
    # # ######### for Xi in X:
    # plt.figure(1)
    # plt.hist2d(Xi[2]*3397,Xi[1]*3397,bins=30,cmap="binary")
    # plt.xlabel('Crossrange (km)')
    # plt.ylabel('Downrange (km)')
    # plt.colorbar()
    #
    # h   = edl.altitude(Xf[0], km=True) # altitude, km
    # import pdb
    # pdb.set_trace()
    for Xi in [Xf,Xf_energy]:
        h   = edl.altitude(Xi[0], km=True)

        plt.figure()
        # plt.scatter(Xi[2]*3397,Xi[1]*3397,c=h)
        plt.plot(Xi[2]*3397,Xi[1]*3397,'o')
        theta = np.linspace(0,2*np.pi,100)
        x = np.cos(theta)
        y = np.sin(theta)
        # for fig in [1,3]:
            # plt.figure(fig)
        for r in [1,2,8]:
            plt.plot(x*r,lonTarget*3397 + y*r,label="{} km".format(r))
        plt.legend()
        plt.xlabel('Crossrange (km)')
        plt.ylabel('Downrange (km)')
        # plt.colorbar()
        plt.axis('equal')


    # plt.figure(2)
    #
    # plt.plot(Xi[3],h,'o')
    plt.show()


    h   = edl.altitude(Xf[0], km=True) # altitude, km
    DR = Xf[1]*edl.planet.radius/1000 # downrange, km
    CR = Xf[2]*edl.planet.radius/1000 # -crossrange, km


    # Previous cost function
    # J = -np.percentile(h,1) +  0.1* (np.percentile(lon,99)-np.percentile(lon,1) + np.percentile(lat,99)-np.percentile(lat,1)) + np.abs(lat.mean())
    # Perhaps more theoretically sound?
    # J = (np.abs(DR-lonTarget*3397) + np.abs(CR)) #idea: altitude is handled by the trigger, i.e. too low altitude and many undershoots arise which raises the DR/CR errors
    Jnorm = np.sqrt((DR-lonTarget*3397)**2+CR**2) # Norm squared, to be differentiable for finite differencing
    J = Jnorm
    # print "{}% of {} samples landed with combined DR/CR (norm) error < 1 km".format(np.sum(Jnorm<=1)/float(J.size )*100,optSize)
    # print "{}% of {} samples landed with combined DR/CR (norm) error < 2 km".format(np.sum(Jnorm<=2)/float(J.size )*100,optSize)
    # print "{}% of {} samples landed with combined DR/CR (norm) error < 10 km".format(np.sum(Jnorm<=10)/float(J.size )*100,optSize)
    J = Bootstrap(np.mean, J, [J.size], resamples=20)[0][0] # Boostrapped estimate of the mean cost

    # plt.hist(J, bins=optSize/10, range=None, normed=False, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color=None, label=None, stacked=False, hold=None, data=None)
    # plt.show()

    print "input = {}\ncost = {}\n".format(inputs,J)

    return J


def Trigger(traj, targetLon, minAlt=0e3, maxVel=600):
    for state in traj:
        alt = state[0]-3397e3
        vel = state[3]
        longitude = state[1]
        if alt < minAlt or (vel<maxVel and longitude>=targetLon):
            return state
    return traj[-1]# No better trigger point so final point is used

def OptimizeController():
    from scipy.optimize import differential_evolution
    perturb = getUncertainty()['parametric']
    optSize = 500
    samples = perturb.sample(optSize,'S')
    nmpc = generateController()
    edl = EDL(samples,Energy=True)

    bounds = [(0,10),(0,10),(1,30)]
    # Cost([0.1,2,2.8],nmpc,samples,optSize,edl)
    sol = differential_evolution(CostController,args = (nmpc,samples,optSize,edl), bounds=bounds, tol=1e-2, disp=True, polish=False)
    print "Optimized parameters (N={}) are:".format(optSize)
    print sol.x
    print sol
    return

def CostController(inputs, nmpc, samples, optSize, edl):
    nmpc.dt = inputs[2]
    nmpc.Q = np.array([[inputs[0],0],[0,inputs[1]]])

    x = InitialState()
    x = np.tile(x,(optSize,1)).T
    X = [x]
    energy0 = edl.energy(x[0],x[3],False)[0]
    energyf = edl.energy(edl.planet.radius + 1.5e3, 500, False)

    energy = energy0
    E = [energy]
    while energy > energyf:

        Xc = X[-1]
        energys = edl.energy(Xc[0],Xc[3],False)
        lift,drag = edl.aeroforces(Xc[0],Xc[3],Xc[7])
        u = nmpc.controller(energy=energys, current_state=Xc,lift=lift,drag=drag,rangeToGo=None,planet=edl.planet)
        u.shape = (1,optSize)
        u = np.vstack((u,np.zeros((2,optSize))))
        de = -np.mean(drag)*np.mean(Xc[3])
        if (energy + de) <  energyf:
            # print "Final step"
            de = energyf - energy
        eom = edl.dynamics(u)
        X.append(RK4(eom, X[-1], np.linspace(energy,energy+de,10),())[-1])
        energy += de
        E.append(energy)
        # print "Finished integration step {}".format(len(E)-1)
        if len(E)>600:
            break
    X = np.array(X)
    # X = X.transpose((2,1,0))
    # print X.shape
    # import matplotlib.pyplot as plt
    #
    # for Xi in X:
    #     plt.figure(1)
    #     plt.plot(Xi[1]*3397,Xi[2]*3397,'o')
    #
    #     plt.figure(2)
    #     plt.plot(Xi[3],(Xi[0]-3397e3)/1000,'o')
    # plt.show()
    Xf = X[-1]
    h   = edl.altitude(Xf[0], km=True) # km
    lon = Xf[1]*edl.planet.radius/1000
    lat = Xf[2]*edl.planet.radius/1000

    J = -np.percentile(h,1) +  0.1* (np.percentile(lon,99)-np.percentile(lon,1) + np.percentile(lat,99)-np.percentile(lat,1))
    # print J
    # print np.percentile(h,1)
    # print np.percentile(lon,99)-np.percentile(lon,1)
    # print np.percentile(lat,99)-np.percentile(lat,1)
    return J

def test():
    test_dynamics()


def FD():
    from scipy.optimize import differential_evolution, minimize
    from numpy import pi, radians as rad
    from Simulation import Simulation, Cycle, EntrySim
    import matplotlib.pyplot as plt

    sim = Simulation(cycle=Cycle(1),output=False,**EntrySim(Vf=460))
    perturb = getUncertainty()['parametric']
    optSize = 500
    samples = perturb.sample(optSize,'S')
    edl = EDL(samples,Energy=True)

    # MSL, reduced input set
    # sol = [ 113.82395707,  170.07337194,              # Reversal times
    #         1.40719634,    0.97780072,    0.45524235, # Bank angles
    #         1.66167718,    0.20598009,    7.78547546] # Controller

    # Heavy, full input set
    sol = np.array([  26.06441256,  115.16979593,  167.14750033,
                      0.37717073, 1.494434, 1.06315079, 0.54208874,
                      2.31450607, 4.48346113, 8.30596081])
    I = np.eye(sol.size)
    # J0 = Cost(sol, sim, samples, optSize,edl,False)

    J = []
    deltas = np.array([1e-5,1e-4,1e-3,1e-2,0.1,1])
    labels = ['Switch 1','Switch 2','Switch 3','Bank 1','Bank 2','Bank 3','Bank 4','h','G1','G2']
    for delta in deltas:
        Ji = []
        for vector in I:
            delta_vector = delta*vector
            Jp = (Cost(sol+delta_vector, sim, samples, optSize,edl,False))
            Jn = (Cost(sol-delta_vector, sim, samples, optSize,edl,False))
            Ji.append((Jp-Jn)/(2*delta)) # Central difference
        J.append(Ji)
    J = np.array(J).T

    plt.figure(1)
    for label,ji in zip(labels,J):
        plt.semilogx(deltas, ji, label=label)
    plt.xlabel('$\Delta$Input used in central differencing')
    plt.ylabel('$\Delta$J$/\Delta Input}$')
    plt.title('MC Size = {}'.format(optSize))
    plt.legend()

    plt.figure(2)
    for label,ji in zip(labels[:3],J[:3]):
        plt.semilogx(deltas, ji, label=label)
    plt.xlabel('$\Delta$Input (s) used in central differencing')
    plt.ylabel('$\Delta$J$/\Delta Input}$')
    plt.title('Switch Times, MC Size = {}'.format(optSize))
    plt.legend()

    plt.figure(3)
    for label,ji in zip(labels[3:7],J[3:7]):
        plt.semilogx(deltas, ji, label=label)
    plt.xlabel('$\Delta$Input (rad) used in central differencing')
    plt.ylabel('$\Delta$J$/\Delta Input}$')
    plt.title('Bank Angles, MC Size = {}'.format(optSize))
    plt.legend()


    plt.show()

if __name__ == "__main__":
    # test()
    # OptimizeController()
    Optimize()
    # FD()
