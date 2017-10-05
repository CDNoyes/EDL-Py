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

    """
    from scipy.optimize import differential_evolution, minimize
    from numpy import pi, radians as rad
    from Simulation import Simulation, Cycle, EntrySim

    sim = Simulation(cycle=Cycle(1),output=False,**EntrySim())
    perturb = getUncertainty()['parametric']
    optSize = 500
    samples = perturb.sample(optSize,'S')
    edl = EDL(samples,Energy=True)

    bounds = [(0,120),(50,250),(150,300)] + [(0,rad(40)),(rad(50),rad(90)),(rad(50),rad(90)),(0,rad(40))] + [(0,30),(0,10),(0,10)]
    # sol = differential_evolution(Cost,args = (sim,samples,optSize,edl), bounds=bounds, tol=1e-2, disp=True, polish=False)
    # print "Optimized parameters (N={}) are:".format(optSize)
    # print sol.x
    # print sol

    sol =  np.array([  9.29183808e+00,   1.16993103e+02,   1.59447150e+02,
                       6.73319176e-01,   1.56897827e+00,   9.93751178e-01,
                       1.04070920e-03,   5.12963117e+00,   1.63277499e-01,
                       5.00426148e+00])
    Cost(sol, sim, samples, optSize,edl)
    # new_sol = minimize(Cost, sol, args = (sim,samples,optSize,edl), tol=1e-2, method='Nelder-Mead')

    return

def Cost(inputs, reference_sim, samples, optSize, edl):
    # Reference Trajectory
    switches = inputs[0:3]
    banks = inputs[3:7]*np.array([1,-1,1,-1])

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
    # Jref = -hf + (0*(dr_target-dr)**2 + 0*(cr_target-cr)**2)**0.5
    high_crossrange = np.abs(cr) > 3
    low_altitude = hf <= 0
    if high_crossrange or low_altitude:
        return 30 + 500*np.abs(cr) - 50*hf # arbitrary high cost for bad reference trajectory
    # Otherwise, we have a suitable reference and we can run the QMC

    # Closed loop statistics generation
    refs = reference_sim.getFBL()

    nmpc = NMPC(fbl_ref=refs, debug=False)
    nmpc.dt = inputs[7]
    nmpc.Q = np.array([[inputs[8],0],[0,inputs[9]]])

    x = np.tile(x,(optSize,1)).T
    X = [x]
    energy0 = edl.energy(x[0],x[3],False)[0]
    energyf = Xf[1]*0.95

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
    # Xf = X[-1]
    Xf = np.array([Trigger(traj,lonTarget) for traj in X.transpose((2,0,1))]).T
    # X = X.transpose((2,1,0))
    # print X.shape
    # import matplotlib.pyplot as plt
    # Xi = Xf
    # for Xi in X:
    # plt.figure(1)
    # plt.plot(Xi[1]*3397,Xi[2]*3397,'o')
    #
    # plt.figure(2)
    # plt.plot(Xi[3],(Xi[0]-3397e3)/1000,'o')
    # plt.show()


    h   = edl.altitude(Xf[0], km=True) # altitude, km
    DR = Xf[1]*edl.planet.radius/1000 # downrange, km
    CR = Xf[2]*edl.planet.radius/1000 # -crossrange, km


    # Previous cost function
    # J = -np.percentile(h,1) +  0.1* (np.percentile(lon,99)-np.percentile(lon,1) + np.percentile(lat,99)-np.percentile(lat,1)) + np.abs(lat.mean())
    # Perhaps more theoretically sound?
    w = 2
    J = -h + w*(np.abs(DR-lonTarget*3397) + np.abs(CR))
    # plt.hist(J, bins=optSize/10, range=None, normed=False, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color=None, label=None, stacked=False, hold=None, data=None)
    # plt.show()
    J = J.mean()
    # J = np.percentile(J,95) # maybe a percentile is better?
    print "input = {}\ncost = {}".format(inputs,J)

    # print J
    # print np.percentile(h,1)
    # print np.percentile(lon,99)-np.percentile(lon,1)
    # print np.percentile(lat,99)-np.percentile(lat,1)
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

if __name__ == "__main__":
    # test()
    # OptimizeController()
    Optimize()
