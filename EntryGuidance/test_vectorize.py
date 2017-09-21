import numpy as np 
import pyaudi as da 
from pyaudi import gdual_vdouble as gd 
import time 

N = 1000
applyUncertainty = 1 

def test_planet():
    from Planet import Planet 
    rho0 = 0.05 * np.random.randn(N) * applyUncertainty
    hs = 0.05 * np.random.randn(N) /3 * applyUncertainty
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
    from ParametrizedPlanner import profile
    from InitialState import InitialState
    from NMPC import NMPC 
    
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
    Q = np.array([[.1,0],[0,2]])
    nmpc = NMPC(fbl_ref=refs, Q=Q, Ef = reference_sim.df['energy'].values[-1], update_type=0,update_tol=2,debug=False)
    return nmpc 
    
def test():
    test_dynamics()

if __name__ == "__main__":
    # npc = generateController()
    test()
    