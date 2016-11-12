import os, sys, inspect
import numpy as np
from scipy.integrate import odeint, trapz
from scipy import linalg
from scipy.interpolate import interp1d
from scipy.io import savemat, loadmat
    
import logging
from transitions import Machine, State, logger
from EntryEquations import Entry
from Planet import Planet
from EntryVehicle import EntryVehicle

# Graphing specific imports
from transitions.extensions import GraphMachine as MGraph
import webbrowser

   
class Cycle(object): #Guidance Cycle
    def __init__(self, duration=1, freq=None):
        if freq is None:
            self.duration = duration
            self.rate = 1./duration
        else:
            self.duration = 1./freq
            self.rate = freq
               
    
class Simulation(Machine):   
    '''
        Defines a simulation class. The class is initialized to create its finite-state machine. 
        
        
    '''
    
    def __init__(self, states, conditions, cycle=None, output=True):

        if len(states) != len(conditions):
            raise ValueError("Number of states must equal number of conditions.")
            
        if cycle is None:
            if output:
                print "Simulation using default guidance cycle."
            cycle = Cycle()
        
        self.__conditions = conditions
        self.__states = states
        self.__output = output
        
        self.cycle = cycle          # The guidance cycle governing the simulation. Data logging and control updates occur every cycle.duration seconds while trigger checking happens 10x per cycle
        self.time = 0.0             # Current simulation time
        self.times = []             # Collection of times at which the state history is logged
        self.index = 0              # The index of the current phase
        self.sample = None          # Uncertainty sample to be run
        self.x = None               # Current state vector
        self.history = []           # Collection of state vectors
        self.u = None               # Previous controls
        self.control_history = []   # Collection of controls
        self.ie = [0]                # Indices of event transitions
        self.edlModel = None        # The dynamics and other functions associated with EDL
        self.triggerInput = None    # An input to triggers and controllers
        
        states.append('Complete')
        transitions = [{'trigger':'advance', 'source':states[i-1], 'dest':states[i], 'conditions':'integrate'} for i in range(1,len(states))]
        try:
            iSRP = states.index('SRP')
            transitions[iSRP-1]['after'] = 'ignite'
        except:
            pass
        Machine.__init__(self, model=None, states=states, initial=states[0], transitions=transitions, auto_transitions=False, after_state_change='printState')

        
    # def dump(self):

    # def __call__(self):
    
    def integrate(self):
    
        while not self.__conditions[self.index](self.triggerInput):
            if self.__output and not len(self.history)%10:
                print "current simulation time = {} s".format(self.time) # Should define a pretty print function and call that here
            temp = self.__step() #Advance the numerical simulation, save resulting states for next check etc

        return True
    
    
    def __step(self):
        if self.edlModel.powered:
            throttle, mu = self.control[self.index](**self.triggerInput)
            sigma = 0.
        else:
            sigma = self.control[self.index](**self.triggerInput)
            throttle = 0.
            mu = 0.
            
            
        X = odeint(self.edlModel.dynamics((sigma,throttle,mu)), self.x, np.linspace(self.time,self.time+self.cycle.duration,10))
        #find nearest endpoint here
        self.update(X,self.cycle.duration,np.asarray([sigma,throttle,mu]))

        
    def run(self, InitialState, Controllers, InputSample=None, AeroRatios=(1,1)):
        """ Runs the simulation from a given a initial state, with the specified controllers in each state, and using a chosen sample of the uncertainty space """
        
        self.reset()
        
        if InputSample is None:
            InputSample = np.zeros(4)
        CD,CL,rho0,sh = InputSample
        
        self.sample = InputSample
        self.edlModel = Entry(PlanetModel = Planet(rho0=rho0,scaleHeight=sh), VehicleModel = EntryVehicle(CD=CD,CL=CL))
        self.edlModel.update_ratios(*AeroRatios)
        self.update(np.asarray(InitialState),0.0,None)
        self.control = Controllers
        while not self.is_Complete():
            temp = self.advance()
    
        self.history = np.vstack(self.history)  # So that we can work with the data more easily than a list of arrays
        self.control_history.append(self.u)     # So that the control history has the same length as the data;
        self.control_history = np.vstack(self.control_history) 
        
        return self.postProcess()

       
    def update(self,x,dt,u):
        if len(x.shape) == 1:
            self.x = x
        else:
            self.x = x[-1,:]
        
        if u is not None:
            self.u = u    
            self.control_history.append(self.u)
            
        self.history.append(self.x)
        self.time += dt
        self.times.append(self.time)
        self.triggerInput = self.getDict()

        
    def printState(self):        
        
        if self.__output:
            print('Transitioning from state {} to {} because the following condition was met:'.format(self.__states[self.index],self.state))
            print(self.__conditions[self.index].dump())
            # print('t = {}: {}\n'.format(self.time,self.x))
            for key,value in self.triggerInput.items():
                print '{} : {}\n'.format(key,value)
        self.index += 1
        self.ie.append(len(self.history)-1)
    
    
    def getDict(self):
        L,D = self.edlModel.aeroforces(np.array([self.x[0]]),np.array([self.x[3]]))

        d =  {
              'time'     : self.time,
              'altitude' : self.edlModel.altitude(self.x[0]),
              'longitude': self.x[1],
              'latitude' : self.x[2],
              'velocity' : self.x[3],
              'fpa'      : self.x[4],
              'mass'     : self.x[7],
              'rangeToGo': self.x[6],
              'drag'     : D[0],
              'lift'     : L[0],
              'vehicle'  : self.edlModel.vehicle,
              'current_state' : self.x 
              }
        
        return d
    
    def ignite(self):
        self.edlModel.ignite()
        
    def plot(self, plotEvents = True):   
        import matplotlib.pyplot as plt
        
        # To do: replace calls to self.history etc with data that can be passed in; If data=None, data = self.postProcess()
        
        # Altitude vs Velocity
        plt.figure(1)
        plt.plot(self.history[:,3], self.edlModel.altitude(self.history[:,0],km=True), lw = 3)
        if plotEvents:
            for i in self.ie:
                plt.plot(self.history[i,3],self.edlModel.altitude(self.history[i,0],km=True),'o',label = self.__states[self.ie.index(i)], markersize=12)
        plt.legend(loc='upper left')   
        plt.xlabel('Velocity (m/s)')
        plt.ylabel('Altitude (km)')
        
        # #Latitude/Longitude
        # plt.figure(2)
        # plt.plot(self.history[:,1]*180/np.pi, self.history[:,2]*180/np.pi)
        # if plotEvents:        
            # for i in self.ie:
                # plt.plot(self.history[i,1]*180/np.pi, self.history[i,2]*180/np.pi,'o',label = self.__states[self.ie.index(i)])
        # # plt.legend()
        
        # Range vs Velocity
        plt.figure(3)
        plt.plot(self.history[:,3], self.history[:,6]/1000)
        if plotEvents:
            for i in self.ie:
                plt.plot(self.history[i,3],self.history[i,6]/1000,'o',label = self.__states[self.ie.index(i)])
        # plt.legend(loc='upper left')   
        plt.xlabel('Velocity (m/s)')
        plt.ylabel('Range to Target (km)')
        
        # Bank Angle Profile
        plt.figure(4)
        plt.plot(self.times, np.degrees(self.control_history[:,0]))
        for i in self.ie:
            plt.plot(self.times[i], np.degrees(self.control_history[i,0]),'o',label = self.__states[self.ie.index(i)])
        plt.legend(loc='best')   
        plt.xlabel('Time (s)')
        plt.ylabel('Bank Angle (deg)')
        
        # Downrange vs Crossrange
        plt.figure(5)
        plt.plot(self.output[:,11], self.output[:,10])
        for i in self.ie:
            plt.plot(self.output[i,11], self.output[i,10],'o',label = self.__states[self.ie.index(i)])
        plt.legend(loc='best')   
        plt.xlabel('Cross Range (km)')
        plt.ylabel('Down Range (km)')
        
        # Flight path vs Velocity
        plt.figure(6)
        plt.plot(self.history[:,3], self.history[:,4]*180/np.pi)
        if plotEvents:
            for i in self.ie:
                plt.plot(self.history[i,3],self.history[i,4]*180/np.pi,'o',label = self.__states[self.ie.index(i)])
        # plt.legend(loc='upper left')   
        plt.xlabel('Velocity (m/s)')
        plt.ylabel('Flight path angle (deg)')
        
        
        
    def show(self):
        import matplotlib.pyplot as plt
        plt.show()
        
    # def analyze(self):
    
    def postProcess(self, dict=False):

        bank = np.degrees(self.control_history[:,0])

        r,theta,phi = self.history[:,0], np.degrees(self.history[:,1]), np.degrees(self.history[:,2])
        v,gamma,psi = self.history[:,3], np.degrees(self.history[:,4]), np.degrees(self.history[:,5])
        s,m         = (self.history[0,6]-self.history[:,6])/1000, self.history[:,7]
        
        x0 = self.history[0,:]
        range = [self.edlModel.planet.range(*x0[[1,2,5]],lonc=np.radians(lon),latc=np.radians(lat),km=True) for lon,lat in zip(theta,phi)]
        energy = self.edlModel.energy(r,v)
        # eInterp = np.linspace(0,1,1000)
        # tInterp = interp1d(energy,time,'cubic')(eInterp)
        # xInterp = interp1d(energy,self.history[istart:idx,:],'cubic',axis=0)(eInterp)  
            
        h = [self.edlModel.altitude(R,km=True) for R in r]
        L,D = self.edlModel.aeroforces(r,v)

        data = np.c_[self.times, energy, bank, h,   r,      theta,       phi,      v,         gamma, psi,       range,     L,      D]
        self.output = data
        return data
        
    def reset(self):
        if self.__output:
            print "Resetting simulation states.\n"
        self.set_state(self.__states[0])
        self.time = 0.0
        self.times = []
        self.index = 0
        self.sample = None          # Input uncertainty sample
        self.x = None               # Current State vector
        self.history = []           # Collection of State Vectors
        self.u = None
        self.control_history = []   # Collection of State Vectors
        self.ie = [0]
        self.edlModel = None
        self.triggerInput = None
        self.control = None
        self.output = None
        
        
    def getRef(self):
        """ Computes a reference object for use in tracking based guidance
            There are many options for what this could be, and which variables to include.
            2d array, interp object, functional fit object?
            drag vs time, energy, velocity?
            range?
        
        """
        
        drag = np.flipud(self.output[:,13])
        vel = np.flipud(self.output[:,7])
        i_vmax = np.argmax(vel)
        
        return interp1d(vel[:i_vmax],drag[:i_vmax], fill_value=(drag[0],drag[i_vmax]), assume_sorted=True, bounds_error=False)
        
        
    # def save(self): #Create a .mat file
    
    


def SRP():
    """ Defines states and conditions for a trajectory from Pre-Entry through SRP-based EDL """
    from Triggers import AccelerationTrigger, VelocityTrigger, AltitudeTrigger, MassTrigger
    states = ['PreEntry','Entry','SRP']
    def combo(inputs):
        return (AltitudeTrigger(2)(inputs) or MassTrigger(6400)(inputs))
    combo.dump = AltitudeTrigger(2).dump    
    conditions = [AccelerationTrigger('drag',2), VelocityTrigger(700), VelocityTrigger(50)]
    input = { 'states' : states,
              'conditions' : conditions }

    return input

def EntrySim():
    ''' Defines conditions for a simple one phase guided entry '''
    from Triggers import VelocityTrigger
    states = ['Entry']
    trigger = [VelocityTrigger(500)]
    return {'states':states, 'conditions':trigger}
    
def testSim():

    sim = Simulation(cycle=Cycle(1),**SRP())
    f = lambda **d: 0
    f2 = lambda **d: (1,2.88)
    c = [f,f,f2]
    r0, theta0, phi0, v0, gamma0, psi0,s0 = (3540.0e3, np.radians(-90.07), np.radians(-43.90),
                                             5505.0,   np.radians(-14.15), np.radians(4.99),   1180e3)
    x0 = np.array([r0, theta0, phi0, v0, gamma0, psi0, s0, 8500.0])
    sim.run(x0,c)
    return sim
 

def NMPCSim(options):
    from Triggers import TimeTrigger
    states = ['State{}'.format(i) for i in range(0,options['N'])]
    times = np.linspace(0,options['T'],options['N']+1)
    triggers = [TimeTrigger(t) for t in times[1:]]
    
    return {'states':states, 'conditions':triggers}

def testNMPCSim():
    from functools import partial
    
    sim = Simulation(**NMPCSim({'N': 3, 'T' : 120}))
    
    vals = [0,1.5,0.25]
    c = [partial(constant, value=v) for v in vals]
    
    r0, theta0, phi0, v0, gamma0, psi0,s0 = (3540.0e3, np.radians(-90.07), np.radians(-43.90),
                                             5505.0,   np.radians(-14.15), np.radians(4.99),   1180e3)
    x0 = np.array([r0, theta0, phi0, v0, gamma0, psi0, s0, 8500.0])
    sim.run(x0,c)
    return sim
    
def constant(value, **kwargs):
    return value
    
    
# #########################
# Visualization Extension #           
# #########################
            
class SimulationGraph(MGraph,Machine):

    '''
    Defines a simulation graph class for viewing the simulation's core finite state-machine. 
    '''
    
    def __init__(self, *args, **kwargs):
        self.nDrawn = 0
        super(SimulationGraph, self).__init__(*args, **kwargs)

    def show(self):
        self.graph.draw('SimulationFSM-{}.png'.format(self.nDrawn), prog='dot')
        webbrowser.open('SimulationFSM-{}.png'.format(self.nDrawn))
        self.nDrawn += 1

        
            
def getSim():
    states = ['Pre-Entry','Entry','Heading\nAlignment','SRP']                 # SRP-enabled mission
    transitions = [{'trigger' : 'begin_entry', 'source' : 'Pre-Entry', 'dest' : 'Entry'},# 'conditions' : 'sufficientDrag'},
                   {'trigger' : 'align', 'source' : 'Entry', 'dest' : 'Heading\nAlignment'},# 'conditions' : 'Mach_less_than_M_align'},
                   {'trigger' : 'ignite', 'source' : 'Heading\nAlignment', 'dest' : 'SRP'}]
                    
    
    sim = SimulationGraph( model = None,
                           states = states,
                           transitions = transitions,
                           auto_transitions = False,
                           initial = states[0],
                           title = 'EDL Simulation',
                           show_conditions = True)
                           
    return sim
                           
                           
def fsmGif(states = range(4) ):
    '''
        Creates a GIF out of individual images, named after their respective states.
        
        Inputs:
            states : List of states
        Outputs:
            .gif file
    
    '''
    from images2gif import writeGif
    from PIL import Image

    files = ['SimulationFSM-{}.png'.format(state) for state in states]                  
    images = [Image.open(file) for file in files]
    size = (750,750)
    for image in images:
        image.thumbnail(size,Image.ANTIALIAS)
    output = 'SimulationFSM.gif'
    writeGif(output,images,duration=1)
    
    
if __name__ == '__main__':
    from argparse import ArgumentParser
    import multiprocessing as mp
    import chaospy as cp
    import os
    from Simulation import Simulation, SRP
    from functools import partial
    from scipy.io import savemat, loadmat
    import JBG
    from ParametrizedPlanner import HEPBankReducedSmooth
    import matplotlib.pyplot as plt
    
    # Parse Arguments and Setup Pool Environment
    mp.freeze_support()
    # pool = mp.Pool(mp.cpu_count()/2.)
    pool = mp.Pool(4)     
        
    # Define Uncertainty Joint PDF
    CD          = cp.Uniform(-0.10, 0.10)   # CD
    CL          = cp.Uniform(-0.10, 0.10)   # CL
    rho0        = cp.Normal(0, 0.0333)      # rho0
    scaleHeight = cp.Uniform(-0.05,0.05)    # scaleheight
    pdf         = cp.J(CD,CL,rho0,scaleHeight)
    
    n = 30
    samples = pdf.sample(n)    
    p = pdf.pdf(samples)

    sim = Simulation(cycle=Cycle(1),**SRP())
    f = lambda **d: 0
    f1 = lambda **d: HEPBankReducedSmooth(d['time'], t1=30, t2=100)
    # f2 = lambda **d: (1,2.87)
    f2 = JBG.controller
    c = [f,f1,f2]
    r0, theta0, phi0, v0, gamma0, psi0,s0 = (3540.0e3, np.radians(-90.07), np.radians(-43.90),
                                             5505.0,   np.radians(-14.15), np.radians(4.99),   935e3)
    x0 = np.array([r0, theta0, phi0, v0, gamma0, psi0, s0, 2804.0])
    sim.run(x0,c)
    sim.plot()
    # mc = partial(sim.run, x0, c, output=False)
    
    # for s in samples.T:
        # sim.run(x0,c,s)
        # sim.plot()
    plt.show()

    # stateTensor = pool.map(mc,samples.T)
    # stateTensor = [mc(s) for s in samples.T]
    saveDir = './data/'
    # savemat(saveDir+'MC',{'states':stateTensor, 'samples':samples, 'pdf':p})