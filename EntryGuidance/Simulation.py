import os, sys, inspect
import numpy as np
from scipy.integrate import odeint, trapz
from scipy import linalg
from scipy.interpolate import interp1d
from scipy.io import savemat, loadmat



# cmd_folder =    os.path.realpath(
                # os.path.dirname(
                # os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0])))

# if cmd_folder not in sys.path:
    # sys.path.insert(0, cmd_folder)

# from Utils.redirect import stdout_redirected
    
import logging
from transitions import Machine, State, logger
from EntryGuidance.EntryEquations import Entry
from EntryGuidance.Planet import Planet
from EntryGuidance.EntryVehicle import EntryVehicle

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
    
    def __init__(self, states, conditions, cycle = None):
        # logger.setLevel(logging.INFO)

        if len(states) != len(conditions):
            raise ValueError("Number of states must equal number of conditions.")
            
        if cycle is None:
            cycle = Cycle()
        
        self.__conditions = conditions
        self.__states = states
        self.cycle = cycle          # The guidance cycle governing the simulation. Data is logged ever cycle.duration seconds while trigger checking and control updates occur 10x per cycle.
        self.time = 0.0             # Current simulation time
        self.times = [0.0]          # Collection of times at which the state history is logged
        self.index = 0              # The index of the current phase
        self.sample = None          # Uncertainty sample to be run
        self.x = None               # Current state vector
        self.history = []           # Collection of state sectors
        self.ie = []                # Indices of event transitions
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
            temp = self.__step() #Advance the numerical simulation, save resulting states for next check etc


        return True
    
    def __step(self):
        if self.edlModel.powered:
            throttle, mu = self.control[self.index](**self.triggerInput) #Pass stuff to the controllers here!
            sigma = 0
        else:
            sigma = self.control[self.index](**self.triggerInput)
            throttle = 0
            mu = 0
            
            
        X = odeint(self.edlModel.dynamics((sigma,throttle,mu)), self.x, np.linspace(self.time,self.time+self.cycle.duration,10))
        #find nearest endpoint here
        self.update(X,self.cycle.duration)
        # return X
    
    def run(self, InitialState, Controllers, InputSample=None):
    
        self.reset()
        
        if InputSample is None:
            InputSample = np.zeros(4)
        CD,CL,rho0,sh = InputSample
        
        self.sample = InputSample
        self.edlModel = Entry(PlanetModel = Planet(rho0=rho0,scaleHeight=sh), VehicleModel = EntryVehicle(CD=CD,CL=CL))
        self.update(np.asarray(InitialState),0)
        self.control = Controllers
        while not self.is_Complete():
            temp = self.advance()
    
        self.history = np.vstack(self.history) # So that we can work with the data more easily than a list of arrays
        

       
    def update(self,x,dt):
        if len(x.shape) == 1:
            self.x = x
        else:
            self.x = x[-1,:]
        self.history.append(self.x)
        self.time += dt
        self.times.append(self.time)
        self.triggerInput = self.getDict()

        
    def printState(self):        
        
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
              }
        
        return d
    
    def ignite(self):
        self.edlModel.ignite()
        
    def plot(self):   
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(self.history[:,3], self.edlModel.altitude(self.history[:,0],km=True))
        for i in self.ie:
            plt.plot(self.history[i,3],self.edlModel.altitude(self.history[i,0],km=True),'o')
            
        plt.figure()
        plt.plot(self.history[:,1]*180/np.pi, self.history[:,2]*180/np.pi)
        for i in self.ie:
            plt.plot(self.history[i,1]*180/np.pi, self.history[i,2]*180/np.pi,'o')

        plt.show()
        
        
    # def analyze(self):
    
    def postProcess(self, dict=False):
        
        r,theta,phi = self.history[:,0], np.degrees(self.history[:,1]), np.degrees(self.history[:,2])
        v,gamma,psi = self.history[:,3], np.degrees(self.history[:,4]), np.degrees(self.history[:,5])
        s,m         = (self.history[0,6]-self.history[:,6])/1000, self.history[:,7]
        
        range = [self.edlModel.planet.range(*x0[[1,2,5]],lonc=np.radians(lon),latc=np.radians(lat),km=True) for lon,lat in zip(theta,phi)]
        energy = self.edlModel.energy(r,v)
        # eInterp = np.linspace(0,1,1000)
        # tInterp = interp1d(energy,time,'cubic')(eInterp)
        # xInterp = interp1d(energy,self.history[istart:idx,:],'cubic',axis=0)(eInterp)  
            
        h = [self.edlModel.altitude(R,km=True) for R in r]
        bank = [np.degrees(hep(xx,tt)) for xx,tt in zip(self.history,self.times)]
        L,D = self.edlModel.aeroforces(r,v)
        
        data = np.c_[time, energy, bank, h,   r,      theta,       phi,      v,         gamma, psi,       range,     L,      D]
        self.output = data
        return data
        
    def reset(self):
        print "Resetting simulation states.\n"
        self.set_state(self.__states[0])
        self.time = 0.0
        self.times = [0.0]
        self.index = 0
        self.sample = None
        self.x = None # Current State vector
        self.history = [] # Collection of State Vectors
        self.ie = []
        self.edlModel = None
        self.triggerInput = None
        self.control = None
        self.output = None
    # def save(self): #Create a .mat file
    
    


def SRP():
    from Triggers import AccelerationTrigger, VelocityTrigger, AltitudeTrigger, MassTrigger
    states = ['PreEntry','Entry','SRP']
    def combo(inputs):
        return (AltitudeTrigger(0.1)(inputs) or MassTrigger(1000)(inputs))
    combo.dump = MassTrigger(1000).dump    
    conditions = [AccelerationTrigger('drag',2), VelocityTrigger(500), combo]
    input = { 'states' : states,
              'conditions' : conditions }

    return input
              
def testSim():
    # from EntryGuidance.ParamtrizedPlanner import HEPBankReducedSmooth
    sim = Simulation(cycle=Cycle(1),**SRP())
    # f = lambda T: HEPBankReducedSmooth(T, 106,133)
    f = lambda **d: 0
    f2 = lambda **d: (1,3.04)
    c = [f,f,f2]
    r0, theta0, phi0, v0, gamma0, psi0,s0 = (3540.0e3, np.radians(-90.07), np.radians(-43.90),
                                             5505.0,   np.radians(-14.15), np.radians(4.99),   780e3)
    x0 = np.array([r0, theta0, phi0, v0, gamma0, psi0, s0, 2804.0])
    sim.run(x0,c)
    return sim
    
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