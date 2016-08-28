import os, sys, inspect
import numpy as np
from scipy.integrate import odeint, trapz
from scipy import linalg
from scipy.interpolate import interp1d
from scipy.io import savemat, loadmat

from Utils.redirect import stdout_redirected

# cmd_folder =    os.path.realpath(
                # os.path.dirname(
                # os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0])))

# if cmd_folder not in sys.path:
    # sys.path.insert(0, cmd_folder)

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
            self.duration = 1./rate
            self.rate = rate
               
    
class Simulation(Machine):   
    '''
        Defines a simulation class.
        
        
    '''
    
    def __init__(self, states, conditions, cycle = None):
        # logger.setLevel(logging.INFO)

        if len(states) != len(conditions):
            raise ValueError("Number of states must equal number of conditions.")
            
        if cycle is None:
            cycle = Cycle()
        
        self.__conditions = conditions
        self.cycle = cycle
        self.time = 0.0
        self.index = 0
        self.sample = None
        self.x = None # Current State vector
        self.history = [] # Collection of State Vectors
        self.ie = []
        self.edlModel = None
        self.triggerInput = None
        states.append('Complete')
        transitions = [{'trigger':'advance', 'source':states[i-1], 'dest':states[i], 'conditions':'integrate'} for i in range(1,len(states))]
        iSRP = states.index('SRP')
        transitions[iSRP-1]['after'] = 'ignite'
        Machine.__init__(self, model=None, states=states, initial=states[0], transitions=transitions, auto_transitions=False, after_state_change='printState')

        
    # def dump(self):

    # def __call__(self):
    
    def integrate(self):
    
        while not self.__conditions[self.index](self.triggerInput):
            _ = self.step() #Advance the numerical simulation, save resulting states for next check etc

        print('Transitioning from state {} because the following condition was met:'.format(self.state))
        print(self.__conditions[self.index].dump())
        print('{0} : {1}'.format(self.index,self.state))
        print('{}: {}'.format(self.time,self.x))
        self.index += 1
        self.ie.append(len(self.history))
        return True
    
    def step(self):
        # with stdout_redirected():    
        if self.edlModel.powered:
            throttle, mu = self.control[self.index]() #Pass stuff to the controllers here!
            sigma = 0
        else:
            sigma = self.control[self.index]()
            throttle = 0
            mu = 0
            
            
        X = odeint(self.edlModel.dynamics((sigma,throttle,mu)), self.x, np.linspace(self.time,self.time+self.cycle.duration,10))
        #check trigger here
        self.update(X,self.cycle.duration)
    
    def run(self, InitialState, Controllers, InputSample=None):
        if InputSample is None:
            InputSample = np.zeros(4)
        CD,CL,rho0,sh = InputSample
        
        self.sample = InputSample
        self.edlModel = Entry(PlanetModel = Planet(rho0=rho0,scaleHeight=sh), VehicleModel = EntryVehicle(CD=CD,CL=CL))
        self.update(np.asarray(InitialState),0)
        self.control = Controllers
        while not self.is_Complete():
            self.advance()
    
        self.history = np.array(self.history) # So that we can work with the data more easily
    # def plot(self):   
    
    # def analyze(self):
       
    def update(self,x,dt):
        if len(x.shape) == 1:
            self.x = x
        else:
            self.x = x[-1,:]
        self.history.append(x)
        self.time += dt
        self.triggerInput = self.getDict()

        
    def printState(self):
        print('Entered state {}'.format(self.state))
    
    
    def getDict(self):
        d =  {'altitude' : self.edlModel.altitude(self.x[0]),
              'velocity' : self.x[3],
              'drag'     : self.edlModel.aeroforces(np.array([self.x[0]]),np.array([self.x[3]]))[1]}
        
        return d
    
    def ignite(self):
        self.edlModel.ignite()
    # def postProcess(self):
    # def reset(self):
    
    # def save(self): #Create a .mat file
    
    


def SRP():
    from Triggers import AccelerationTrigger, VelocityTrigger, AltitudeTrigger 
    states = ['PreEntry','Entry','SRP']
    conditions = [AccelerationTrigger('drag',2), VelocityTrigger(500), AltitudeTrigger(0.1)]
    input = { 'states' : states,
              'conditions' : conditions }

    return input
              
def testSim():
    sim = Simulation(cycle=Cycle(0.1),**SRP())
    f = lambda : 0
    f2 = lambda : (1,1.5)
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