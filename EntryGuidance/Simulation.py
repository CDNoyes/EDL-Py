import os, sys, inspect
import numpy as np
from scipy.integrate import odeint, trapz
from scipy import linalg
from scipy.interpolate import interp1d
from scipy.io import savemat, loadmat
    
import logging
from transitions import Machine, State, logger
from EntryEquations import Entry, System
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
        
        Methods:
            run    - Runs the current simulation from a given state acting under a series of controllers and a realization of the uncertainty space.
            getRef - Returns a dictionary of interpolation objects
            plot   - Plots a set of standard graphs. Does not show them, use Simulation.show() to bring them up. This can be useful to plot multiple trajectories before calling show.
            
        Members:
        
    '''
    
    def __init__(self, states, conditions, cycle=None, output=True, find_transitions=True):

        if len(states) != len(conditions):
            raise ValueError("Number of fsm states must equal number of transition conditions.")
            
        if cycle is None:
            if output:
                print "Simulation using default guidance cycle."
            cycle = Cycle()
        
        self.__conditions = conditions
        self.__states = states
        self.__output = output
        self.__find_transitions = find_transitions
        
        self.cycle = cycle          # The guidance cycle governing the simulation. Data logging and control updates occur every cycle.duration seconds while trigger checking happens 10x per cycle
        self.time = 0.0             # Current simulation time
        self.times = []             # Collection of times at which the state history is logged
        self.index = 0              # The index of the current phase
        self.sample = None          # Uncertainty sample to be run
        self.x = None               # Current state vector
        self.history = []           # Collection of state vectors
        self.u = None               # Previous controls
        self.control_history = []   # Collection of controls
        self.ie = [0]               # Indices of event transitions
        self.edlModel = None        # The dynamics and other functions associated with EDL
        self.fullEDL = None         # The type of edl model used - "ideal" with perfect knowledge and no bank angle constraints, or "full" truth/nav/constraints/filters etc
        self.triggerInput = None    # An input to triggers and controllers
        
        states.append('Complete')
        transitions = [{'trigger':'advance', 'source':states[i-1], 'dest':states[i], 'conditions':'integrate'} for i in range(1,len(states))]
        try:
            iSRP = states.index('SRP')
            transitions[iSRP-1]['after'] = 'ignite'
        except:
            pass
        Machine.__init__(self, model=None, states=states, initial=states[0], transitions=transitions, auto_transitions=False, after_state_change='printState')

    
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
        self.update(X,self.cycle.duration,np.asarray([sigma,throttle,mu]))

        
    def run(self, InitialState, Controllers, InputSample=None, FullEDL=False, AeroRatios=(1,1)):
        """ Runs the simulation from a given a initial state, with the specified controllers in each phase, and using a chosen sample of the uncertainty space """
        
        self.reset()
        
        if InputSample is None:
            InputSample = np.zeros(4)
        CD,CL,rho0,sh = InputSample
        
        self.sample = InputSample
        self.fullEDL = FullEDL
        if self.fullEDL:
            self.edlModel = System(InputSample=InputSample)     # Need to eventually pass knowledge error here
        else:
            self.edlModel = Entry(PlanetModel=Planet(rho0=rho0, scaleHeight=sh), VehicleModel=EntryVehicle(CD=CD, CL=CL))
            self.edlModel.update_ratios(LR=AeroRatios[0],DR=AeroRatios[1])
        self.update(np.asarray(InitialState),0.0,np.asarray([0]*3))
        self.control = Controllers
        while not self.is_Complete():
            temp = self.advance()
    
        self.history = np.vstack(self.history)                  # So that we can work with the data more easily than a list of arrays
        self.control_history.append(self.u)                     # So that the control history has the same length as the data;
        self.control_history = np.vstack(self.control_history[1:]) 
        
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
        #find nearest endpoint here - the trigger was met in the last ten steps
        if self.__find_transitions:
            self.findTransition()
        if self.__output:
            print('Transitioning from state {} to {} because the following condition was met:'.format(self.__states[self.index],self.state))
            print(self.__conditions[self.index].dump())
            for key,value in self.triggerInput.items():
                print '{} : {}\n'.format(key,value)
        self.index += 1
        self.ie.append(len(self.history)-1)
    
    
    def getDict(self):
        if self.fullEDL:
            L,D = self.edlModel.nav.aeroforces(np.array([self.x[8]]),np.array([self.x[11]]),np.array([self.x[15]]))

            d =  {
                  'time'            : self.time,
                  'altitude'        : self.edlModel.nav.altitude(self.x[8]),
                  'longitude'       : self.x[9],
                  'latitude'        : self.x[10],
                  'velocity'        : self.x[11],
                  'fpa'             : self.x[12],
                  'heading'         : self.x[13],
                  'rangeToGo'       : self.x[14],
                  'mass'            : self.x[15],
                  'drag'            : D[0],
                  'lift'            : L[0],
                  'vehicle'         : self.edlModel.nav.vehicle,
                  'current_state'   : self.x[8:16], 
                  'aero_ratios'     : self.x[16:18],
                  'bank'            : self.u[0], # Should this be the current command or the current state?
                  'energy'          : self.edlModel.nav.energy(self.x[8],self.x[11],Normalized=False), # Estimated energy                 
                  }        
        else:
            L,D = self.edlModel.aeroforces(np.array([self.x[0]]),np.array([self.x[3]]),np.array([self.x[7]]))

            d =  {
                  'time'            : self.time,
                  'altitude'        : self.edlModel.altitude(self.x[0]),
                  'longitude'       : self.x[1],
                  'latitude'        : self.x[2],
                  'velocity'        : self.x[3],
                  'fpa'             : self.x[4],
                  'heading'         : self.x[5],
                  'rangeToGo'       : self.x[6],
                  'mass'            : self.x[7],
                  'drag'            : D[0],
                  'lift'            : L[0],
                  'vehicle'         : self.edlModel.vehicle,
                  'current_state'   : self.x,
                  'aero_ratios'     : (self.edlModel.lift_ratio, self.edlModel.drag_ratio),
                  'bank'            : self.u[0],
                  'energy'          : self.edlModel.energy(self.x[0],self.x[3],Normalized=False),
                  }
        
        return d
    
    def ignite(self):
        self.edlModel.ignite()
        
    def plot(self, plotEvents=True, compare=True):   
        import matplotlib.pyplot as plt
        
        # To do: replace calls to self.history etc with data that can be passed in; If data=None, data = self.postProcess()
        
        if self.fullEDL:
            fignum = simPlot(self.edlModel.truth, self.times, self.history[:,0:8], self.history[:,18], plotEvents, self.__states, self.ie, fignum=1)
            if compare:
                fignum = simPlot(self.edlModel.nav, self.times, self.history[:,8:16], self.control_history[:,0], plotEvents, self.__states, self.ie, fignum=1)             # Use same fignum for comparisons, set fignum > figures for new ones
            # else:
                # fignum = simPlot(self.edlModel.nav, self.times, self.history[:,8:16], self.control_history[:,0], plotEvents, self.__states, self.ie, fignum=fignum, label="Navigated ")
            plt.figure(fignum)        
            plt.plot(self.times, self.history[:,16],label='Lift')
            plt.plot(self.times, self.history[:,17], label='Drag')
            plt.title('Aerodynamic Filter Ratios')
       
        else:
            simPlot(self.edlModel, self.times, self.history, self.control_history[:,0], plotEvents, self.__states, self.ie, fignum=1)
        
        
    def show(self):
        import matplotlib.pyplot as plt
        plt.show()
        
    
    def postProcess(self):

        if self.fullEDL:
            bank_cmd = np.degrees(self.control_history[:,0])

            r,theta,phi = self.history[:,0], np.degrees(self.history[:,1]), np.degrees(self.history[:,2])
            v,gamma,psi = self.history[:,3], np.degrees(self.history[:,4]), np.degrees(self.history[:,5])
            s,m         = (self.history[0,6]-self.history[:,6])/1000, self.history[:,7]
            
            r_nav,theta_nav,phi_nav = self.history[:,8], np.degrees(self.history[:,9]), np.degrees(self.history[:,10])
            v_nav,gamma_nav,psi_nav = self.history[:,11], np.degrees(self.history[:,12]), np.degrees(self.history[:,13])
            s_nav, m_nav         = (self.history[0,14]-self.history[:,14])/1000, self.history[:,15]
            
            RL,RD = self.history[:,16], self.history[:,17]
            
            bank, bank_rate = np.degrees(self.history[:,18]), np.degrees(self.history[:,19])
            
            x0 = self.history[0,:]
            range = [self.edlModel.truth.planet.range(*x0[[1,2,5]],lonc=np.radians(lon),latc=np.radians(lat),km=True) for lon,lat in zip(theta,phi)]
            range_nav = [self.edlModel.nav.planet.range(*x0[[9,10,13]],lonc=np.radians(lon),latc=np.radians(lat),km=True) for lon,lat in zip(theta_nav,phi_nav)]
            
            energy = self.edlModel.truth.energy(r,v)
            energy_nav = self.edlModel.nav.energy(r_nav,v_nav)

                
            h = [self.edlModel.truth.altitude(R,km=True) for R in r]
            h_nav = [self.edlModel.nav.altitude(R,km=True) for R in r_nav]
            L,D = self.edlModel.truth.aeroforces(r,v,m)
            L_nav,D_nav = self.edlModel.nav.aeroforces(r_nav,v_nav,m_nav)
        
        
        
            data = np.c_[self.times, energy, bank_cmd, h,   r,      theta,       phi,      v,         gamma,     psi,       range,     L,      D,
                                     energy_nav, bank, h_nav, r_nav, theta_nav,  phi_nav,  v_nav,     gamma_nav, psi_nav,   range_nav, L_nav,  D_nav]
        else:
            bank_cmd = np.degrees(self.control_history[:,0])

            r,theta,phi = self.history[:,0], np.degrees(self.history[:,1]), np.degrees(self.history[:,2])
            v,gamma,psi = self.history[:,3], np.degrees(self.history[:,4]), np.degrees(self.history[:,5])
            s,m         = (self.history[0,6]-self.history[:,6])/1000, self.history[:,7]
            
            x0 = self.history[0,:]
            range = [self.edlModel.planet.range(*x0[[1,2,5]],lonc=np.radians(lon),latc=np.radians(lat),km=True) for lon,lat in zip(theta,phi)]
            energy = self.edlModel.energy(r,v)
                
            h = [self.edlModel.altitude(R,km=True) for R in r]
            L,D = self.edlModel.aeroforces(r,v,m)
            
            data = np.c_[self.times, energy, bank_cmd, h,   r,      theta,       phi,      v,         gamma, psi,       range,     L,      D]
            
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
        ref = {}
        
        vel = np.flipud(self.output[:,7]) # Flipped to be increasing for interp1d limitation
        alt = np.flipud(self.output[:,3]) 
        range = np.flipud(self.output[-1,10]*1e3-self.output[:,10]*1e3) # Should probably be range to go instead, since thats the state the sim has access to
        drag = np.flipud(self.output[:,13])
        drag_rate = np.flipud(np.diff(self.output[:,13])/np.diff(self.output[:,0]))
        dragcos = np.flipud(self.output[:,13]/np.cos(np.radians(self.output[:,8])))
        bank = np.flipud(self.output[:,2])
        
        i_vmax = np.argmax(vel)             # Only interpolate from the maximum downward so the reference is monotonic
        
        # Should probably use a loop or comprehension at this point...
        ref['drag'] = interp1d(vel[:i_vmax],drag[:i_vmax], fill_value=(drag[0],drag[i_vmax]), assume_sorted=True, bounds_error=False, kind='cubic')
        ref['drag_rate'] = interp1d(vel[:i_vmax],drag_rate[:i_vmax], fill_value=(drag_rate[0],drag_rate[i_vmax]), assume_sorted=True, bounds_error=False, kind='cubic')
        ref['altitude'] = interp1d(vel[:i_vmax],alt[:i_vmax], fill_value=(alt[0],alt[i_vmax]), assume_sorted=True, bounds_error=False, kind='cubic')
        ref['dragcos'] = interp1d(vel[:i_vmax],dragcos[:i_vmax], fill_value=(dragcos[0],dragcos[i_vmax]), assume_sorted=True, bounds_error=False, kind='cubic')
        ref['rangeToGo'] = interp1d(vel[:i_vmax],range[:i_vmax], fill_value=(range[0],range[i_vmax]), assume_sorted=True, bounds_error=False)
        ref['bank'] = interp1d(vel[:i_vmax],bank[:i_vmax], fill_value=(bank[0],bank[i_vmax]), assume_sorted=True, bounds_error=False, kind='nearest')
        return ref
        
    def findTransition(self):
        print "DEBUG> Finding transition point"
        n = len(self.times)

        for i in range(n-2,n-12,-1):
            self.time = self.times[i]
            self.x = self.history[i]
            self.u = self.control_history[i]
            self.triggerInput = self.getDict()
            if not self.__conditions[self.index](self.triggerInput): # Interpolate between i and i+1 states
                for j in np.linspace(0.05,0.95,20): # The number of points used here will determinte the accuracy of the final state
                    # Find a better state:
                    self.time = ((1-j)*self.times[i] + j*self.times[i+1])
                    self.x = ((1-j)*self.history[i] + j*self.history[i+1])
                    self.u = ((1-j)*self.control_history[i] + j*self.control_history[i+1])
                    self.triggerInput = self.getDict()
                    if self.__conditions[self.index](self.triggerInput):
                        break

                
                # Remove the extra states:
                self.history = self.history[0:i+1]
                self.times = self.times[0:i+1]
                self.control_history = self.control_history[0:i+1]

                # Update the final point
                self.history.append(self.x)
                self.control_history.append(self.u)
                self.times.append(self.time)

                return
        if self.__output:        
            print "No better endpoint found"
        return
 
    # def save(self): #Create a .mat file

def simPlot(edlModel, time, history, control_history, plotEvents, fsm_states, ie, fignum=1, label=''):
    import matplotlib.pyplot as plt

    #history = [r, theta, phi, v, gamma, psi, s, m, DR, CR]
    # Altitude vs Velocity
    plt.figure(fignum)
    fignum += 1
    plt.plot(history[:,3], edlModel.altitude(history[:,0],km=True), lw = 3)
    if plotEvents:
        for i in ie:
            plt.plot(history[i,3],edlModel.altitude(history[i,0],km=True),'o',label = fsm_states[ie.index(i)], markersize=12)
    plt.legend(loc='upper left')   
    plt.xlabel(label+'Velocity (m/s)')
    plt.ylabel(label+'Altitude (km)')
    
    # #Latitude/Longitude
    plt.figure(fignum)
    fignum += 1
    plt.plot(history[:,1]*180/np.pi, history[:,2]*180/np.pi)
    if plotEvents:        
        for i in ie:
            plt.plot(history[i,1]*180/np.pi, history[i,2]*180/np.pi,'o',label = fsm_states[ie.index(i)])
    # plt.legend()
    
    # Range vs Velocity
    plt.figure(fignum)
    fignum += 1
    plt.plot(history[:,3], history[:,6]/1000)
    if plotEvents:
        for i in ie:
            plt.plot(history[i,3],history[i,6]/1000,'o',label = fsm_states[ie.index(i)])
    # plt.legend(loc='upper left')   
    plt.xlabel(label+'Velocity (m/s)')
    plt.ylabel(label+'Range to Target (km)')
    
    # Bank Angle Profile
    plt.figure(fignum)
    fignum += 1
    plt.plot(time, np.degrees(control_history[:]))
    for i in ie:
        plt.plot(time[i], np.degrees(control_history[i]),'o',label = fsm_states[ie.index(i)])
    plt.legend(loc='best')   
    plt.xlabel(label+'Time (s)')
    plt.ylabel(label+'Bank Angle (deg)')
    
    # plt.figure(fignum)
    # fignum += 1
    # plt.plot(history[:,3], np.degrees(control_history[:]))
    # for i in ie:
        # plt.plot(history[i,3], np.degrees(control_history[i]),'o',label = fsm_states[ie.index(i)])
    # plt.legend(loc='best')   
    # plt.xlabel(label+'Velocity (m/s)')
    # plt.ylabel(label+'Bank Angle (deg)')
    
    # Downrange vs Crossrange
    range = np.array([edlModel.planet.range(*history[0,[1,2,5]],lonc=lon,latc=lat,km=True) for lon,lat in zip(history[:,1],history[:,2])])
    plt.figure(fignum)
    fignum += 1        
    plt.plot(range[:,1], range[:,0])
    for i in ie:
        plt.plot(range[i,1], range[i,0],'o',label = fsm_states[ie.index(i)])
    plt.legend(loc='best')   
    plt.xlabel(label+'Cross Range (km)')
    plt.ylabel(label+'Down Range (km)')
    
    # Flight path vs Velocity
    plt.figure(fignum)
    fignum += 1        
    plt.plot(history[:,3], history[:,4]*180/np.pi)
    if plotEvents:
        for i in ie:
            plt.plot(history[i,3],history[i,4]*180/np.pi,'o',label = fsm_states[ie.index(i)])
    # plt.legend(loc='upper left')   
    plt.xlabel(label+'Velocity (m/s)')
    plt.ylabel(label+'Flight path angle (deg)')    
        
    return fignum+1
        
# ########################################################## #
# Simple functions to create various simulation combinations #
# ########################################################## #
    
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
    from Triggers import VelocityTrigger,AltitudeTrigger
    states = ['Entry']
    trigger = [VelocityTrigger(500)]
    # trigger = [AltitudeTrigger(5.1)]
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
 
def testFullSim():

    sim = Simulation(cycle=Cycle(1),output=True,**EntrySim())
    f = lambda **d: 0
    f2 = lambda **d: (1,2.88)
    c = [f,f,f2]
    r0, theta0, phi0, v0, gamma0, psi0,s0 = (3540.0e3, np.radians(-90.07), np.radians(-43.90),
                                             5505.0,   np.radians(-14.15), np.radians(4.99),   1180e3)
    x0 = np.array([r0, theta0, phi0, v0, gamma0, psi0, s0, 8500.0]*2 +[1,1] + [0,0])
    sim.run(x0,c, FullEDL=True)
    return sim

def NMPCSim(options):
    # Move this to MPC.py
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

    sim = testFullSim()
    sim.plot(compare=False)
    sim.show()
    # Monte Carlo Stuff:

    # from argparse import ArgumentParser
    # import multiprocessing as mp
    # import chaospy as cp
    # import os
    # from Simulation import Simulation, SRP, EntrySim
    # from functools import partial
    # from scipy.io import savemat, loadmat
    # import JBG
    # from ParametrizedPlanner import HEPBankReducedSmooth, HEPBank
    # from Uncertainty import getUncertainty
    # from Triggers import AccelerationTrigger,VelocityTrigger
    # import matplotlib.pyplot as plt
    # from MPC import controller,options
    # from numpy import pi

    # # Define Uncertainty Joint PDF
    # pdf = getUncertainty()['parametric']
    
    # n = 2000
    # samples = pdf.sample(n)    
    # p = pdf.pdf(samples)
    
    # reference_sim = Simulation(cycle=Cycle(1),output=False,**EntrySim())
    # bankProfile = lambda **d: HEPBank(d['time'],*[ 165.4159422 ,  308.86420218,  399.53393904])
    
    # r0, theta0, phi0, v0, gamma0, psi0,s0 = (3540.0e3, np.radians(-90.07), np.radians(-43.90),
                                             # 5505.0,   np.radians(-14.15), np.radians(4.99),   1000e3)
                                             
    # x0 = np.array([r0, theta0, phi0, v0, gamma0, psi0, s0, 8500.0])
    # output = reference_sim.run(x0,[bankProfile])

    # references = reference_sim.getRef()
    # drag_ref = references['drag']
    
    
    # # Create the simulation model:
        
    # states = ['PreEntry','Entry']
    # conditions = [AccelerationTrigger('drag',4), VelocityTrigger(500)]
    # input = { 'states' : states,
              # 'conditions' : conditions }
              
    # sim = Simulation(cycle=Cycle(1),output=False,**input)

    # # Create the controllers
    
    # option_dict = options(N=1,T=5)
    # mpc = partial(controller, control_options=option_dict, control_bounds=(0,pi/2), aero_ratios=(1,1), references=references)
    # pre = partial(constant, value=bankProfile(time=0))
    # controls = [pre,mpc]
    
    # # Run the off-nominal simulations
    # stateTensor = [sim.run(x0,controls,s) for s in samples.T]
    # saveDir = './data/'
    # savemat(saveDir+'MC',{'states':stateTensor, 'samples':samples, 'pdf':p})