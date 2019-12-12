import sys
from os import path
# sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
sys.path.append("./")
sys.path.append("../")
from Utils.RK4 import RK4
from Utils import DA as da

import pandas as pd
import numpy as np
from scipy.integrate import odeint, trapz
from scipy import linalg
from scipy.interpolate import interp1d
from scipy.io import savemat, loadmat

import logging
from transitions import Machine, State
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

    def __init__(self, states, conditions, cycle=None, output=True, find_transitions=True, use_da=False, final_state="Complete"):

        if len(states) != len(conditions):
            raise ValueError("Number of fsm states must equal number of transition conditions.")

        if cycle is None:
            if output:
                print("Simulation using default guidance cycle.")
            cycle = Cycle()

        self._conditions = conditions
        self._states = states
        self._output = output
        self._find_transitions = find_transitions
        self._use_da = use_da

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
        self.simulations = 0        # The number of simulations run

        states.append(final_state)
        transitions = [{'trigger':'advance', 'source':states[i-1], 'dest':states[i], 'conditions':'integrate'} for i in range(1,len(states))]
        try:
            iPre = states.index('PreEntry')
            transitions[0]['after'] = 'bank_reversal'
        except:
            pass

        try:
            iSRP = states.index('SRP')
            if iSRP:
                transitions[iSRP-1]['after'] = 'ignite'
        except:
            pass

        Machine.__init__(self, model='self', states=states, initial=states[0], transitions=transitions, auto_transitions=False, after_state_change='printState')

    def set_output(self, boolean):
        self._output = boolean

    def integrate(self):

        while not self._conditions[self.index](da.const_dict(self.triggerInput)):
            if self._output and not (len(self.history)-1*self.cycle.rate)%int(10*self.cycle.rate):
                print("current simulation time = {} s".format(int(self.time))) # Should define a pretty print function and call that here
            temp = self.__step() #Advance the numerical simulation, save resulting states for next check etc

        return True

    def __step(self):
        if self.edlModel.powered:
            throttle, mu, zeta = self.control[self.index](**self.triggerInput)
            sigma = 0.
        else:
            sigma = self.control[self.index](**self.triggerInput)
            throttle = 0.   # fraction of max thrust
            mu = 0.         # pitch angle
            zeta = 0.       # yaw angle

        if self._use_da:
            X = RK4(self.edlModel.dynamics((sigma, throttle, mu)), self.x, np.linspace(self.time,self.time+self.cycle.duration,self.spc),())
        else:
            X = odeint(self.edlModel.dynamics((sigma, throttle, mu)), self.x, np.linspace(self.time,self.time+self.cycle.duration,self.spc))
        self.update(X, self.cycle.duration, np.asarray([sigma, throttle, mu]))


    def run(self, InitialState, Controllers, InputSample=None, FullEDL=False, AeroRatios=(1,1), StepsPerCycle=10):
        """ Runs the simulation from a given a initial state, with the specified controllers in each phase, and using a chosen sample of the uncertainty space """
        self.reset()
        self.spc = StepsPerCycle

        if InputSample is None:
            InputSample = np.zeros(4)
        CD,CL,rho0,sh = InputSample

        self.sample = InputSample
        self.fullEDL = FullEDL
        if self.fullEDL:
            self.edlModel = System(InputSample=InputSample)     # Need to eventually pass knowledge error here
            if self._output:
                print("L/D: {:.2f}".format(self.edlModel.truth.vehicle.LoD))
                print("BC : {} kg/m^2".format(self.edlModel.truth.vehicle.BC(InitialState[6])))

        else:
            self.edlModel = Entry(PlanetModel=Planet(rho0=rho0, scaleHeight=sh, da=self._use_da), VehicleModel=EntryVehicle(CD=CD, CL=CL), DifferentialAlgebra=self._use_da)
            self.edlModel.update_ratios(LR=AeroRatios[0],DR=AeroRatios[1])
            if self._output:
                print("L/D: {:.2f}".format(self.edlModel.vehicle.LoD))
                print("BC : {} kg/m^2".format(self.edlModel.vehicle.BC(InitialState[6])))
        self.update(np.asarray(InitialState),0.0,np.asarray([0]*3))
        self.control = Controllers
        while not self.is_Complete():
            temp = self.advance()

        self.history = np.vstack(self.history)                  # So that we can work with the data more easily than a list of arrays
        self.control_history.append(self.u)                     # So that the control history has the same length as the data;
        self.control_history = np.vstack(self.control_history[1:])
        self.simulations += 1
        if not self.simulations % 10:
            print("{} simulations complete.".format(self.simulations))
        # print self.x[0]
        return self.postProcess()


    def update(self, x, dt, u):
        if len(x.shape) == 1:
            self.x = x
        else:
            self.x = x[-1, :]

        if u is not None:
            self.u = u
            self.control_history.append(self.u)

        self.history.append(self.x)
        self.time += dt
        self.times.append(self.time)
        self.triggerInput = self.getDict()


    def printState(self):
        #  find nearest endpoint here - the trigger was met in the last ten steps
        if self._find_transitions:
            self.findTransition()
        if self._output:
            print('Transitioning from state {} to {} because the following condition was met:'.format(self._states[self.index], self.state))
            print(self._conditions[self.index].dump())
            for key,value in self.triggerInput.items():
                if key not in ('vehicle', 'current_state', 'planet'):
                    print('{} : {}\n'.format(key, value))
        self.index += 1
        self.ie.append(len(self.history)-1)


    def getDict(self):
        if self.fullEDL:
            L,D = self.edlModel.nav.aeroforces(np.array([self.x[8]]),np.array([self.x[11]]),np.array([self.x[15]]))

            d =  {
                  'time'            : self.time,
                  'altitude'        : self.edlModel.nav.altitude(self.x[7]),
                  'longitude'       : self.x[8],
                  'latitude'        : self.x[9],
                  'velocity'        : self.x[10],
                  'fpa'             : self.x[11],
                  'heading'         : self.x[12],
                #   'rangeToGo'       : self.x[14], # has to be computed 
                  'mass'            : self.x[13],
                  'drag'            : D[0],
                  'lift'            : L[0],
                  'vehicle'         : self.edlModel.nav.vehicle,
                  'planet'          : self.edlModel.nav.planet,
                  'current_state'   : self.x[7:14],
                  'aero_ratios'     : self.x[14:16],
                  'bank'            : self.x[16], # Should this be the current command or the current state?
                  'energy'          : self.edlModel.nav.energy(self.x[8],self.x[11],Normalized=False), # Estimated energy
                  }


        else:
            L,D = self.edlModel.aeroforces(self.x[0],self.x[3],self.x[6])
            rtg = 0 # TODO: Compute this 

            d =  {
                  'time'            : self.time,
                  'altitude'        : self.edlModel.altitude(self.x[0]),
                  'longitude'       : self.x[1],
                  'latitude'        : self.x[2],
                  'velocity'        : self.x[3],
                  'fpa'             : self.x[4],
                  'heading'         : self.x[5],
                  'rangeToGo'       : rtg,
                  'mass'            : self.x[6],
                  'drag'            : D,
                  'lift'            : L,
                  'vehicle'         : self.edlModel.vehicle,
                  'planet'          : self.edlModel.planet,
                  'current_state'   : self.x,
                  'aero_ratios'     : (self.edlModel.lift_ratio, self.edlModel.drag_ratio),
                  'bank'            : self.u[0],
                  'energy'          : self.edlModel.energy(self.x[0],self.x[3],Normalized=False),
                  'disturbance'     : 0,
                  }

        return d

    def ignite(self):
        self.edlModel.ignite()

    def bank_reversal(self):
        self.u[0] *= -1
        self.triggerInput = self.getDict()


    def viz(self, **kwargs):
        from TrajPlot import TrajPlot
        h = self.df['altitude'].values
        lat = np.radians(self.df['latitude'].values)
        long = np.radians(self.df['longitude'].values)

        # z = h*np.sin(lat)
        # x = h*np.cos(lat)*np.cos(long)
        # y = h*np.cos(lat)*np.sin(long)

        z = h
        y = self.df['crossrange']
        x = self.df['downrange']

        TrajPlot(x, y, z, **kwargs)

    def plot(self, plotEvents=True, compare=True, legend=True, plotEnergy=False):
        import matplotlib.pyplot as plt

        # To do: replace calls to self.history etc with data that can be passed in; If data=None, data = self.postProcess()

        if self.fullEDL:
            fignum = simPlot(self.edlModel.truth, self.times, self.history[:,0:7], self.history[:,16], plotEvents, self._states, self.ie, fignum=1, legend=legend, plotEnergy=plotEnergy)
            if compare:
                fignum = simPlot(self.edlModel.nav, self.times, self.history[:,7:14], self.control_history[:,0], plotEvents, self._states, self.ie, fignum=1, legend=legend, plotEnergy=False)  # Use same fignum for comparisons, set fignum > figures for new ones
            # else:
                # fignum = simPlot(self.edlModel.nav, self.times, self.history[:,8:16], self.control_history[:,0], plotEvents, self._states, self.ie, fignum=fignum, label="Navigated ")
            plt.figure(fignum)
            plt.plot(self.times, self.history[:,14],label='Lift')
            plt.plot(self.times, self.history[:,15], label='Drag')
            if legend:
                plt.legend(loc='best')
            plt.title('Aerodynamic Filter Ratios')

        else:
            simPlot(self.edlModel, self.times, self.history, self.control_history[:,0], plotEvents, self._states, self.ie, fignum=1, plotEnergy=plotEnergy, legend=legend)


    def show(self):
        import matplotlib.pyplot as plt
        plt.show()


    def postProcess(self):
        if self._use_da:
            from Utils.DA import degrees, radians
        else:
            from numpy import degrees, radians
            self.control_history = self.control_history.astype(float)
            self.history = self.history.astype(float)

        if self.fullEDL:

            bank_cmd = degrees(self.control_history[:,0])

            r,theta,phi = self.history[:,0], degrees(self.history[:,1]), degrees(self.history[:,2])
            v,gamma,psi = self.history[:,3], degrees(self.history[:,4]), degrees(self.history[:,5])
            m           = self.history[:,6]

            r_nav,theta_nav,phi_nav = self.history[:,8], degrees(self.history[:,9]), degrees(self.history[:,10])
            v_nav,gamma_nav,psi_nav = self.history[:,11], degrees(self.history[:,12]), degrees(self.history[:,13])
            m_nav                   = self.history[:,15]

            RL,RD = self.history[:,16], self.history[:,17]

            bank, bank_rate = degrees(self.history[:,18]), degrees(self.history[:,19])

            x0 = self.history[0,:]
            range = [self.edlModel.truth.planet.range(*x0[[1,2,5]],lonc=radians(lon),latc=radians(lat),km=True) for lon,lat in zip(theta,phi)]
            range_nav = [self.edlModel.nav.planet.range(*x0[[9,10,13]],lonc=radians(lon),latc=radians(lat),km=True) for lon,lat in zip(theta_nav,phi_nav)]

            energy = self.edlModel.truth.energy(r, v, Normalized=False)
            energy_nav = self.edlModel.nav.energy(r_nav, v_nav, Normalized=False)


            h = [self.edlModel.truth.altitude(R,km=True) for R in r]
            h_nav = [self.edlModel.nav.altitude(R,km=True) for R in r_nav]
            L,D = self.edlModel.truth.aeroforces(r,v,m)
            L_nav,D_nav = self.edlModel.nav.aeroforces(r_nav,v_nav,m_nav)



            data = np.c_[self.times, energy, bank_cmd, h,   r,      theta,       phi,      v,         gamma,     psi,       range,     L,      D,
                                     energy_nav, bank, h_nav, r_nav, theta_nav,  phi_nav,  v_nav,     gamma_nav, psi_nav,   range_nav, L_nav,  D_nav]
            vars = ['energy','bank','altitude','radius','longitude','latitude','velocity','fpa','heading','downrange','crossrange','lift','drag']
            all = ['time'] + vars + [var + '_nav' for var in vars ]
            self.df = pd.DataFrame(data, columns=all)

        else:
            bank_cmd = degrees(self.control_history[:,0])

            r,theta,phi = self.history[:,0], degrees(self.history[:,1]), degrees(self.history[:,2])
            v,gamma,psi = self.history[:,3], degrees(self.history[:,4]), degrees(self.history[:,5])
            m           = self.history[:,6]

            x0 = self.history[0,:]
            range = [self.edlModel.planet.range(*x0[[1,2,5]],lonc=radians(lon),latc=radians(lat),km=True) for lon,lat in zip(theta,phi)]
            energy = self.edlModel.energy(r, v, Normalized=False)

            h = [self.edlModel.altitude(R, km=True) for R in r]
            if self._use_da:
                L,D = np.array([self.edlModel.aeroforces(ri,vi,mi) for ri,vi,mi in zip(r,v,m)]).T
            else:
                L,D = self.edlModel.aeroforces(r,v,m)

            data = np.c_[self.times, energy, bank_cmd, h,   r,      theta,       phi,      v,         gamma, psi,       range,     L,      D, m]
            self.df = pd.DataFrame(data, columns=['time','energy','bank','altitude','radius','longitude','latitude','velocity','fpa','heading','downrange','crossrange','lift','drag','mass'])

        self.output = data
        return data

    def reset(self):
        """ Resets all simulation states to prepare for the 'run' method to be used again.
            The only exception is the .simuations member whose purpose to record the number of times
            'run' has been used for data reporting in e.g. Monte Carlo simulations.
        """
        if self._output:
            print("Resetting simulation states.\n")
        self.set_state(self._states[0])
        self.time = 0.0
        self.times = []
        self.index = 0
        self.sample = None          # Input uncertainty sample
        self.x = None               # Current State vector
        self.history = []           # Collection of State Vectors
        self.u = None
        self.control_history = []   # Collection of Control Vectors
        self.ie = [0]
        self.edlModel = None
        self.triggerInput = None
        self.control = None
        self.output = None


    def getRef(self):
        """ Computes a reference object for use in tracking based guidance
        """
        ref = {}

        vel = np.flipud(self.output[:,7]) # Flipped to be increasing for interp1d limitation
        alt = np.flipud(self.output[:,3]) # km
        radius = np.flipud(self.output[:,4]) # m
        range = np.flipud(self.output[-1,10]*1e3-self.output[:,10]*1e3) # Range to go
        drag = np.flipud(self.output[:,13])
        drag_rate = np.flipud(np.diff(self.output[:,13])/np.diff(self.output[:,0]))
        dragcos = np.flipud(self.output[:,13]/np.cos(np.radians(self.output[:,8])))

        bank = np.flipud(self.output[:,2])
        u = np.cos(np.radians(bank))

        hdot = vel*np.flipud(np.sin(np.radians(self.output[:,8])))

        i_vmax = np.argmax(vel)             # Only interpolate from the maximum downward so the reference is monotonic

        energy = np.flipud(self.output[:,1])
        # i_emax = np.argmax(energy)
        i_emax=i_vmax
        # Should probably use a loop or comprehension at this point...

        # Velocity as independent variable
        ref['drag'] = interp1d(vel[:i_vmax],drag[:i_vmax], fill_value=(drag[0],drag[i_vmax]), assume_sorted=True, bounds_error=False, kind='cubic')
        ref['drag_rate'] = interp1d(vel[:i_vmax],drag_rate[:i_vmax], fill_value=(drag_rate[0],drag_rate[i_vmax]), assume_sorted=True, bounds_error=False, kind='cubic')
        ref['altitude'] = interp1d(vel[:i_vmax],alt[:i_vmax], fill_value=(alt[0],alt[i_vmax]), assume_sorted=True, bounds_error=False, kind='cubic')
        ref['rangeToGo'] = interp1d(vel[:i_vmax],range[:i_vmax], fill_value=(range[0],range[i_vmax]), assume_sorted=True, bounds_error=False)
        ref['bank'] = interp1d(vel[:i_vmax],bank[:i_vmax], fill_value=(bank[0],bank[i_vmax]), assume_sorted=True, bounds_error=False, kind='nearest')
        ref['u'] = interp1d(vel[:i_vmax],u[:i_vmax], fill_value=(u[0],u[i_vmax]), assume_sorted=True, bounds_error=False, kind='nearest')

        fpa = np.radians(self.output[:,8])
        ref['fpa'] = interp1d(vel[:i_vmax],fpa[:i_vmax], fill_value=(fpa[0],fpa[i_vmax]), assume_sorted=True, bounds_error=False, kind='nearest')

        # Range as independent variable
        # import matplotlib.pyplot as plt
        # plt.figure(660)
        # plt.plot(range)
        # plt.show()
        # ref['altitude_range'] = interp1d(range, alt, fill_value=(alt[0],alt[-1]), assume_sorted=True, bounds_error=False, kind='cubic')

        # Energy as independent variable
        ref['dragcos'] = interp1d(energy[:i_emax],dragcos[:i_emax], fill_value=(dragcos[0],dragcos[i_emax]), assume_sorted=True, bounds_error=False, kind='cubic')
        ref['drag_energy'] = interp1d(energy[:i_emax], drag[:i_emax], fill_value=(drag[0],drag[i_emax]), assume_sorted=True, bounds_error=False, kind='cubic')
        ref['drag_rate_energy'] = interp1d(energy[:i_emax],drag_rate[:i_emax], fill_value=(drag_rate[0],drag_rate[i_emax]), assume_sorted=True, bounds_error=False, kind='cubic')
        ref['altitude_rate'] = interp1d(energy[:i_emax], hdot[:i_emax], fill_value=(hdot[0],hdot[i_emax]), assume_sorted=True, bounds_error=False, kind='cubic')

        return ref

    def getFBL(self):
        from FBL import drag_dynamics, drag_derivatives
        fbl = {}

        df = self.df

        # States
        radius = df['radius'].values   # m
        vel = df['velocity'].values
        fpa = np.radians(df['fpa'].values)
        bank = np.radians(df['bank'].values)
        u = np.cos(bank)

        # Accels
        drag = df['drag'].values
        lift = df['lift'].values
        g = self.edlModel.gravity(radius)

        # Drag derivs
        drag_rate,drag_accel = drag_derivatives(u, lift, drag, g, radius, vel, fpa, self.edlModel.planet.atmosphere(radius-self.edlModel.planet.radius)[0], self.edlModel.planet.scaleHeight)
        a,b = drag_dynamics(drag, drag_rate, g, lift, radius, vel, fpa, self.edlModel.planet.atmosphere(radius-self.edlModel.planet.radius)[0], self.edlModel.planet.scaleHeight)

        # Independent variable
        energy = df['energy'].values
        i_vmax = np.argmax(vel)             # Only interpolate from the maximum downward so the reference is monotonic
        i_emax=i_vmax

        # Interpolation objects (fill values are backward because energy is decreasing)
        fbl['bank']  = interp1d(energy[i_emax:], bank[i_emax:],    fill_value=(bank[-1],bank[i_emax]),             bounds_error=False, kind='cubic')

        fbl['a']  = interp1d(energy[i_emax:], a[i_emax:],          fill_value=(a[-1],a[i_emax]),                   bounds_error=False, kind='cubic')
        fbl['b']  = interp1d(energy[i_emax:], b[i_emax:],          fill_value=(b[-1],b[i_emax]),                   bounds_error=False, kind='cubic')

        fbl['D']  = interp1d(energy[i_emax:], drag[i_emax:],       fill_value=(drag[-1],drag[i_emax]),             bounds_error=False, kind='cubic')
        fbl['D1'] = interp1d(energy[i_emax:], drag_rate[i_emax:],  fill_value=(drag_rate[-1],drag_rate[i_emax]),   bounds_error=False, kind='cubic')
        fbl['D2'] = interp1d(energy[i_emax:], drag_accel[i_emax:], fill_value=(drag_accel[-1],drag_accel[i_emax]), bounds_error=False, kind='cubic')
        return fbl

    def findTransition(self):
        n = len(self.times)

        for i in range(n-2,n-12,-1):
            self.time = self.times[i]
            self.x = self.history[i]
            self.u = self.control_history[i]
            self.triggerInput = self.getDict()
            if self._use_da:
                trigger_input = da.const_dict(self.triggerInput)
            else:
                trigger_input = self.triggerInput
            if not self._conditions[self.index](trigger_input): # Interpolate between i and i+1 states
                for j in np.linspace(0.01,0.99,50): # The number of points used here will determine the accuracy of the final state
                    # Find a better state:
                    self.time = ((1-j)*self.times[i] + j*self.times[i+1])
                    self.x = ((1-j)*self.history[i] + j*self.history[i+1])
                    self.u = ((1-j)*self.control_history[i] + j*self.control_history[i+1])
                    self.triggerInput = self.getDict()
                    if self._conditions[self.index](trigger_input):
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
        if self._output:
            print("No better endpoint found")
        return

    # def save(self): #Create a .mat file
    def gui(self):
        import datetime
        uniq_filename = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')

def simPlot(edlModel, time, history, control_history, plotEvents, fsm_states, ie, fignum=1, label='', legend=True, plotEnergy=False):
    import matplotlib.pyplot as plt

    #history = [r, theta, phi, v, gamma, psi, s, m, DR, CR]
    # Altitude vs Velocity
    plt.figure(fignum)
    fignum += 1
    plt.plot(history[:,3], edlModel.altitude(history[:,0], km=True), lw = 3)
    if plotEvents:
        for i in ie:
            if legend:
                plt.plot(history[i,3],edlModel.altitude(history[i,0],km=True),'o',label = fsm_states[ie.index(i)], markersize=12)
            else:
                plt.plot(history[i,3],edlModel.altitude(history[i,0],km=True),'o', markersize=12)
    if legend:
        plt.legend(loc='upper left')
    plt.xlabel(label+'Velocity (m/s)')
    plt.ylabel(label+'Altitude (km)')


    e = edlModel.energy(history[:,0],history[:,3],Normalized=False)
    if plotEnergy: # Draw constant energy contours
        V,R = np.meshgrid(history[:,3], history[:,0])
        E = (np.array([edlModel.energy(r,V[0],Normalized=False) for r in R])-np.max(e))/(np.min(e)-np.max(e))

        V,H = np.meshgrid(history[:,3], edlModel.altitude(history[:,0],km=True))
        levels = (np.linspace(0,1,101))
        CS = plt.contour(V,H,(E),levels=levels,cmap='RdBu')
        plt.colorbar(format="%.2f")
        # plt.clabel(CS, inline=1, fontsize=10)

    if False: # Draw constant drag contours
        V,R = np.meshgrid(history[:,3], history[:,0])
        D_matrix = []
        for r in R:
            L,D = edlModel.aeroforces(r,V[0],history[:,6])
            D_matrix.append(D)
        levels = np.logspace(-5,2.4,11, endpoint=True)
        CS = plt.contour(V,H,(D_matrix),levels=levels,colors='k')
        # plt.clabel(CS, inline=1, fontsize=10)
        plt.clabel(CS)



    if False:
        en = edlModel.energy(history[:,0],history[:,3],Normalized=True)
        plt.figure(fignum)
        fignum += 1
        plt.plot(history[:,3], en, lw = 3)
        if plotEvents:
            for i in ie:
                plt.plot(history[i,3],en[i],'o',label = fsm_states[ie.index(i)], markersize=12)
        if legend:
            plt.legend(loc='upper left')
        plt.xlabel(label+'Velocity (m/s)')
        plt.ylabel(label+'Energy (-)')

    # #Latitude/Longitude
    plt.figure(fignum)
    fignum += 1
    plt.plot(history[:,1]*180/np.pi, history[:,2]*180/np.pi)
    if plotEvents:
        for i in ie:
            if legend:
                plt.plot(history[i,1]*180/np.pi, history[i,2]*180/np.pi,'o',label = fsm_states[ie.index(i)])
            else:
                plt.plot(history[i,1]*180/np.pi, history[i,2]*180/np.pi,'o')
    if legend:
        plt.legend(loc='best')
    plt.xlabel(label+'Longitude (deg)')
    plt.ylabel(label+'Latitude (deg)')
    # plt.legend()

    # Range vs Velocity
    plt.figure(fignum)
    fignum += 1
    plt.plot(history[:,3], (history[0,6]-history[:,6])/1000)
    if plotEvents:
        for i in ie:
            plt.plot(history[i,3],(history[0,6]-history[i,6])/1000,'o',label = fsm_states[ie.index(i)])
    if legend:
        plt.legend(loc='best')
    plt.xlabel(label+'Velocity (m/s)')
    plt.ylabel(label+'Trajectory length (km)')

    # Bank Angle Profile
    plt.figure(fignum)
    fignum += 1
    plt.plot(time, np.degrees(control_history[:]))
    for i in ie:
        plt.plot(time[i], np.degrees(control_history[i]),'o',label = fsm_states[ie.index(i)])
    if legend:
        plt.legend(loc='best')
    plt.xlabel(label+'Time (s)')
    plt.ylabel(label+'Bank Angle (deg)')

    # vs energy
    plt.figure(fignum)
    fignum += 1
    plt.plot(e, np.degrees(control_history[:]))
    for i in ie:
        plt.plot(e[i], np.degrees(control_history[i]),'o',label = fsm_states[ie.index(i)])
    if legend:
        plt.legend(loc='best')
    plt.xlabel(label+'Energy (s)')
    plt.ylabel(label+'Bank Angle (deg)')
    # Control vs Velocity Profile
    # plt.figure(fignum)
    # fignum += 1
    # plt.plot(history[:,3], np.cos(control_history[:]))
    # plt.plot(history[:,3], np.ones_like(control_history[:]),'k--',label='Saturation limit')
    # plt.plot(history[:,3], -np.ones_like(control_history[:]),'k--')
    # for i in ie:
        # plt.plot(history[i,3], np.cos(control_history[i]),'o',label = fsm_states[ie.index(i)])
    # if legend:
        # plt.legend(loc='best')
    # plt.axis([300,5505,-1.5,1.5])
    # plt.xlabel(label+'Velocity (m/s)')
    # plt.ylabel(label+'u=cos(sigma) (-)')

    # Downrange vs Crossrange
    range = np.array([edlModel.planet.range(*history[0,[1,2,5]],lonc=lon,latc=lat,km=True) for lon,lat in zip(history[:,1],history[:,2])])
    plt.figure(fignum)
    fignum += 1
    plt.plot(range[:,1], range[:,0])
    for i in ie:
        plt.plot(range[i,1], range[i,0],'o',label = fsm_states[ie.index(i)])
    if legend:
        plt.legend(loc='best')
    plt.xlabel(label+'Cross Range (km)')
    plt.ylabel(label+'Down Range (km)')
    plt.axis('equal')

    # Flight path vs Velocity
    plt.figure(fignum)
    fignum += 1
    plt.plot(history[:,3], history[:,4]*180/np.pi)
    if plotEvents:
        for i in ie:
            plt.plot(history[i,3],history[i,4]*180/np.pi,'o',label = fsm_states[ie.index(i)])
    if legend:
        plt.legend(loc='best')
    plt.xlabel(label+'Velocity (m/s)')
    plt.ylabel(label+'Flight path angle (deg)')


    L,D = edlModel.aeroforces(history[:,0],history[:,3],history[:,6])
    g = edlModel.gravity(history[:,0])

    plt.figure(fignum)
    fignum += 1
    plt.plot(history[:,3],D)
    if plotEvents:
        for i in ie:
            plt.plot(history[i,3], D[i],'o',label = fsm_states[ie.index(i)])
    plt.ylabel('Drag (m/s^2)')
    plt.xlabel('Velocity (m/s)')
    if legend:
        plt.legend(loc='best')

# #########################################################################
    if False:
        from FBL import drag_derivatives, drag_dynamics
        # u, L, D, g, r, V, gamma, rho, scaleHeight
        Ddot,Dddot = drag_derivatives(np.cos(control_history), L,D,g, history[:,0],history[:,3],history[:,4], edlModel.planet.atmosphere(history[:,0]-edlModel.planet.radius)[0],edlModel.planet.scaleHeight)
        plt.figure(fignum)
        fignum += 1
        plt.plot(time[1:],np.diff(D,1)/time[1],'--')
        plt.plot(time,Ddot)
        if plotEvents:
            for i in ie:
                plt.plot(time[i], Ddot[i],'o',label = fsm_states[ie.index(i)])
        plt.ylabel('Drag Rate (ms^-3)')
        # plt.xlabel('Velocity (m/s)')
        if legend:
            plt.legend(loc='best')

        plt.figure(fignum)
        fignum += 1
        plt.plot(time[2:],np.diff(D,2)/time[1]**2,'--')
        plt.plot(time,Dddot)
        if plotEvents:
            for i in ie:
                plt.plot(time[i], Dddot[i],'o',label = fsm_states[ie.index(i)])
        plt.ylabel('Drag second deriv  (ms^-4)')
        # plt.xlabel('Velocity (m/s)')
        if legend:
            plt.legend(loc='best')

        a,b=drag_dynamics(D, Ddot, g, L, history[:,0],history[:,3],history[:,4], edlModel.planet.atmosphere(history[:,0]-edlModel.planet.radius)[0],edlModel.planet.scaleHeight)
        u_test = (Dddot - a)/b
        bank_test = np.arccos(u_test)*np.sign(control_history[:])
        plt.figure(4)
        plt.plot(time, np.degrees(bank_test),'k--')

# ##########################################################################

    return fignum

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

def EntrySim(Vf=500):
    ''' Defines conditions for a simple one phase guided entry '''
    from Triggers import VelocityTrigger,AltitudeTrigger
    states = ['Entry']
    trigger = [VelocityTrigger(Vf)]
    # trigger = [AltitudeTrigger(5.1)]
    return {'states':states, 'conditions':trigger}

def TimedSim(time=30):
    from Triggers import TimeTrigger
    states = ['Entry']
    trigger = [TimeTrigger(time)]
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
    from InitialState import InitialState
    sim = Simulation(cycle=Cycle(1),output=True,**EntrySim())
    f = lambda **d: 0
    f2 = lambda **d: (1,2.88)
    c = [f,f,f2]
    r0, theta0, phi0, v0, gamma0, psi0,s0 = (3540.0e3, np.radians(-90.07), np.radians(-43.90),
                                             5505.0,   np.radians(-14.15), np.radians(4.99),   1180e3)
    # x0 = np.array([r0, theta0, phi0, v0, gamma0, psi0, s0, 8500.0]*2 +[1,1] + [0,0])
    x0 = InitialState(full_state=True)
    sim.run(x0, c, FullEDL=True)
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


def fsmGif(states = range(4)):
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
