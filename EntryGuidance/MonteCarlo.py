""" Defines utilities for conducting Monte Carlo simulations """
import multiprocess as mp
import numpy as np
import time
from datetime import datetime
import os
# import copy
from scipy.io import savemat


class MonteCarlo(object):
    """ Monte carlo class """

    def __init__(self):

        self.sim        = None
        self.samples    = None
        self.controls   = None


    def simulation(self, NewSimulation=None):
        """ Returns the current simulation object when called without an argument.
            Sets the current simulation to the input NewSimulation otherwise.
        """
        if NewSimulation is None:
            return self.sim
        else:
            print "Simulation updated"
            self.sim = NewSimulation
            return


    def sim_helper(self, phases, triggers):
        from Simulation import Simulation, Cycle
        inputs = {  'states'      : phases,
                    'conditions'  : triggers }
        self.sim = Simulation(cycle=Cycle(1), output=False, **inputs)
        print "Simulation created"
        print " "


    def __run(self, inputs):
        import copy
        # delta_x0, sample = inputs
        # return self.sim.run(self.x0_nom, self.controls, InputSample=inputs, FullEDL=True)
        return copy.deepcopy(self.sim).run(self.x0_nom, copy.deepcopy(self.controls), InputSample=inputs, FullEDL=True)


    def set_controls(self, controls):
        self.controls = controls
        return


    def reference_data(self):
        """ In closed loop simulations, generate reference data once and use it
            each simulation
        """
        print "Generating reference data for closed-loop guidance..."
        from Simulation import Simulation, Cycle, EntrySim
        from InitialState import InitialState
        from ParametrizedPlanner import profile

        x0 = InitialState()
        switch = [    62.30687581,  116.77385384,  165.94954234]
        bank = [-np.radians(30),np.radians(75),-np.radians(75),np.radians(30)]
        bankProfile = lambda **d: profile(d['time'],switch=switch, bank=bank,order=2)
        sim =  Simulation(cycle=Cycle(1),output=False, **EntrySim())
        res = sim.run(x0,[bankProfile])
        self.ref_sim = sim
        s0 = sim.history[0,6]-sim.history[-1,6] # This ensures the range to go is 0 at the target for the real simulation

        self.x0_nom = InitialState(1, range=s0, bank=bank[0])
        print "...done. "
        return


    def sample(self, N, sample_type='S', parametric=True, initial=False, knowledge=False):
        """ Generates samples for use in Monte Carlo simulations """
        from Uncertainty import getUncertainty
        uncertainty = getUncertainty(parametric=parametric, initial=initial, knowledge=knowledge)

        self.samples = uncertainty['parametric'].sample(N,sample_type)
        self.psamples = uncertainty['parametric'].pdf(self.samples)
        print(" ")
        print("Generating {} samples...".format(N))

        str2name = {'S':'Sobol','R':'Random','L':'Latin hypercube'}

        print("     {} sampling method".format(str2name[sample_type]))
        print("     Parametric uncertainty only")


    def run(self, save=None):

        if self.sim is None:
            print(" ")
            print("Warning: The simulation must be set prior to running.")
            print("Exiting.")
            return

        if self.controls is None:
            print(" ")
            print("Warning: The controls must be set prior to running.")
            print("Exiting.")
            return

        if self.samples is None:
            print(" ")
            print("Warning: The number and type of sampling must be set prior to running.")
            print("Exiting.")
            return

        pool    = mp.Pool(np.max([1,mp.cpu_count()-1]))
        print(" ")
        print("Running Monte Carlo...")
        t_start = time.time()
        outputs = pool.map(self.__run, self.samples.T)
        # outputs = [self.__run(sample) for sample in self.samples.T]
        print("Monte Carlo complete, time elapsed = {} s.".format(time.time()-t_start))
        pool.terminate()


        saveDir = './data/'
        if save is None:
            filename = getFileName('MC_{}'.format(len(self.samples.T)),saveDir)
            fullfilename = saveDir + filename
            print "Saving {} to default location {}".format(filename,saveDir)
        else:
            print "Saving data to user location {}".format(save)
            fullfilename = save


        savemat(fullfilename, {'states':outputs, 'samples':self.samples, 'pdf':self.psamples})



def getFileName(name, save_dir):
    """
        Looks in 'dir' for files with the pattern name-date-number

        I.e. if name = 'MC' and dir/MC-date-1 exists, MC-date-2 will be returned.
    """
    date = datetime.now().date()

    files =  os.listdir(save_dir)
    current = 1
    fname = "{}_{}_{}.mat".format(name,date,current)
    while fname in files:
        current += 1
        fname = "{}_{}_{}.mat".format(name,date,current)

    return fname

if __name__ == "__main__":
    from Triggers import TimeTrigger
    from NMPC import NMPC

    mc = MonteCarlo()
    mc.reference_data()
    mc.sim_helper(phases=['Entry'], triggers=[TimeTrigger(mc.ref_sim.time)])
    nmpc = NMPC(Ef=mc.ref_sim.df['energy'].values[-1],fbl_ref=mc.ref_sim.getFBL(),update_type=0,update_tol=2)
    controls = [nmpc.controller]
    mc.set_controls(controls)
    mc.sample(2000)
    # print len(mc.samples.T)
    mc.run()
