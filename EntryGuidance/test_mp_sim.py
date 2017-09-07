""" Short script to test parallel simulations using multiprocess """


from functools import partial
import numpy as np

from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt


import multiprocess as mp


def multi_sim_runner(samples):
    from Simulation import Simulation, Cycle, EntrySim
    from InitialState import InitialState
    from ParametrizedPlanner import profile

    import numpy as np

    x0 = InitialState()
    switch = [    62.30687581,  116.77385384,  165.94954234]
    bank = [-np.radians(30),np.radians(75),-np.radians(75),np.radians(30)]
    bankProfile = lambda **d: profile(d['time'],switch=switch, bank=bank,order=2) # Profile is more generic
    sim =  Simulation(cycle=Cycle(1),output=False, **EntrySim())
    res = [sim.run(x0,[bankProfile],InputSample=sample) for sample in samples]
    return res

def sim_runner(sample):
        from Simulation import Simulation, Cycle, EntrySim
        from InitialState import InitialState
        from ParametrizedPlanner import profile

        import numpy as np

        x0 = InitialState()
        switch = [    62.30687581,  116.77385384,  165.94954234]
        bank = [-np.radians(30),np.radians(75),-np.radians(75),np.radians(30)]
        bankProfile = lambda **d: profile(d['time'],switch=switch, bank=bank,order=2) # Profile is more generic

        return Simulation(cycle=Cycle(1),output=False, **EntrySim()).run(x0,[bankProfile],InputSample=sample)

def sim():
    from Uncertainty import getUncertainty


    N = 1000
    print("Running Monte Carlo with {} samples".format(N))
    perturb = getUncertainty()['parametric']
    samples = perturb.sample(N,'S')
    p = perturb.pdf(samples)

    import time
    t0 = time.time()
    simpool = mp.Pool(mp.cpu_count()-1)
    # simpool = mp.Pool(1)
    outputs = simpool.imap(sim_runner,samples.T)
    outputs1 = [output for output in outputs]

    simpool.terminate()
    t1 = time.time()
    print "MP time elapsed = {} s".format(t1-t0)

    # outputs2 = multi_sim_runner(samples.T)
    # print "Serial time elapsed = {} s".format(time.time()-t1)
    saveDir = './data/'
    savemat(saveDir+'MC_MPTest_{}'.format(N),{'states':outputs1, 'samples':samples, 'pdf':p})
    return


if __name__ == "__main__":

    mp.freeze_support()

    sim()
