import pickle
import time 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

from scipy.interpolate import interp1d, Rbf
from scipy.io import loadmat, savemat

import sys, os
sys.path.append("./")

from Utils.RK4 import RK4
from Utils.boxgrid import boxgrid 
from Utils.compare import compare 

from EntryGuidance.EntryEquations import Entry, EDL
from EntryGuidance.EntryPlots import EntryPlots
from EntryGuidance.Simulation import Simulation, Cycle, EntrySim
from EntryGuidance.InitialState import InitialState
from EntryGuidance.Planet import Planet 
from EntryGuidance.SRPController import SRPController, SRPControllerTrigger, update_rule_maker
from EntryGuidance.SRPData import SRPData 
from EntryGuidance.SRPUtils import range_from_entry, srp_from_entry
from EntryGuidance.Target import Target 
# from EntryGuidance.VMC import VMC, velocity_trigger
from EntryGuidance.Triggers import Trigger, VelocityTrigger
from EntryGuidance.Uncertainty import getUncertainty

SRPFILE = os.path.join(os.getcwd(), "data\\FuelOptimal\\srp_7200kg.pkl")




def simulate(x0, InputSample, plot=False):
    """ Runs a scalar simulation with the SRP Controller called multiple times """

    target = Target(0, 700/3397, 0)   
    TC = 2      # time constant 
    srpdata = pickle.load(open(SRPFILE,'rb'))

    # mcc = SRPController(N=[60, 200], target=target, srpdata=srpdata, update_function=update_rule_maker([5490, 4000, 2000, 1200]), debug=False, time_constant=TC)
    mcc = SRPController(N=[6, 2], target=target, srpdata=srpdata, update_function=update_rule_maker([5490, 4500, 3000, 2000, 1000]), debug=False, time_constant=TC)
    # mcc = SRPController(N=[6, 2], target=target, srpdata=srpdata, update_function=update_rule_maker([5490, 4700, 4000, 3000, 2000, 1000]), debug=False, time_constant=TC)
    Vf = 450     # Anything lower than the optimal trigger point is fine 
    
    states = ['Entry']
    trigger = [SRPControllerTrigger(mcc, -10)] # Go 10 m/s lower than the true trigger point says to
    sim_inputs = {'states': states, 'conditions': trigger}
    # sim = Simulation(cycle=Cycle(1), output=True, use_da=False, **EntrySim(Vf=Vf), )
    sim = Simulation(cycle=Cycle(1), output=False, use_da=False, **sim_inputs)
    sim.run(x0, [mcc], TimeConstant=TC, InputSample=InputSample, StepsPerCycle=10) 

    mf = mcc.srp_trim(sim.history)
    # mcc.plot_history()
    print("Final fuel consumed: {:.1f} kg".format(mf))
    if plot:
        sim.plot()    # The trajectory resulting from integrating the controller commands 
        mcc.plot_history()
        # mcc.sim.plot() # The (last) trajectory predicted by the controller
        # v1 = sim.df['velocity']
        # v2 = mcc.sim.df['velocity']
        # for var in ['lift']:
        #     y1 = sim.df[var]
        #     y2 = mcc.sim.df[var]
        #     compare(v1, y1, v2, y2, N=None, plot=True)
        #     plt.suptitle(var.capitalize())


    return mf, sim.df, mcc.history


def monte_carlo():

    # generates a set of inputs and saves them to a csv file
    # after each simulation, the relevant states are added to the csv file 
    # and the simulation is stored in a separate pickle file
    # then it moves on to the next sample 

    # TODO: Consider what happens if a sim fails (error) or bad SRP solutions (100k)

    savefile = "first_monte_carlo"

    EFPA = -15.75 # nominal 
    x0 = InitialState(vehicle='heavy', fpa=np.radians(EFPA))    
    # P0 = 
    # gaussian = cp.MvNormal(x0, P0)
    # X0 = gaussian.sample(N, 'L')
    N = 3
    n_parametric = 4
    n_states = 6 # up to 6, but we don't NEED to perturb altitude, velocity shouldn't have a significant impact either
    U = getUncertainty(initial=True)

    Ns = 60
    parametric_samples = U['parametric'].sample(Ns, 'S').T
    state_samples = U['initial'].sample(Ns, 'S').T

    # parametric_samples = boxgrid([[-0.1, 0.1] for _ in range(n_parametric)], N)
    print(parametric_samples.shape)
    # print(parametric_samples)

    if os.path.isfile(f"./data/FuelOptimal/{savefile}.csv"): # should just be a check to see if the savefile already exists 
        df_existing = pd.read_csv(f"./data/FuelOptimal/{savefile}.csv")
        data_existing = df_existing.values.T
        sample_existing = data_existing[3:7].T
        traj_data_existing = pickle.load(open(f"./data/FuelOptimal/{savefile}.pkl", 'rb'))

    else: # Use this when starting a new run, so the savefiles dont already exist 
        df_existing = None
        sample_existing = []
        traj_data_existing = {}

    data = [] # the summary data to go in a csv 
    traj_data = {}
    
    for n,full_sample in enumerate(zip(parametric_samples,state_samples)):
        sample,dx0 = full_sample
        dx0 = np.append(dx0, [0])
        print("\nSample {}: ".format(n+1))
        print(sample)
        print("State delta:")
        print(dx0)
        already_run = False
        for previous_sample in sample_existing:
            if tuple(sample) == tuple(previous_sample):
                already_run = True 
                break
        if already_run:
            print("Already run this sample\n")
            continue
        try:
            propellant, traj, history = simulate(x0+dx0, sample, False)
            data.append([propellant, *history['params'][-1], *sample, EFPA+np.degrees(dx0[4]), np.degrees(dx0[5]), *history['ignition_state'][-1], *history['entry_state'][-1]])
            traj_data[tuple(sample)] = {'traj': traj, 'history': history}
            traj_data_existing.pop(tuple(sample), None) # ensures we save the correct traj/hist 
        except:
            print("Simulation failed")

    if data:
        data = np.array(data).T
        data[1] = np.degrees(data[1])

        df = pd.DataFrame(data.T, columns=["fuel", "bank"," vr", 'cd','cl','rho0','hs',"efpa", "eazi", 'x','y','z','vx','vz', 'r', 'lon','lat', 'v', 'fpa','azi','m'])
        if df_existing is not None:
            df = pd.concat([df_existing, df], ignore_index=True)

        # try-except because one time my dumb@$$ had temp.csv open and I lost 2+ hours worth of computations 
        try:
            df.to_csv(f"./data/FuelOptimal/{savefile}.csv", index=False)
        except:
            df.to_csv("./data/FuelOptimal/temp1283787391.csv", index=False)

        if df_existing is not None:
            traj_data.update(traj_data_existing)
        pickle.dump(traj_data, open(f"./data/FuelOptimal/{savefile}.pkl", 'wb'))

def plot_monte_carlo_data():

    # savefile = "parametric_boxgrid"
    savefile = "first_monte_carlo"

    # summary data - solution, ignition state 
    df = pd.read_csv(f"./data/FuelOptimal/{savefile}.csv")
    data = df.values.T
    samples = data[3:7].T

    pmf = df['fuel'].values/df['m'].values * 100
    bad = df['fuel'].values > 5000

    # plt.figure()
    # plt.scatter(df['cd'], df['cl'], c=bad)
    # plt.xlabel("Cd")
    # plt.ylabel("Cl")

    # plt.figure()
    # plt.scatter(df['rho0'], df['hs'], c=bad)
    # plt.xlabel("rho0")
    # plt.ylabel("hs")
    
    plt.figure()
    # plt.hist(pmf)
    plt.hist(pmf[np.invert(bad)])

    # dictionary with samples as keys and 'traj' and 'history' entries 
    traj_data = pickle.load(open(f"./data/FuelOptimal/{savefile}.pkl", 'rb'))
    plot_trajectories(traj_data)


    plt.show()

def plot_trajectories(data):

    for key in data.keys():
        traj = data[key]['traj']
        hist = data[key]['history']
        # check that the prop cost wasnt ridic, skip if so 
        mf = hist['fuel'][-1]
        vf = hist['velocity'][-1]
        nbad = 0
        if mf > 5000:
            nbad += 1
            continue
        # trim DF for final vf 
        v = traj['velocity'].values
        h = traj['altitude'].values 
        keep = np.logical_and(v >= vf, h >= 3)
        EntryPlots(traj[v>=vf])

    print("{} bad trajectories".format(nbad))



if __name__ == "__main__":
    monte_carlo()
    # plot_monte_carlo_data()
    # N = 3
    # n_parametric = 4
    # n_states = 6 # up to 6, but we don't NEED to perturb altitude, velocity shouldn't have a significant impact either

    # parametric_samples = boxgrid([[-0.1, 0.1] for _ in range(n_parametric)], N)
    # for i,samp in enumerate(parametric_samples):
    #     print(f"{i}: {samp}")
