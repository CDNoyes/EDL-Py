import pickle
import numpy as np
import chaospy as cp 
import matplotlib.pyplot as plt 

from scipy.interpolate import interp1d
from scipy.io import loadmat, savemat

import sys, os
sys.path.append("./")

from Utils.RK4 import RK4
from Utils.boxgrid import boxgrid 
from Utils.gpops import srp 

from EntryGuidance.EntryEquations import Entry, EDL
from EntryGuidance.Simulation import Simulation, Cycle, EntrySim
from EntryGuidance.InitialState import InitialState
from EntryGuidance.Planet import Planet 
from EntryGuidance.SRPData import SRPData 
from EntryGuidance.SRPUtils import range_from_entry, srp_from_entry
from EntryGuidance.SRPController import reversal_controller
from EntryGuidance.Target import Target 
from EntryGuidance.VMC import VMC, velocity_trigger


def srp_landing(figsize=(15,10)):
    """ Shows an entry trajectory with predicted states powered and unpowered, separated at the predicted ignition point """

    params = (np.radians(26.5), 2804)
    profile = reversal_controller(*params, vectorized=False)
    def ref_profile(velocity, **kwargs):
        sigma = profile(velocity)
        return sigma 

    x0 = InitialState(vehicle='heavy', fpa=np.radians(-16.9))
    Vf = 480 

    dr = 753.7
    target = Target(0, dr/3397, 0)

    if 0:  # Run the actual simulation, find optimal ignition, and call GPOPS to get the powered trajectory 


        sim = Simulation(cycle=Cycle(0.1), output=False, use_da=False, **EntrySim(Vf=Vf), )
        sim.run(x0, [ref_profile], TimeConstant=2) 
        # sim.plot()

        # Load srpdata, trim the full traj
        srpdata = pickle.load(open(os.path.join(os.getcwd(), "data\\FuelOptimal\\srp_27k_5d.pkl"), 'rb'))
        sol = srpdata.srp_trim(sim.history, target, full_return=True)

        # print(sol)
        # call gpops to get trajectory from point for plotting 
        x0_srp = sol['ignition_state']
        x0_srp = np.insert(x0_srp, 4, 0).tolist()
        x0_srp.append(x0[-1])
        x0_srp = [float(x) for x in x0_srp]
        print(x0_srp)

        srp_traj = np.array(srp((x0_srp, 0, 0))['state'])
        x = sim.history 

        data = {'entry': sim.history, 'srp': srp_traj, 'sol': sol}
        pickle.dump(data, open(os.path.join(os.getcwd(), "data\\FuelOptimal\\diagrams_srp_landing_data.pkl"), 'wb'))
    else:  # Load data from a file 

        data = pickle.load(open(os.path.join(os.getcwd(), "data\\FuelOptimal\\diagrams_srp_landing_data.pkl"), 'rb'))
        x = data['entry']
        srp_traj = data['srp']
        sol = data['sol']

    # Then, plot full traj, plot ignition point, etc

    v_srp = sol['terminal_state'][3]
    dv_srp = 50 # additional velocity to show as the prediction 
    v_max = 600  # Doesn't have to be the real one, just for plotting 

    lines = {'linewidth': 3, }
    markers = {'markersize': 10}
    text = {'fontsize': 16}
    text_labels =  {'fontsize': 20}
    numbers = False   # whether to show numerical values or not 
    show_text = False

    for v_plot in [625,]:

        r,th,ph,v,fpa,azi,m = x[x.T[3]<v_plot].T
        s = th*3397 
        h = (r-3397e3)/1000
        if v_max <= v_plot:
            first = interp1d(v, np.array([dr-s, h]))(v_max)
        else:
            first = None 

        plt.figure(figsize=figsize)
        plt.plot(dr-s[v>=v_srp+dv_srp], h[v>=v_srp+dv_srp], 'b', label = "Unpowered Entry Trajectory", **lines)
        rtg = dr-s[v <= v_srp+dv_srp]
        plt.plot(rtg[rtg>=0], h[v <= v_srp+dv_srp][rtg>=0], 'darkorange', label="Predicted Unpowered Flight", **lines)
        plt.plot(*interp1d(v, np.array([dr-s, h]))(v_srp+dv_srp), 'k*', **markers)
        if show_text:
            plt.text(10.2, 3.9, "Current estimated state", **text)

        if first is not None:
            plt.plot(*first, 'mo', **markers) # based on fuel availability i.e. max velocity 
            if show_text:
                plt.text(first[0]+0.1, first[1]-0.4,"Maximum ignition \nvelocity", **text)

        plt.plot(dr-s[v>=v_srp][-1], h[v>=v_srp][-1], 'ro', **markers)
        if show_text:
            plt.annotate("Optimal\nignition\nstate", xy=(dr-s[v>=v_srp][-1], h[v>=v_srp][-1]),xytext=(dr-s[v>=v_srp][-1]-1, h[v>=v_srp][-1]+0.15), **text)

        # s_srp = np.linalg.norm(srp_traj.T[0:2], axis=0)/1000
        s_srp = np.abs(srp_traj.T[0])/1000 * 0.97225
        h_srp = srp_traj.T[2]/1000

        plt.plot(s_srp, h_srp, 'r', label="Powered Flight", **lines)
        plt.plot(0, 0, 'ko', **markers)
        if show_text:

            plt.text(0.45, -0.1, 'Powered \ndescent \ntarget', **text)
            plt.text(10, 1.6, "Minimum ignition altitude", **text)
            plt.text(2.8, 4.4, "Maximum ignition\ndistance to target", **text)

        plt.hlines(1.8, 0, np.max(dr-s[v>=v_srp+dv_srp]), 'm', '--')

        plt.vlines(7, 0, np.max(h[v>=v_srp+dv_srp]), 'g', '--')

        x = 4.5
        if show_text:
        
            plt.annotate('States checked \nfor propellant costs', xy=(x, 3.5), xytext=(x, 3.75),
                fontsize=text['fontsize'], ha='center', va='bottom',
                # bbox=dict(boxstyle='square', fc='white'),
                arrowprops=dict(arrowstyle='-[, widthB=8.0, lengthB=1.5', lw=2.0))

        plt.legend(loc='best', **text)
        if numbers:
            plt.xlabel("Range to go (km)", **text_labels)
            plt.ylabel("Altitude (km)", **text_labels)
        else:
            if show_text:
            
                plt.xlabel("Range to go", **text_labels)
                plt.ylabel("Altitude", **text_labels)
            
            plt.tick_params(
            axis='both',       # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            # top=False,         # ticks along the top edge are off
            left=False,
            labelbottom=False, # labels along the bottom edge are off
            labelleft=False,) # labels along the bottom edge are off

        plt.savefig("./Documents/FuelOptimal/H_Vs_S_{}.png".format(v_plot), bbox_inches='tight')


    plt.show()


def two_phase_edl():
    """ Creates a plot of the two phases - Entry vs SRP 
        Color the two sides of the plot 
    
    """

    plt.figure()
    plt.fill_between([0, 1], 1, 0, alpha=0.1, color='b')
    plt.fill_between([1, 2], 1, 0, alpha=0.1, color='r')

    plt.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    # top=False,         # ticks along the top edge are off
    left=False,
    labelbottom=False, # labels along the bottom edge are off
    labelleft=False,) # labels along the left edge are off
    plt.text(0.25, 0.9, "Entry Phase")
    plt.text(1.15, 0.9, "Powered Descent Phase")

    plt.savefig("./Documents/FuelOptimal/EDLPhaseDiagramBackdrop.png", bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    srp_landing()
    # two_phase_edl()