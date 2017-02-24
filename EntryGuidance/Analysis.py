""" Scripts and methods for analysis of monte carlo data """

import numpy as np
import MCF
    
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt


def filter():
    # mcdata = loadmat('./data/Box_3sig_Apollo_K1_energy.mat')
    # mcdata = loadmat('./data/MC_Apollo_1000_K1_energy.mat')
    mcdata = loadmat('./data/MC_Apollo_1000_K1_energy_no_rate.mat')
    
    outputs = mcdata['states'].flatten()
    inputs = mcdata['samples']
    if inputs.shape[0] == outputs.shape[0]:
        inputs = inputs.T
    
    # print "Cases on the lower altitude boundary:"
    # data = MCF.mcsplit(inputs,outputs,low_alt)
    # MCF.mcfilter(*data, input_names=['CD','CL','p0','hs'], plot=False) 
    
    # print "Cases more than 10 m/s different than nominal ignition velocity:"
    # data = MCF.mcsplit(inputs,outputs,high_vel)
    # MCF.mcfilter(*data, input_names=['CD','CL','p0','hs'], plot=False) 
    
    print "Cases more than 1 km from the target downrange:"
    data = MCF.mcsplit(inputs,outputs,large_miss)
    MCF.mcfilter(*data, input_names=['CD','CL','p0','hs'], plot=True) 
    
    
def low_alt(output):
    xf = output[-1]
    alt = xf[3] # in km
    return alt > 0.6
    
def high_vel(output):
    xf = output[-1]
    vel = xf[7] # in km
    return vel < 510 and vel > 490
    
def large_miss(output):    
    xf = output[-1]
    range = np.abs(output[-1,10]-905.5) # Range to go
    return range < 1

if __name__ == "__main__":
    filter()