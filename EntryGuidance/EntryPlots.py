import pickle
import time 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

def EntryPlots(tr, figsize=(10,6), fontsize=20, ticksize=16, savedir=None, fignum_offset=0, label=None):
    """ Takes a dataframe and plots all the things one could want, 
    with options for fontsize, figsize, saving, labeling

    """

    t = tr['time'].values
    v = tr['velocity'].values   
    figs = ['alt_vel','dr_cr','bank_vel','alt_vel_zoomed','fpa'] # used to autogenerate figure names 


    plt.figure(fignum_offset + 1, figsize=figsize)
    plt.plot(tr['velocity'], tr['altitude'], label=label)
    plt.xlabel('Velocity', fontsize=fontsize)
    plt.ylabel('Altitude (km)', fontsize=fontsize)
    plt.tick_params(labelsize=ticksize)
    plt.grid()
    plt.legend()

    plt.figure(fignum_offset + 2, figsize=figsize)
    plt.plot(tr['crossrange'], tr['downrange'], label=label)
    plt.ylabel('Downrange (km)', fontsize=fontsize)
    plt.xlabel('Crossrange (km)', fontsize=fontsize)
    # plt.axis('equal')
    plt.tick_params(labelsize=ticksize)
    plt.grid()
    plt.legend()

    plt.figure(fignum_offset + 3, figsize=figsize)
    plt.plot(tr['velocity'], tr['bank'], label=label)
    plt.tick_params(labelsize=ticksize)
    plt.xlabel('Velocity (m/s)', fontsize=fontsize)
    plt.ylabel('Bank Angle (deg)', fontsize=fontsize)
    plt.grid()
    plt.legend()

    plt.figure(fignum_offset + 4, figsize=figsize)
    plt.plot(tr['velocity'][v<=1200], tr['altitude'][v<1200], label=label)
    plt.xlabel('Velocity', fontsize=fontsize)
    plt.ylabel('Altitude (km)', fontsize=fontsize)
    plt.tick_params(labelsize=ticksize)
    plt.grid()
    plt.legend()

    plt.figure(fignum_offset + 5, figsize=figsize)
    plt.plot(tr['velocity'], tr['fpa'], label=label)
    plt.tick_params(labelsize=ticksize)
    plt.xlabel('Velocity (m/s)', fontsize=fontsize)
    plt.ylabel('Flight Path Angle (deg)', fontsize=fontsize)
    plt.grid()
    plt.legend()

    if savedir is not None:
        for i in range(len(figs)):
            plt.figure(fignum_offset + i+1)
            plt.savefig(os.path.join(savedir, "{}_{}".format(figs[i], names[fignum_mult-1])))