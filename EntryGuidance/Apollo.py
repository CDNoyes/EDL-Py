""" Modified Apollo Final Phase Hypersonic Guidance for Mars Entry """

import numpy as np

# def controller(references, **kwargs):





# def get_ref_from_sim(sim):
    """ Outputs a set of interpolants for the reference range-to-go, drag, vertical L/D (i.e. Lcos(sigma)/D), and altitude rate as a function of navigated velocity """


# def gains(ref_traj):
    """ Determines the sensitivities based on a reference trajectory. 
            dR/dD
            dR/dr_dot
            dR/d(L/D)
    
    """
    
def predict_range(V, D, r_dot, ref, gains):
    # We could pass gamma instead of rdot if its simpler
    # Since the gains are all sensitivities of range, we could name them drag and r_dot etc to be consistent
    return ref['range'](V) + gains['dRdD'](V)*(D-ref['drag'](V)) + gains['dRdr_dot'](V)*(r_dot-ref['r_dot'](V))
    
    
def LoD_command(V, R, Rp, ref, gains):
    return ref['LoD'](V) + gains['k3']*(R-Rp)/gains['LoD'](V)
    
    
def bank_command(LoD, LoD_com):
    return np.arccos(LoD_com/LoD)
    
    