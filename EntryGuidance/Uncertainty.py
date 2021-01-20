''' Defines utilities for setting pertubations in uncertain systems '''

import numpy as np
import chaospy as cp


def getUncertainty(parametric=True, initial=False, knowledge=False):

    perturbations = {'parametric' : None,
                     'initial'    : None,
                     'knowledge'  : None,
                    }

    if parametric:
        # Define Uncertainty Joint PDF
        # CD          = cp.Uniform(-0.15, 0.15)   # CD
        CD          = cp.Normal(0, 0.15/3)   # CD
        # CL          = cp.Uniform(-0.15, 0.15)   # CL
        CL          = cp.Normal(0, 0.15/3)   # CL
        rho0        = cp.Normal(0, 0.20/3)      # rho0
        # scaleHeight = cp.Uniform(-0.02,0.01)      # scaleheight
        scaleHeight = cp.Normal(0, 0.02/3)           # scaleheight        # Positive delta scale height results in slower decay, i.e. thicker atm at the same altitude
        perturbations['parametric'] = cp.J(CD, CL, rho0, scaleHeight)

    if knowledge:
        pitch       = cp.Normal(0, np.radians(1./3.)) # Initial attitude uncertainty in angle of attack, 1-deg 3-sigma
        perturbations['knowledge'] = cp.J(pitch)

    if initial:
        R     = cp.Normal(0, 1)            # Entry velocity deviation
        theta     = cp.Normal(0, 2/3397)            # Entry velocity deviation
        phi     = cp.Normal(0, 2/3397)            # Entry velocity deviation
        V     = cp.Normal(0, 1)            # Entry velocity deviation
        gamma = cp.Normal(0, np.radians(0.25/3.0))       # Entry FPA deviation, +- 0.15 deg 3-sigma
        azi = cp.Normal(0, np.radians(0.25/3.0))       # Entry Azimuth deviation, +- 0.15 deg 3-sigma
        perturbations['initial'] = cp.J(R,theta,phi,V,gamma,azi)

    return perturbations
