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
        CD          = cp.Uniform(-0.10, 0.10)   # CD
        CL          = cp.Uniform(-0.10, 0.10)   # CL
        rho0        = cp.Normal(0, 0.0333)      # rho0
        scaleHeight = cp.Uniform(-0.05,0.05)    # scaleheight        
        perturbations['parametric'] = cp.J(CD,CL,rho0,scaleHeight)
        
    if knowledge:
        pitch       = cp.Normal(0, np.radians(1./3.)) # Initial attitude uncertainty in angle of attack, 1-deg 3-sigma
        perturbations['knowledge'] = cp.J(pitch)
    
    if initial:
        V     = cp.Uniform(-150,150)     # Entry velocity deviation
        gamma = cp.Normal(0, 2.0/3.0)    # Entry FPA deviation, +- 2 deg 3-sigma
        perturbations['initial'] = cp.J(V, gamma)
    
    return perturbations