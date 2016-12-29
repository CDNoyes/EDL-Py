""" Reconstruction of families of trajectories defined by a reference profile. """

from EntryEquations import EDL

import numpy as np
from scipy.optimize import root
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt



def cost(x, E, D, m, model):
    ''' Roots of this function find the (r,v) pair to match the drag D at energy E for a given (possibly perturbed) model with mass m'''
    r,v = x
    
    e = model.energy(r, v, Normalized=False)
    l,d = model.aeroforces(np.array([r]), np.array([v]),np.array([m]))
    
    g = [e-E, d[0]-D]
    return g
    
def solve(E, D, m, model):
    ''' Computes a radius vs velocity profile matching the D vs E profile input for a given mass and system model. '''
    guess = [model.planet.radius + 85e3, 5500]
    r = []
    v = []
    for e,d in zip(E,D): 
        sol = root(cost, guess, args=(e,d,m,model), tol=1e-5, options={'eps': 1e-8, 'diag':(3e-7,2e-4)})
        r.append(sol['x'][0])
        v.append(sol['x'][1])
        guess = sol['x']
    
    h = model.altitude(np.array(r),km=True)
    return h,v
  

def reconstruct(E,D,h,v,m,model):
    ''' 
    Given an Energy vs Drag profile and the corresponding Altitude vs Velocity solution, this method
    reconstructs the flight path angle, downrange, and control trajectories.
    The initial conditions of the perturbed solution will not in general match the nominal solution.
    
    '''
    
    r = h*1e3 + model.planet.radius

    if E is None:                                   # h-v profile
        E = model.energy(r, v, Normalized=False)
        L,D = model.aeroforces(r,v,m)
    else:                                           # D-E profile
        L = model.aeroforces(r,v,m)[0]
   
    g = model.planet.mu/r**2

    e_dot = -D*v
    
    hE = interp1d(E[::-1], h[::-1]*1e3, fill_value=(h[-1]*1e3,h[0]*1e3), assume_sorted=True, bounds_error=False, kind='cubic')
    dE = 1 # Might be small enough since energy is large
    h_prime = (hE(E+dE)-hE(E-dE))/2/dE
    h_prime[0] = h_prime[1]
    h_prime[-1] = h_prime[-2]

    fpa = np.arcsin(np.clip(-h_prime*D,-1,1))
    
    s = cumtrapz(-np.cos(fpa)/D, E, initial=0)/1000 # True
    t = cumtrapz(-1/D/v, E, initial=0)
    
    dt = 1e-5
    fpaT = interp1d(t, fpa, fill_value=(fpa[0],fpa[-1]), assume_sorted=True, bounds_error=False, kind='cubic')
    gamma_dot = (fpaT(t+dt)-fpaT(t-dt))/2/dt
    gamma_dot[0]=gamma_dot[1]
    gamma_dot[-1]=gamma_dot[-2]
    
    u = (gamma_dot + (g/v-v/r)*np.cos(fpa))*v/L

    return np.degrees(fpa), s, u, t
    
if __name__ == '__main__':

    from Simulation import Simulation, Cycle, EntrySim
    from ParametrizedPlanner import HEPBank,HEPBankSmooth
    from Uncertainty import getUncertainty
    from InitialState import InitialState
    from MCF import mcsplit, mcfilter
    
    # Plan the nominal profile:
    reference_sim = Simulation(cycle=Cycle(0.1),output=False,**EntrySim())
    bankProfile = lambda **d: HEPBankSmooth(d['time'],*[99.67614316,  117.36691891,  146.49573609],minBank=np.radians(30))
                                                
    x0 = InitialState()
    # output = reference_sim.run(x0,[bankProfile]) 
    output = reference_sim.run(x0,[bankProfile],[0,-.15,-0.15,0.0]) #Plan with less lift
    references = reference_sim.getRef()

    # track_drag = 1
    for track_drag in range(2):
    
        if track_drag:
            # istart = np.argmax(output[:,1])
            istart = np.where(output[:,13] > 8)[0][0]
            iv = output[istart:,1]
        else:
            istart = np.argmax(output[:,7])
            iv = output[istart:,7]
            
        energy = output[istart:,1]
        time = output[istart:,0]
        altitude = output[istart:,3]
        velocity = output[istart:,7]
        drag = output[istart:,13] #/np.cos(np.radians(output[istart:,8]))
        # lift = output[istart:,12]
        # sstart = output[istart,10] 
        sstart = (reference_sim.history[0,6]-reference_sim.history[istart,6])/1000 
        # reference_sim.plot(plotEnergy=True)  
        
        # samples = [np.zeros(4)]
        # samples = np.array([np.zeros(4), [-.2, -.1, .1, .2], np.zeros(4), np.zeros(4)]) # Lift dispersions
        # samples = np.array([[-.2, -.1, .1, .2], np.zeros(4), np.zeros(4), np.zeros(4)]) # Drag dispersions
        
        samples = getUncertainty()['parametric'].sample(200, 'S')
        
        # pdf = getUncertainty()['parametric'].pdf(samples)
        hdata = []
        sdata = []
        fdata = []
        udata = []
        
        for sample in samples.T:
            if track_drag:
                h, v = solve(energy, drag, x0[7], model = EDL(sample))
            else:    
                h, v = output[istart:,3], output[istart:,7]
                energy = None
            fpa, s, u, t = reconstruct(energy, drag, h, v, x0[7]*np.ones_like(h), model = EDL(sample))
            
            hdata.append(h)
            sdata.append(s)
            fdata.append(fpa)
            
            plt.figure(1+track_drag)
            plt.plot(iv,h,'m',alpha=0.1,label="pert: {}".format(sample))
            
            plt.figure(3+track_drag)
            plt.plot(iv,s+sstart-2.06,'m',alpha=0.1,label="pert: {}".format(sample))
            
            plt.figure(5+track_drag)
            plt.plot(iv,u,'m',alpha=0.1,label="pert: {}".format(sample))        
            
            plt.figure(7+track_drag)
            plt.plot(iv,fpa,'m',alpha=0.1,label="pert: {}".format(sample))
            

        sdata = np.array(sdata).T + sstart - 2.06
        s_mean = np.mean(sdata, axis=1)
        s_var = np.diag(np.cov(sdata))
        s_std = np.sqrt(s_var)
        s_upper = s_mean + 3*s_std
        s_lower = s_mean - 3*s_std
        
        hdata = np.array(hdata).T
        h_mean = np.mean(hdata, axis=1)
        h_var = np.diag(np.cov(hdata))
        h_std = np.sqrt(h_var)
        h_upper = h_mean + 3*h_std
        h_lower = h_mean - 3*h_std
        
        fdata = np.array(fdata).T
        f_mean = np.mean(fdata, axis=1)
        f_var = np.diag(np.cov(fdata))
        f_std = np.sqrt(f_var)
        f_upper = f_mean + 3*f_std
        f_lower = f_mean - 3*f_std
        
        plt.figure(1+track_drag)
        plt.plot(iv, h_upper, 'k--',lw = 3)
        plt.plot(iv, h_lower, 'k--',lw = 3)
        plt.plot(iv, h_mean, 'k',lw = 3)
        
        plt.figure(3+track_drag)
        plt.plot(iv, s_upper, 'k--',lw = 3)
        plt.plot(iv, s_lower, 'k--',lw = 3)
        plt.plot(iv, s_mean, 'k',lw = 3)
        
        plt.figure(7+track_drag)
        plt.plot(iv, f_upper, 'k--',lw = 3)
        plt.plot(iv, f_lower, 'k--',lw = 3)
        plt.plot(iv, f_mean, 'k',lw = 3)
    
    # mcsplit(samples, hdata, criteria)
    
    # plt.legend(loc='best')
    plt.show()
    
    # import chaospy as cp
    # polynomials = cp.orth_ttr(order=5, dist=getUncertainty()['parametric']) 
    # nodes, weights = cp.generate_quadrature(order=5, domain=getUncertainty()['parametric'], rule="Gaussian")
    # print nodes.shape