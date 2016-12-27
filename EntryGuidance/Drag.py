from EntryEquations import EDL, Saturate

import numpy as np
from scipy.optimize import root
from scipy.integrate import cumtrapz
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
        sol = root(cost, guess, args=(e,d,m,model), tol=1e-5, options={'eps': 1e-6, 'diag':(3e-7,2e-4)})
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

    if E is None:
        E = model.energy(r, v, Normalized=False)
        L,D = model.aeroforces(r,v,m)
    else:
        L = model.aeroforces(r,v,m)[0]
   
    g = model.planet.mu/r**2

    e_dot = -D[:-2]*v[:-2]
    
    h_prime = np.diff(h*1e3)/np.diff(E)
    fpa = np.arcsin(np.clip(-h_prime*D[:-1],-1,1))
    
    s = cumtrapz(-np.cos(fpa)/D[:-1], E[:-1], initial=0)/1000 # True
    # s = cumtrapz(-1/D, E)/1000  # Approx - turns out the approximation is very good...
    t = cumtrapz(-1/D/v, E, initial=0)
    
    
    gamma_dot = np.diff(fpa)/np.diff(E[:-1])*e_dot
    
    u = (gamma_dot + (g[:-2]/v[:-2]-v[:-2]/r[:-2])*np.cos(fpa[:-1]))*v[:-2]/L[:-2]
    
    
    return np.degrees(fpa), s, u, t
    
if __name__ == '__main__':

    from Simulation import Simulation, Cycle, EntrySim
    from ParametrizedPlanner import HEPBank,HEPBankSmooth
    from Uncertainty import getUncertainty
    from InitialState import InitialState
    
    # Plan the nominal profile:
    reference_sim = Simulation(cycle=Cycle(.1),output=False,**EntrySim())
    bankProfile = lambda **d: HEPBankSmooth(d['time'],*[99.67614316,  117.36691891,  146.49573609],minBank=np.radians(30))
                                                
    x0 = InitialState()
    output = reference_sim.run(x0,[bankProfile])

    track_drag = 1
    
    references = reference_sim.getRef()
    if track_drag:
        istart = np.argmax(output[:,1])
    else:
        istart = np.argmax(output[:,7])
        
    energy = output[istart:,1]
    drag = output[istart:,13]/np.cos(np.radians(output[istart:,8]))
    # lift = output[istart:,12]
    sstart = output[istart,10] 
    reference_sim.plot(plotEnergy=True)  
    
    # samples = [np.zeros(4)]
    # samples = np.array([np.zeros(4), [-.2, -.1, .1, .2], np.zeros(4), np.zeros(4)]) # Lift dispersions
    # samples = np.array([[-.2, -.1, .1, .2], np.zeros(4), np.zeros(4), np.zeros(4)]) # Drag dispersions
    
    samples = getUncertainty()['parametric'].sample(10, 'S')
    
    # pdf = getUncertainty()['parametric'].pdf(samples)
    for sample in samples.T:
        if track_drag:
            h, v = solve(energy, drag, x0[7], model = EDL(sample))
        else:    
            h, v = output[istart:,3], output[istart:,7]
            energy = None
        fpa, s, u, t = reconstruct(energy, drag, h, v, x0[7]*np.ones_like(h), model = EDL(sample))
        
        plt.figure(1)
        plt.plot(v,h,label="pert: {}".format(sample))
        
        plt.figure(3)
        plt.plot(v[:-1],s+sstart,label="pert: {}".format(sample))
        
        plt.figure(5)
        plt.plot(v[:-2],u,label="pert: {}".format(sample))        
        
        plt.figure(7)
        plt.plot(v[:-1],fpa,label="pert: {}".format(sample))
        
    plt.legend(loc='best')
    plt.show()
    
    import chaospy as cp
    polynomials = cp.orth_ttr(order=5, dist=getUncertainty()['parametric']) 
    nodes, weights = cp.generate_quadrature(order=5, domain=getUncertainty()['parametric'], rule="Gaussian")
    print nodes.shape