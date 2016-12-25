from EntryEquations import EDL

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
    guess = [model.planet.radius + 65e3, 5000]
    r = []
    v = []
    for e,d in zip(E,D): 
        sol = root(cost, guess, args=(e,d,m,model), tol=1e-5, options={'eps': 1e-6, 'diag':(.0001,.01)})
        r.append(sol['x'][0])
        v.append(sol['x'][1])
        guess = sol['x']
    
    h = model.altitude(np.array(r),km=True)
    return h,v
  

def reconstruct(E,L,D,h,v,m,model):
    
    r = h[:-1] + model.planet.radius
    g = model.planet.mu/r**2
    e_dot = -D*v
    h_prime = np.diff(h*1e3)/np.diff(E)
    # h_dot = h_prime*e_dot[:-1]
    fpa = np.arcsin(-h_prime*D[:-1])
    
    s = cumtrapz(-np.cos(fpa)/D[:-1], E[:-1], initial=0)/1000 
    
    gamma_dot = np.diff(fpa)/np.diff(E[:-1])*e_dot[:-2]
    
    u = (gamma_dot + (g[:-1]/v[:-2]-v[:-2]/r[:-1])*np.cos(fpa[:-1]))*v[:-2]/L[:-2]
    
    
    return np.degrees(fpa), s, u
    
if __name__ == '__main__':

    from Simulation import Simulation, Cycle, EntrySim
    from ParametrizedPlanner import HEPBank,HEPBankSmooth
    from Uncertainty import getUncertainty
    
    # Plan the nominal profile:
    reference_sim = Simulation(cycle=Cycle(.1),output=False,**EntrySim())
    bankProfile = lambda **d: HEPBankSmooth(d['time'],*[ 165.4159422 ,  308.86420218, 399.53393904])
    # bankProfile = lambda **d: HEPBank(d['time'],*[ 165.4159422 ,  308.86420218,  399.53393904])
    
    r0, theta0, phi0, v0, gamma0, psi0,s0 = (3540.0e3, np.radians(-90.07), np.radians(-43.90),
                                             5505.0,   np.radians(-14.15), np.radians(4.99),   1000e3)
                                             
    x0 = np.array([r0, theta0, phi0, v0, gamma0, psi0, s0, 2804.0])
    output = reference_sim.run(x0,[bankProfile])

    references = reference_sim.getRef()
    istart = np.argmax(output[:,7])
    energy = output[istart:,1]
    drag = output[istart:,13]
    lift = output[istart:,12]
    sstart = references['rangeToGo'](output[istart,7])/1000. + 8
    reference_sim.plot(plotEnergy=True)  

    # sample = [-.1,0,0.2,0.01]
    for sample in getUncertainty()['parametric'].sample(5).T:
        h, v = solve(energy, drag, 2804.0, model = EDL(sample))
        
        fpa, s, u = reconstruct(energy, lift, drag, h, v, 2804.0, model = EDL(sample))
        
        plt.figure(1)
        plt.plot(v,h,label="pert: {}".format(sample))
        
        plt.figure(3)
        plt.plot(v[:-1],sstart-s,label="pert: {}".format(sample))
        
        plt.figure(5)
        plt.plot(v[:-2],u,label="pert: {}".format(sample))        
        
        plt.figure(7)
        plt.plot(v[:-1],fpa,label="pert: {}".format(sample))
        
    plt.legend(loc='best')
    plt.show()