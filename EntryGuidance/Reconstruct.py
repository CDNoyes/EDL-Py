""" Reconstruction of families of trajectories defined by a reference profile. """

from EntryEquations import EDL

import numpy as np
from scipy.optimize import root
from scipy.integrate import cumtrapz, odeint
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
    
    # hE = interp1d(E[::-1], h[::-1]*1e3, fill_value=(h[-1]*1e3,h[0]*1e3), assume_sorted=True, bounds_error=False, kind='cubic')
    hE = interp1d(E[::-1], h[::-1]*1e3, fill_value="extrapolate", assume_sorted=True, bounds_error=False, kind='linear')
    dE = 1 # Might be small enough since energy is large
    h_prime = (hE(E+dE)-hE(E-dE))/2/dE
    h_prime[0] = h_prime[1]
    h_prime[-1] = h_prime[-2]

    fpa = np.arcsin(np.clip(-h_prime*D,-1,1))
    
    s = cumtrapz(-np.cos(fpa)/D, E, initial=0)/1000 # True
    t = cumtrapz(-1/D/v, E, initial=0)
    
    dt = 1e-5
    # fpaT = interp1d(t, fpa, fill_value=(fpa[0],fpa[-1]), assume_sorted=True, bounds_error=False, kind='cubic')
    fpaT = interp1d(t, fpa, fill_value="extrapolate", assume_sorted=True, bounds_error=False, kind='linear')
    gamma_dot = (fpaT(t+dt)-fpaT(t-dt))/2/dt
    gamma_dot[0]=gamma_dot[1]
    gamma_dot[-1]=gamma_dot[-2]
    
    u = (gamma_dot + (g/v-v/r)*np.cos(fpa))*v/L

    return np.degrees(fpa), s, u, t
    
def reconstruct_rs(r,s,gamma,v0,m,model):
    
    rfun = interp1d(s, r, fill_value="extrapolate", assume_sorted=True, bounds_error=False, kind='linear')
    gammafun = interp1d(s, gamma, fill_value="extrapolate", assume_sorted=True, bounds_error=False, kind='linear')
    v = odeint(v_prime,v0,s,args=(rfun,gammafun,m[0],model))[:,0]

    L,D = model.aeroforces(r,v,m)
    E = model.energy(r,v,False)
    
    t = cumtrapz(-1/D/v, E, initial=0)
    dt = 1e-5
    gammafunt = interp1d(t, gamma, fill_value="extrapolate", assume_sorted=True, bounds_error=False, kind='linear')
    gamma_dot = (gammafunt(t+dt)-gammafunt(t-dt))/2/dt
    g = model.planet.mu/r**2

    u = (gamma_dot + (g/v-v/r)*np.cos(gamma))*v/L
    
    return v, u, t
    
def v_prime(v,s,r,gamma,m,model):
    g = model.planet.mu/r(s)**2
    L,D = model.aeroforces(np.asarray([r(s)]),[v],[m])
    Vdot = -D - g*np.sin(gamma(s))
    sdot = v*np.cos(gamma(s))
    return Vdot/sdot
    
if __name__ == '__main__':

    from Simulation import Simulation, Cycle, EntrySim
    from ParametrizedPlanner import HEPBank,HEPBankSmooth
    from Uncertainty import getUncertainty
    from InitialState import InitialState
    from MCF import mcsplit, mcfilter
    import chaospy as cp
    
    # Plan the nominal profile:
    reference_sim = Simulation(cycle=Cycle(.1),output=False,**EntrySim())
    bankProfile = lambda **d: HEPBankSmooth(d['time'],*[99.67614316,  117.36691891,  146.49573609],minBank=np.radians(30))
                                                
    x0 = InitialState()
    output = reference_sim.run(x0,[bankProfile]) 
    # output = reference_sim.run(x0,[bankProfile],[0,-.15,-0.15,0.0]) #Plan with less lift
    references = reference_sim.getRef()
    N = 50
    colors = ['b','r','m','g','k']
    # labels = ['CD','CL','rho0','hs','Total']
    labels = ['Total']
    
    delta = cp.Normal(0, 1).sample(N)
    sample_sets = [ #np.array([delta*0.15/3, np.zeros(N), np.zeros(N), np.zeros(N)]),       # Drag dispersions
                    #np.array([np.zeros(N), delta*0.15/3, np.zeros(N), np.zeros(N)]),       # Lift dispersions
                    #np.array([np.zeros(N), np.zeros(N), delta*0.05, np.zeros(N)]),         # Density at h=0 dispersions
                    #np.array([np.zeros(N), np.zeros(N), np.zeros(N), delta*0.02/3]),       # Scale height dispersions
                    getUncertainty()['parametric'].sample(N, 'S')]                         # Full dispersion set, sampled by Sobol' sequence
    sample_sets = [np.zeros(4)]
    for samples, color, label in zip(sample_sets, colors, labels):  
        print "Running deltas on {}, plotted with color {}".format(label,color)
        for family in range(2,3):
        
            if family == 0:
                # istart = np.argmax(output[:,1])
                istart = np.where(output[:,13] > 8)[0][0]
                iv = output[istart:,1]
                xlabel = 'Energy'
                title = 'D(E)'
            elif family == 1:
                istart = np.argmax(output[:,7])
                iv = output[istart:,7]
                xlabel = 'Velocity (m/s)'
                title = 'h(V)'
            else:
                # istart = np.argmax(output[:,7])
                istart = 0
                iv = output[istart:,10]-output[istart,10]
                xlabel = 'Range (km)'
                title = 'h(s)' 
                
            energy = output[istart:,1]
            time = output[istart:,0]
            altitude = output[istart:,3]
            radius = output[istart:,4]
            velocity = output[istart:,7]
            flightpath = output[istart:,8]
            drag = output[istart:,13]
            # lift = output[istart:,12]
            # sstart = output[istart,10] 
            sstart = (reference_sim.history[0,6]-reference_sim.history[istart,6])/1000 
            # reference_sim.plot(plotEnergy=True)  
            

            hdata = []
            sdata = []
            fdata = []
            udata = []
            vdata = []
            
            for sample in samples.T:
                if family == 0:
                    h, v = solve(energy, drag, x0[7], model = EDL(sample))
                elif family == 1:    
                    h, v = altitude, velocity
                    energy = None

                if family in (0,1):    
                    fpa, s, u, t = reconstruct(energy, drag, h, v, x0[7]*np.ones_like(h), model = EDL(sample))
                else:
                    h = altitude
                    s = iv
                    fpa = flightpath
                    v, u, t = reconstruct_rs(radius, iv*1e3, np.radians(flightpath), velocity[0], x0[7]*np.ones_like(iv), model=EDL(sample))
                hdata.append(h)
                sdata.append(s)
                fdata.append(fpa)
                vdata.append(v)
                
                if True:
                    plt.figure(1+family)
                    plt.plot(iv,h,'m',alpha=0.1)
                    
                    plt.figure(11+family)
                    plt.plot(iv,s+sstart-2.06,'m',alpha=0.1)       
                    
                    plt.figure(31+family)
                    plt.plot(iv,fpa,'m',alpha=0.1)
                    
                    plt.figure(41+family)
                    plt.plot(iv,v,'m',alpha=0.1)
                    
                plt.figure(21+family)
                plt.plot(iv,u,color,alpha=0.1) 
                plt.xlabel(xlabel)
                plt.ylabel('u=cos(sigma)')
                plt.axis([iv.min(),iv.max(),-2,2])
                
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
            
            vdata = np.array(vdata).T
            v_mean = np.mean(vdata, axis=1)
            v_var = np.diag(np.cov(vdata))
            v_std = np.sqrt(v_var)
            v_upper = v_mean + 3*v_std
            v_lower = v_mean - 3*v_std
            
            plt.figure(1+family)
            plt.plot(iv, h_upper, '--',lw = 3, label=label, color=color)
            plt.plot(iv, h_lower, '--',lw = 3, color=color)
            plt.plot(iv, h_mean, lw = 3, color=color)
            plt.xlabel(xlabel)
            plt.ylabel('Altitude (km)')
            plt.title(title)
            
            plt.figure(11+family)
            plt.plot(iv, s_upper, '--',lw = 3, label=label, color=color)
            plt.plot(iv, s_lower, '--',lw = 3, color=color)
            plt.plot(iv, s_mean, lw = 3, color=color)
            plt.xlabel(xlabel)
            plt.ylabel('Range (km)')
            plt.title(title)
            
            plt.figure(31+family)
            plt.plot(iv, f_upper, '--',lw = 3, label=label, color=color)
            plt.plot(iv, f_lower, '--',lw = 3, color=color)
            plt.plot(iv, f_mean, lw = 3, color=color)
            plt.xlabel(xlabel)
            plt.ylabel('FPA (deg)')
            plt.title(title)
            
            plt.figure(41+family)
            plt.plot(iv, v_upper, '--',lw = 3, label=label, color=color)
            plt.plot(iv, v_lower, '--',lw = 3, color=color)
            plt.plot(iv, v_mean, lw = 3, color=color)
            plt.xlabel(xlabel)
            plt.ylabel('Velocity (m/s)')
            plt.title(title)
    # mcsplit(samples, hdata, criteria)
    
    plt.legend(loc='best')
    plt.show()
    
    # import chaospy as cp
    # polynomials = cp.orth_ttr(order=5, dist=getUncertainty()['parametric']) 
    # nodes, weights = cp.generate_quadrature(order=5, domain=getUncertainty()['parametric'], rule="Gaussian")
    # print nodes.shape