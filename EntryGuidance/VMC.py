import numpy as np
import chaospy as cp 
import matplotlib.pyplot as plt 

from scipy.interpolate import interp1d
from scipy.io import loadmat, savemat

import sys, os
sys.path.append("./")

from Utils.RK4 import RK4
from Utils.submatrix import submatrix 

from EntryGuidance.EntryEquations import Entry, EDL
from EntryGuidance.Simulation import Simulation, Cycle, EntrySim
from EntryGuidance.InitialState import InitialState
from EntryGuidance.Planet import Planet 
from EntryGuidance.SRPUtils import range_from_entry, srp_from_entry


class VMC(object):
    """ Monte carlo class 
    
        mc_full is a dictionary of the entire set of trajectories down to a low energy
        mc is a list of the trajectories after determining their trigger point 
        mc_srp is a dictionary of the optimal SRP ignition point, the corresponding terminal entry state, the fuel required from the optimal point 
    
    """

    def __init__(self):

        self.samples    = None
        self.control    = None
        self.mc         = None
        self._trigger   = None 

    def sample(self, N, sample_type='S', parametric=True, initial=False, knowledge=False):
        """ Generates samples for use in Monte Carlo simulations """
        from Uncertainty import getUncertainty
        uncertainty = getUncertainty(parametric=parametric, initial=initial, knowledge=knowledge)

        self.samples = uncertainty['parametric'].sample(N, sample_type)
        self.psamples = uncertainty['parametric'].pdf(self.samples)
        print(" ")
        print("Generating {} samples...".format(N))

        str2name = {'S': 'Sobol', 'R': 'Random', 'L': 'Latin hypercube'}

        print("     {} sampling method".format(str2name[sample_type]))
        print("     Parametric uncertainty only")
        
    def sample_state(self, x0, P0, N):
        gaussian = cp.MvNormal(x0, P0)
        X0 = gaussian.sample(N, 'L')
        return X0
        
    def null_sample(self, N,):
        """ Generates samples for use in Monte Carlo simulations """

        self.samples = np.zeros((4,N))
        print(" ")
        print("Generating {} samples...".format(N))


    def run(self, x0, stepsize=1, save=None, debug=False, Ef=None, time_constant=0):
        """ Stepsize can be a scalar value or a tuple of (dV, tmin, tmax)"""
        import time 
        if self.control is None:
            print(" ")
            print("Warning: The controls must be set prior to running.")
            print("Exiting.")
            return

        if self.samples is None:
            print(" ")
            print("Warning: The number and type of sampling must be set prior to running.")
            print("Exiting.")
            return


        print(" ")
        print("Running Monte Carlo...")
        t_start = time.time()
        self._run(x0, stepsize, Ef, time_constant, debug)
        print("Monte Carlo complete, time elapsed = {:.1f} s.".format(time.time()-t_start))

        if save is None:
            saveDir = './data/'
            filename = getFileName('MC_{}'.format(len(self.samples.T)), saveDir)
            fullfilename = saveDir + filename
            print("Saving {} to default location {}".format(filename, saveDir))
        elif save is False:
            return
        else:
            print("Saving data to user location {}".format(save))
            fullfilename = save

        savemat(fullfilename, {'xf': self.xf, 'states': self.mc, 'samples': self.samples, 'pdf': self.psamples})

    def _run(self, x, stepsize, Ef, time_constant, debug):
        
        try:
            adaptive = len(stepsize) == 3
            dV, minstep, maxstep = stepsize 
        except:
            adaptive = False 
        
        edl = EDL(self.samples, Energy=True)
        
        self.model = edl
        optSize = self.samples.shape[1]
        if x.ndim == 1:  # Allows a single initial condition or an array
            x = np.tile(x, (optSize, 1)).T
        X = [x]
        energy = np.mean(edl.energy(x[0], x[3], False))
#         print("E0 {:.1f}".format(energy))
        if Ef is None:
            energyf = 0 #edl.energy(edl.planet.radius-6000, 200, False)  # go down to low energy then parse afterward
        else:
            energyf = Ef 
#         print("Ef {:.1f}".format(energyf))

        Aero = []
        U = []
        E = [energy]
        if adaptive:
            S = []
        while True:
            if debug:
                print("\nE: {:.1f}".format(energy))
            Xc = X[-1]

            energys = edl.energy(Xc[0], Xc[3], False)
            lift, drag = edl.aeroforces(Xc[0], Xc[3], Xc[6])
            Aero.append(np.array([lift,drag]))
            
            if adaptive:
                stepsize = dV/np.mean(drag) # Will produce on average dV m/s vel stepsize 
                stepsize = min(stepsize, maxstep) # no steps larger than 10 seconds 
                stepsize = max(minstep, stepsize) # no steps smaller than 50 ms 
                S.append(stepsize)
                

            # Control
            u = self.control(energys, Xc, lift, drag)
            if U and time_constant:
                u = U[-1] + (u - U[-1])/time_constant * stepsize  # Smooth the control 
            U.append(u)
#             if debug:
#                 print(energys)
            # Shape the control
            u.shape = (1, optSize)
            u = np.vstack((u, np.zeros((2, optSize))))
#             de = -np.mean(drag)*np.mean(Xc[3]) * stepsize # This makes the average stepsize 1 s 
            de = -np.min(drag*Xc[3]) * stepsize  # This makes the largest stepsize 1 s for the slowest energy rate cases, smallest for the fastest cases 
            if debug:
                print(Xc.T[0])
                print("D: {:.4f}".format(drag[0]))
                print("Bank: {:.2f} deg".format(np.degrees(U[-1][0][0])))
            if (energy + de) < energyf:  # Make sure the exact final energy is hit. Most cases we wont care but sometimes we will 
                de = energyf - energy
            eom = edl.dynamics(u)
            xnew = RK4(eom, X[-1], np.linspace(energy, energy+de, 2))[-1]  # N x 7? 
            # fix = np.any(np.isnan(xnew), axis=1)
            # xnew[fix] = X[-1][fix] # If any state in a sample is NaN, set that entire state vector = to previous state so it just stops changing 
            X.append(xnew)
            energy += de
            E.append(energy)

            if np.isnan(energy):
                print("Something wrong")
                break

            if energy <= energyf:
                if debug:
                    print("energy depleted ")
                break 


        X = np.array(X)
        U = np.array(U)
        Aero = np.array(Aero)
        S = np.array(S)
        if debug:
            print("MC shape")
            print(X.shape)
            print(U.shape)
            print(Aero.shape)
        self.mc_full = {'state': X, 'control': U, 'aero': Aero, 'steps': S}
        self.trigger()  
        if debug:
            print("Terminal state shape: ")
            print(np.shape(self.xf))
        
        # This will provide flexibility to use different triggers 
    def set_trigger(self, trigger_function):
        self._trigger = trigger_function 

    def trigger(self):
        if self._trigger is None:
            self._trigger = final_trigger # Default trigger, just uses the final state, i.e. the one at the terminal energy 
        xfi = [self._trigger(traj) for traj in np.transpose(self.mc_full['state'], (2,0,1))] # the true stopping point is between this and xfi-1
        xf = [traj[i] for i, traj in zip(xfi, np.transpose(self.mc_full['state'], (2,0,1)))], 
        self.xf = np.array(xf).squeeze()

        self.mc = [traj[:i] for i, traj in zip(xfi, np.transpose(self.mc_full['state'], (2,0,1)))]
        self.mcu = [traj[:i] for i, traj in zip(xfi, np.transpose(self.mc_full['control'], (2,0,1)))]
        self.mca = [traj[:i] for i, traj in zip(xfi, np.transpose(self.mc_full['aero'], (2,0,1)))]
        
    def _obj(self, v, state, srp_data):
        x = state(v)
        return np.abs(srp_data(x))
        
    def _opt(self, bounds, state, srp_data):
        from scipy.optimize import minimize_scalar 
        sol = minimize_scalar(self._obj, method='bounded', bounds=bounds, args=(state, srp_data))
        return sol.x, sol.fun
        
    def srp_trim(self, srpdata, target, vmax=800, hmin=2000, optimize=False):
        """ Takes an SRP database class, and a pinpoint landing Target class 
            to calculate the optimal ignition point along a trajectory 
        
        """
        self.mc_srp = {'traj': [],'control': [],"fuel": [],"ignition_state": [],"terminal_state": [],"target": target}
        
        import time
        t0 = time.time()
        for traj in np.transpose(self.mc_full['state'], (2,0,1)):
            v = traj[:,3]
            k = v < vmax
            rtg,cr = range_from_entry(traj[k], target.coordinates())
            x_srp = srp_from_entry(traj[k], rtg, cr, target_alt=target.altitude).T # 5 x N
            h = x_srp[2]
            high = np.logical_and(h >= hmin, h <= 4000) # 4km is currently the maximum altitude 
            close = np.logical_and(x_srp[0] <= 8000, x_srp[0] >= 500) # 8km is currently the RTG limit in the table 
            close = np.logical_and(close, x_srp[1])   # 5km crossrange is the max in the table 
            high = np.logical_and(close, high)
            
            if np.sum(high) >= 2:
                # Potentially check the length, if more than N elements, use optimization 
                if optimize: # Use optimization of an interpolation function to find the minimum faster
                    v_srp, m_opt = self._opt([np.min(v[k][high]), vmax], interp1d(v[k][high], x_srp.T[high], axis=0, bounds_error=False, fill_value=(x_srp.T[high][-1], x_srp.T[high][0])), srpdata)
                    I = np.argmin(np.abs(v[k][high]-v_srp))
                else:
                    m_srp = srpdata(x_srp.T[high])
                    m_srp[m_srp<=500] = 10000 # for some reason, some models can report masses under zero 
                    I = np.argmin(np.abs(m_srp))
                    m_opt = m_srp[I]
                self.mc_srp['traj'].append(np.concatenate((traj[np.invert(k)], traj[k][high][:I])))
                self.mc_srp['terminal_state'].append(traj[k][high][I])
                self.mc_srp['fuel'].append(m_opt)
                self.mc_srp['ignition_state'].append(x_srp[:,high][:,I]) # may need a transpose
            
            else: # No suitable state was found
                self.mc_srp['traj'].append(np.concatenate(traj[np.invert(k)]))
                self.mc_srp['terminal_state'].append(traj[k][0])
                self.mc_srp['fuel'].append(100000)
                self.mc_srp['ignition_state'].append(x_srp[:,0]) 
        t1 = time.time()
        print("SRP Trim: {:.1f} s".format(t1-t0))
            
    def plot_srp(self, max_fuel_use=10000):
        try:
            self.mc_srp
        except:
            print("srp_trim must be run before plotting srp")
            
        
        mf = np.array(self.mc_srp['fuel'])
        keep = mf < max_fuel_use
        r,lon,lat,v,fpa,psi,m = np.array(self.mc_srp['terminal_state'])[keep].T

    
        h = (r-self.model.planet.radius)/1000
        dr = self.model.planet.radius*lon/1000
        cr = -self.model.planet.radius*lat/1000
        dr_target = self.mc_srp['target'].longitude*self.model.planet.radius/1000
        cr_target = -self.mc_srp['target'].latitude*self.model.planet.radius/1000
        
        plt.figure(3)
        plt.plot(v, h,'o')
        plt.xlabel('Velocity ')
        plt.ylabel('Altitude ')
        
        plt.figure(4)
        plt.subplot(1,2,2)
        plt.scatter(cr, dr, c=mf[keep])
        plt.plot(cr_target, dr_target, 'x', markersize=4)
        plt.xlabel('Crossrange')
        plt.ylabel('Downrange')

        plt.title("Colored by Fuel Used")
        plt.colorbar()
        
#         for X in self.mc_srp['traj']: 
#             r,lon,lat,v,fpa,psi,m = X.T
#             dr = self.model.planet.radius*lon/1000
#             cr = -self.model.planet.radius*lat/1000
#             h = self.model.altitude(r, km=True)
#             plt.figure(10)
#             plt.plot(v, h )
#             plt.xlabel('Velocity (m/s)')

#             plt.figure(11,)
#             plt.plot(cr, dr)
            
#             plt.figure(12, figsize=figsize)
#             plt.plot(v, np.degrees(U))
#             plt.xlabel('Velocity ')
        
    def plot_trajectories(self, figsize=(10, 6)):
        
        plt.figure(10, figsize=figsize)

        for X,U,A in zip(self.mc,self.mcu, self.mca): # each is n_samples x n_points now
            r,lon,lat,v,fpa,psi,m = X.T
            dr = self.model.planet.radius*lon/1000
            cr = -self.model.planet.radius*lat/1000
            h = self.model.altitude(r, km=True)
            plt.figure(10)
            plt.plot(v, h )
            plt.xlabel('Velocity ')

            plt.figure(11, figsize=figsize)
            plt.plot(cr, dr)
            
            plt.figure(12, figsize=figsize)
            plt.plot(v, np.degrees(U))
            plt.xlabel('Velocity ')
            
            plt.figure(13, figsize=figsize)
            plt.plot(v, np.degrees(fpa) ,label="FPA")

            plt.xlabel('Velocity (m/s)')
            plt.ylabel('FPA (deg)')
            
            plt.figure(14, figsize=figsize)
            plt.plot(v, np.degrees(psi), label='Azimuth' )
            plt.xlabel('Velocity (m/s)')
            plt.ylabel('Azimuth (deg)')
            
            plt.figure(15, figsize=figsize)
            plt.plot(v, A, label=['lift','drag'] )
            plt.xlabel('Velocity (m/s)')
            plt.ylabel('Aero acceleration (m/s^2)')

    def plot(self, figsize=(10, 6), fontsize=16):
        try:
            self.mc 
            self.xf 
        except AttributeError:
            if hasattr(self, 'mc_full'):
                print("MonteCarlo has been run but trigger has not been called to determine final states.")
            else:
                print("MonteCarlo must be run before plotting.")
            return 



        r,lon,lat,v,fpa,psi,m = self.xf.T
    
        h = (r-self.model.planet.radius)/1000
        dr = self.model.planet.radius*lon/1000
        cr = -self.model.planet.radius*lat/1000
        
        plt.figure(3)
        plt.plot(v, h,'o')
        plt.xlabel('Velocity ')
        plt.ylabel('Altitude ')
        plt.figure(4, figsize=(10,10))
        plt.subplot(1,2,1)

        plt.scatter(cr, dr, c=h)
        plt.xlabel('Crossrange')
        plt.ylabel('Downrange')

        plt.title("Colored by Altitude")
        plt.colorbar()


        # plt.figure(6, figsize=figsize)
        # plt.hist((r-self.model.planet.radius)/1000., cumulative=True, histtype='step', bins='auto', linewidth=4, density=True)
        # plt.xlabel("Final Altitude (km)")

    def load(self, mat_file):
        data = loadmat(mat_file)
        print(data['states'].shape)
        try:
            self.xf = data['xf']
        except KeyError:
            self.xf = data['states'][-1]

        self.mc = np.transpose(data['states'], (0,2,1))
        self.samples = data['samples']
        self.psamples = data['pdf']
        edl = EDL(self.samples, Energy=True)
        self.model = edl


def getFileName(name, save_dir):
    """
        Looks in 'save_dir' for files with the pattern name-date-number

        I.e. if name = 'MC' and dir/MC-date-1 exists, MC-date-2 will be returned.
    """
    import datetime 
    date = datetime.date.today()

    files = os.listdir(save_dir)
    current = 1
    fname = "{}_{}_{}.mat".format(name, date, current)
    while fname in files:
        current += 1
        fname = "{}_{}_{}.mat".format(name, date, current)

    return fname

def velocity_trigger(Vf=550):
    
    def _trigger(traj):
        for idx, state in enumerate(traj):
            if state[3] <= Vf:  
                return idx
        return -1
    return _trigger 

def altitude_trigger(h=4, vmax=600):
    
    def _trigger(traj):
        for idx,state in enumerate(traj):
            if (state[0]-3397000)/1000. <= h and state[3] <= vmax:
                return idx
        return -1
    return _trigger

def final_trigger(traj):
    return -1 


def test_controller(bank, v_reverse):
    """ This version requires v_reverse to be the same length as the state
    This is so that we can quickly determine where a reversal should go 
    
    """
    def _control(e,x,l,d):
        sigma = np.ones_like(x[0])*bank
        sigma[np.less_equal(x[3], v_reverse)] *= -1 
        return sigma
    return _control


def test():
    current_state = InitialState()
    N = 500

    vmc = VMC()
    vmc.null_sample(N)
    vmc.control = test_controller(np.radians(np.linspace(-90, 90, N)), np.ones((N,))*2750)
    vmc.set_trigger(velocity_trigger(500))
    vmc.run(current_state, save=False, stepsize=[5, 0.5, 10], time_constant=2.0, Ef=30000, debug=False)

    vmc.plot()
    vmc.plot_trajectories()
    plt.show()


if __name__ == "__main__":
    test()