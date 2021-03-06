import os,sys 
import numpy as np
from scipy.interpolate import interp1d, LinearNDInterpolator, Rbf 
from scipy.io import loadmat, savemat

import matplotlib.pyplot as plt 
sys.path.append("./")
from Utils.boxgrid import boxgrid 
from EntryGuidance.SRPUtils import range_from_entry, srp_from_entry

class SRPData:
    """ The RBF method appears best, outperforming both SVR and linear ND
    
    Next we need to optimize kernel and epsilon parameters 
    
    """
    def __init__(self, file, min_alt, max_fuel):
        self.fname = file
        self.data = loadmat(file) # the original data from file 
        self.hmin = min_alt 
        self.mmax = max_fuel
        self.mmin = 0
        self.model_data = {} # the data that actually is used to build the model 
        self.trimmed_data = {} # the data after trimming 
    
    
    def validate(self, vfile=None):
        """ Take all the samples that didn't get used"""

        if vfile is None:
            N = self.trimmed_data['output'].squeeze().size - self.model_data['output'].squeeze().size
            if N == 0:
                print("All data was used for model, no validation possible")
                return
            else:
                print("Validating with {} samples".format(N))

            ind = np.setdiff1d(range(self.trimmed_data['output'].squeeze().size), self.model_indices)
            val_in = self.trimmed_data['input'][:, ind] # The inputs that were not selected for the model 
            mf_true = self.trimmed_data['output'][ind]
            mf_model = self(val_in)
            
        else:
            data = loadmat(vfile)
            input_data = data['initial'].T
            input_data = input_data[:-1] # drop the mass column 
            output_data = data['fuel'].squeeze()

            val_in, mf_true = self.trim(input_data, output_data)  # Makes sense that if we trim the table data a certain way, we shouldn't validate without trimming the data the same way 
            print("Validating with {} samples".format(len(mf_true)))
            mf_model = self(val_in)
            
        k = np.invert(np.isnan(mf_model))
        err = mf_model[k]-mf_true[k]
        I = np.argmax(np.abs(err))
        for percentile in [90, 95, 99, 99.9]:
            print("Absolute Error, {}% = {:.1f} kg".format(percentile, np.percentile(np.abs(err), percentile)))
            print("Relative Error, {}% = {:.2f}%\n".format(percentile, np.percentile(np.abs(err)*100/mf_true[k], percentile)))
            
        print("Input state with largest error (= {:.1f} kg):".format(err[I]))
        print(val_in[:,k][:,I])

        plt.figure()
        plt.hist(err, bins=50)
        plt.xlabel("Model-True, Fuel Consumption (kg)")
        
        plt.figure()
        plt.hist(np.abs(err)*100/mf_true[k], bins=50, density=True, cumulative=True, histtype='step')
        plt.xlabel("Absolute Percent Error")

        plt.figure()
        plt.plot(mf_true[k], err, 'o')
        plt.xlabel("True Fuel (kg)")
        plt.ylabel("Fuel Error (kg)")
        
        # TODO: Rebuild the model with the top N percent error samples added to the dataset

        return err 
        
        
    def trim(self, input_data, output_data):
        """ Removes data points outside the bounds we want 
        In particular low altitude cases and overshoot cases 
        
        
        """
        N = len(output_data)
        x,y,z,u,v,w = input_data 
        
        
        keep = z > self.hmin 
        Nhigh = np.sum(keep)
        print("{} trajectories of {} satisfy ignition altitude >= {} km".format(Nhigh, N, self.hmin/1000))
        
        # Max fuel check - very nice because it vastly reduces the number of points we keep
        keep = np.logical_and(keep, output_data <= self.mmax)
        print("{} trajectories of {} satisfy fuel use <= {} kg".format(np.sum(output_data <= self.mmax), N, self.mmax))
        
        # assuming nominally x is positive and u is negative 
        m0 = self.m0 # obtained from data
        Tmax = 15*m0
        isp = 290
        g0 = 9.81
        mdot = Tmax/(isp*g0)
        tf = m0/mdot * (1-np.exp(-np.abs(u)*mdot/Tmax)) # The time it would take to slow to zero horizontal velocity if all of the available thrust were directed horizontally 
        mf = m0 - mdot*tf 
        dx = u*tf - Tmax/mdot*( (tf-m0/mdot)*np.log(mf/m0) - tf) # This is how much range to go is eliminated with full horizontal thrust until zero horizontal velocity, should be a negative number 
        # We want the actual range to go to be greater than or equal to this, otherwise we are guaranteed to overshoot the target since generally not all thrust will be directed horizontally 
        far_enough = x >= np.abs(dx)
#         print("{} trajectories of {} satisfy the 'no overshoot' criterion".format(np.sum(far_enough), N))
#         keep = np.logical_and(keep, far_enough)
        
#         g = 3.71
#         # Perform the same check for altitudes 
#         tf = m0/mdot * (1-np.exp(-np.abs(u)*mdot/Tmax)) # The time it would take to slow to zero vertical velocity if all of the available thrust were directed horizontally 
#         mf = m0 - mdot*tf 
#         dz = w*tf - Tmax/mdot*( (tf-m0/mdot)*np.log(mf/m0) - tf) - 0.5*g*tf^2 # This is how much altitude is lost with full vertical thrust until zero vertical velocity, should be a negative number 
#         # We want the actual altitude to be greater than or equal to this, otherwise we are guaranteed to crash 
#         high_enough = z >= np.abs(dz)
#         print("{} trajectories of {} satisfy the 'no overshoot' criterion".format(np.sum(high_enough), N))
#         keep = np.logical_and(keep, high_enough)
#         print("{} trajectories trimmed in total".format(N-np.sum(keep)))
        
        V = np.sqrt(u**2 + v**2 + w**2)
        slow_enough = V<=700
        keep = np.logical_and(keep, slow_enough)
        print("{} trajectories of {} satisfy the ||V|| <= 700 criterion".format(np.sum(slow_enough), N))
        input_data = np.delete(input_data, 4, 0) # removes the all zeros cross track velocity 
        return input_data[:, keep], output_data[keep]
        
    def pad(self, min_alt):
        """ Adds a layer of high cost points at the minimum altitude to further discourage very low ignitions"""
        pass 
            
    def build(self, N=0, rbf_kw={'function': 'quintic'}):
        """ model options are [rbf, linear, nearest]
        N is the number of samples to use, only required if the model is too big with full sample size"""
        
        from time import time 
        
        input_data = self.data['initial'].T
        self.m0 = input_data[-1][0]

        input_data = input_data[:-1] # drop the mass column 
        output_data = self.data['fuel'].squeeze()
        
        input_data, output_data = self.trim(input_data, output_data)
        self.trimmed_data['input'] = input_data
        self.trimmed_data['output'] = output_data 
        
        if N and N <= len(output_data):
            print("Reducing {} samples to {}".format(len(output_data), N))
            keep = np.random.choice(output_data.size, N, replace=False)
            
            self.model_indices = keep
            input_data = input_data[:,keep]
            output_data = output_data[keep]
            
        self.model_data['input'] = input_data
        self.model_data['output'] = output_data 

        self.mmin = np.min(output_data)
        self.bounds = np.array([np.min(input_data, axis=1), np.max(input_data, axis=1)])
        
        print("Building SRP propellant model from data with {} samples...".format(output_data.shape[0]))
        print("    bounds = {}".format(self.bounds.T))
        t0 = time()
        self.model = Rbf(*input_data, output_data, **rbf_kw)
#         from scipy.interpolate import LinearNDInterpolator 
#         self.model = LinearNDInterpolator(input_data.T, output_data, rescale=False) # , fill_value=8500
#         self.model = NearestNDInterpolator(input_data.T, output_data, rescale=False)
#         from sklearn.svm import LinearSVR as SVR
#         self.model = SVR(max_iter=10000)
#         self.model.fit(X=input_data.T, y=output_data)
        print("Complete: {:.1f} s".format(time()-t0))
        

    def check_bounds(self, state):
        pass 
    
    def plot(self, resolution=(10,10)):
        """Show some diagnostic plots like
        0 crossrange/velocity, various downrange distances at fixed velocities/fpas?"""
        
        plot_contours = resolution[0]*resolution[1] < 10
        
        input_matrix = {"cartesian": [], "entry": []}
        output_matrix = []
        for Vf in np.linspace(500, 700, resolution[0]):
            for fpa in np.linspace(-35, 0, resolution[1]):
                input_matrix['entry'].append([Vf, fpa])
                
                fpa = np.radians(fpa)
                Vx = -Vf*np.cos(fpa)
                Vz = Vf*np.sin(fpa)
                input_matrix['cartesian'].append([-Vx, Vz])
                N = 50
                x = boxgrid([(8000, self.bounds[1][0]), (0,0), (3000, self.bounds[1][2]), (Vx, Vx), (Vz, Vz)], [N,1,N,1,1], True) # Vary only DR and Altitude 
                mf = self(x)
#                 tf = self.time_of_flight(x)
                imin = np.argmin(mf)
                output_matrix.append([mf[imin], x[imin,0]/1000, x[imin,2]/1000]) #, tf[imin]

                if plot_contours:
                    plt.figure()
#                     plt.scatter(x.T[0], x.T[2], c=mf)
                    plt.tricontourf(x.T[0], x.T[2], mf)
                    plt.plot(x[imin,0], x[imin,2], 'rx', markersize=8, label="Optimal {:.1f} kg at\n{:.3f} DR, {:.3f} altitude, km".format(mf[imin], x[imin,0]/1000, x[imin,2]/1000))
                    plt.xlabel('Range to go (m)')
                    plt.ylabel('Altitude (m)')
                    plt.title("Vf = {:.1f}, fpa = {:.2f}".format(Vf, np.degrees(fpa)))
                    plt.legend()
                    plt.colorbar()
                
        if not plot_contours:
            N = 15
            outs = ['Optimal Fuel Use (kg)','Optimal RTG (km)','Optimal Altitude (km)', 'Optimal Time of Flight (s)']
            output_matrix = np.array(output_matrix).T
            y,x = np.array(input_matrix['entry']).T
            for z, state in zip(output_matrix, outs):
                plt.figure()
#                 plt.scatter(x, y, c=z)
                plt.tricontourf(x,y,z,N)
                plt.ylabel("Vf (m/s)")
                plt.xlabel("FPA (deg)")
                plt.colorbar()
                plt.title(state)

            x,y = np.array(input_matrix['cartesian']).T
            for z, state in zip(output_matrix, outs):
                plt.figure()
#                 plt.scatter(x, y, c=z)
                plt.tricontourf(x,y,z,N)
                plt.xlabel("Horizontal Velocity (m/s)")
                plt.ylabel("Vertical Velocity (m/s)")
                plt.colorbar()
                plt.title(state)
            
        plt.show()
        
    def _obj(self, v, state):
        x = state(v)
        m = self(x)
        if m < self.mmin:
            return 5000
        return m
        
    def _opt(self, bounds, state,):
        from scipy.optimize import minimize_scalar 
        sol = minimize_scalar(self._obj, method='bounded', bounds=bounds, args=(state,))
        return sol.x, sol.fun

    def srp_trim(self, traj, target, vmax=800, default=100000, full_return=False, optimize=False, debug=False):
        """ A method for determining the optimal ignition state along a trajectory 

            default is the value returned when no suitable ignition state is found
            full_return returns a dictionary with:
                terminal entry state, 
                ignition state, 
                propellant required, 
                the entry trajectory clipped at the ignition point

        """
        v = traj[:,3]
        k = v < vmax
        rtg,cr = range_from_entry(traj[k], target.coordinates())
        x_srp = srp_from_entry(traj[k], rtg, cr, target_alt=target.altitude).T # 5 x N
        h = x_srp[2]
        hmin = self.hmin
        # These logic checks greatly reduce the number of points to search over in some cases, resulting in a good speed up 
        maxes = self.bounds[1]
        xmax, ymax, hmax, temp, temp = maxes

        high = np.logical_and(h >= hmin, h <= hmax)
        close = np.logical_and(x_srp[0] <= xmax, x_srp[0] >= 0) 
        close = np.logical_and(close, x_srp[1] <= ymax)   
        high = np.logical_and(close, high)

        if np.any(high):
            if optimize and np.sum(high) > 2: # Use optimization of an interpolation function to find the minimum faster
                vscale = 500
                bounds = np.array([np.min(v[k][high])+0.1,np.max(v[k][high])-0.1])/vscale
                # print(bounds)
                vf, m_opt = self._opt(bounds, interp1d(v[k][high]/vscale, x_srp.T[high], axis=0, bounds_error=True,))
                vf *=  vscale
                # v_srp, m_opt = self._opt([np.min(v[k][high]), vmax], interp1d(v[k][high], x_srp.T[high], axis=0, bounds_error=True, fill_value=(x_srp.T[high][-1], x_srp.T[high][0])), srpdata)
                I = np.argmin(np.abs(v[k][high]-vf))
                # TODO: Check for m_opt < mmin and return default if so 
            else:
                m_srp = self(x_srp.T[high])
                m_srp[m_srp <= self.mmin] = default # Some models return negative values, clip anything lower than what's in the table 
                I = np.argmin(m_srp)
                vf = np.linalg.norm(x_srp.T[high][I][3:])
                m_opt = m_srp[I]

            if full_return:
                mc_srp = {}
                mc_srp['traj'] = traj[v <= vf]
                mc_srp['terminal_state'] = (traj[k][high][I])
                mc_srp['fuel'] = m_opt
                mc_srp['ignition_state'] = (x_srp[:,high][:,I]) # may need a transpose
                return mc_srp 
            return m_opt

        else: # No suitable state was found
            if full_return:
                I = 0
                mc_srp = {}
                mc_srp['traj'] = traj[np.invert(k)]
                mc_srp['terminal_state'] = (traj[k][I])
                mc_srp['fuel'] = (default)
                mc_srp['ignition_state'] = (x_srp[:,I]) # may need a transpose
                return mc_srp
            return default 

    # def time_of_flight(self, state):
    #     try:
    #         return self.tf_model(*np.asarray(state).T)
    #     except ValueError:
    #         return self.tf_model(*np.asarray(state))
        
    def __call__(self, state):
        """ State should be the 5 dimensional SRP ignition state"""
        
        try:
            return self.model(*np.asarray(state)) # for rbf  
#             return self.model(state) # for LinearND 
#             mf = self.model.predict(np.asarray(state).T) # for svm 
#             V = np.sum(state[3:6]**2, axis=0)
#             mf[V>800] = 8500 
#             return mf
        except (ValueError, IndexError):
            return self.model(*np.asarray(state).T)
#             return self.model(np.asarray(state).T) # for LinearND 
#             return self.model.predict(np.asarray(state))

    def optimize(self, state_index, x0):
        """Idea: fix 4 out of the 5 states and perform 1-D
        Generally CR will be zero. Potentially useful to optimize DR for fixed V,gamma and altitude 
        
         """
        pass

def generate_pickle():
    matfile = os.path.join(os.getcwd(), "data\\FuelOptimal\\srp_28k_7200kg.mat")
    pklfile = os.path.join(os.getcwd(), "data\\FuelOptimal\\srp_7200kg.pkl")
    # pklfile = matfile.split('.mat')[0] + ".pkl"
    srpdata = SRPData(matfile, min_alt=3000, max_fuel=3000)
    srpdata.build(28000, rbf_kw={'function': 'linear'})

    import pickle
    pickle.dump(srpdata, open(pklfile, 'wb'))

def test_pickle():
    """ NOTE: To load this elsewhere, you'll need to import SRPData at the module level, not in the function/method """

    import pickle
    srpdata = pickle.load(open(os.path.join(os.getcwd(), "data\\FuelOptimal\\srp_27k_5d.pkl"), 'rb'))
    x = [4000, 0, 2500, -500, -140]
    print(srpdata(x))

def generate_plots():
    import pickle
    srpdata = pickle.load(open(os.path.join(os.getcwd(), "data\\FuelOptimal\\srp_11k_7200kg.pkl"), 'rb'))
    # srpdata.plot((3,3))
    srpdata.plot((15,15))

if __name__ == "__main__":
    generate_pickle()
    # test_pickle()
    # generate_plots()
