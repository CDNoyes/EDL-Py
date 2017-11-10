''' Functionality to build a PCE model out of EDL simulation outputs '''

import chaospy as cp
import numpy as np
from scipy.io import savemat, loadmat
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from functools import partial
import time

from Simulation import Simulation, Cycle, EntrySim
from Triggers import AccelerationTrigger, SRPTrigger
from Uncertainty import getUncertainty
from InitialState import InitialState
from EntryEquations import EDL
from Utils.RK4 import RK4
from Utils import draw
from Utils.submatrix import submatrix
from ParametrizedPlanner import profile
import Parachute
from NMPC import NMPC
# import MPC as mpc
# import Apollo



class PCE(object):

    def __init__(self,verbose=True):
        self.dist = None
        self.resamples = None
        self.IV = None
        self.Xtrue=None
        self.order = None
        self.verbose = verbose

    def sample(self, dist, method='H', N=50, order=2, antithetic=None):
        # Methods: QMC such as Sobol, Halton, Hammersley (for regression based)
                # Quadrature methods
                # Order must be kept relatively low for quadrature methods,
                # even with sparse points.
                # However, it is possible to try higher order fitting with point
                # collocation because the number of samples is an input.
        self.dist = dist
        self.poly = cp.orth_ttr(order, self.dist)
        self.order=order
        self.method=method
        self.is_quad = False
        if method.lower() in ("g","c","e"): # G, C, E
            self.is_quad = True
            sparse = order > 2
            self.samples,self.weights = cp.generate_quadrature(order=order, domain=dist, rule=method,sparse=sparse)

        else: # Point collocation based on MC/QMC sampling, aka regression. R,L,S,H,M are the choices
            self.samples = dist.sample(N,method)
        if self.verbose:
            print "Sampling complete... {}".format(self.samples.shape)


    def change_order(self,order):
        """ Rebuilds the orthogonal polynomials for a new order """
        if self.order == order:
            return
        self.order = order
        self.poly = cp.orth_ttr(order,self.dist)
        evals = self.evals.transpose((2,0,1))

        self.model = cp.fit_regression(self.poly, self.samples, evals,rule='LS')
        if self.verbose:
            print "Order changed, new PCE model built from existing samples."

    def build(self):
        """ Evaluates the EDL sim at the sample points and constructs the PCE model """
        # MSL solution
        switch = [  1.12985487e+02,   1.61527467e+02]
        bank = [1.53352287e+00,   -1.02508346e+00,      4.79475355e-01]
        control = [2.47739391e+00,   1.14726959e-01,   6.88822448e+00]

        # switch = [  40, 1.12985487e+02,   1.61527467e+02]
        # bank = [-0.5, 1.53352287e+00,   -1.02508346e+00,      4.79475355e-01]

        self.sim_inputs = [switch,bank,control]
        t0 = time.time()
        self.evals,U = self.Sim(switch,bank,control)
        t1 = time.time()
        print "Simulation complete ({} s)".format(t1-t0)

        evals = self.evals.transpose((2,0,1))[:,200:,:][:,:,:6] # reshape, and trim any states and/or timesteps that we dont need to generate PCE models for. this saves a lot of time.
        if self.is_quad:
            self.model = cp.fit_quadrature(self.poly, self.samples, self.weights, evals)
        else:
            # evals = self.evals.transpose((2,0,1))
            eval_single = self.evals.transpose((0,2,1)) # iterate over the different energy points
            self.model = cp.fit_regression(self.poly, self.samples, evals,rule='LS')
            # self.models = [cp.fit_regression(self.poly, self.samples, ev) for ev in eval_single]

        t2 = time.time()

        print "PCE model constructed ({} s)".format(t2-t1)

    def eval(self, N=2000, method="H"):
        if self.resamples is None:
            t0 = time.time()
            new_samples = self.dist.sample(N,method)
            t1= time.time()
            self.resamples = new_samples
        self.revals = self.model(*self.resamples)
        t2 = time.time()
        # print "{} new samples generated ({} s)".format(N,t1-t0)
        # print "PCE model sampled ({} s)".format(t2-t1)
        return self.revals

    def coeff_plot(self, index):
        plt.figure()

        keys = self.model[index].keys
        coeff = self.model[index].coeffs()
        if np.abs(coeff[0]) > 1e-3:
            coeff = [c/coeff[0] for c in coeff]
        for c,key in zip(coeff,keys):
            plt.plot(np.sum(key),c,'o')

    def error(self):
        """Generates errors between the sampled values of the model and the true vectorized integration"""
        if self.resamples is None:
            print "Evaluating the model..."
            self.eval()
        if self.Xtrue is None:
            print "Vectorized integration for truth data..."
            Xtrue,Utrue = self.Sim(*self.sim_inputs, samples=self.resamples)
            from scipy.interpolate import interp1d

            self.IV = self.IV[:-41] # Prune the extra energy propagation for the error computations
            Xtrue = interp1d(self.RIV, Xtrue, kind='linear', axis=0, copy=True, bounds_error=None, fill_value=np.nan, assume_sorted=False)(self.IV) # Interpolate the truth data onto the original energy points
            self.Xtrue = Xtrue
        else:
            Xtrue = self.Xtrue

        err = Xtrue-self.revals[:-41]
        mean_err = err.mean(axis=2)
        std_err = err.std(axis=2)
        print err.shape
        print mean_err.shape

        return mean_err, std_err

    def error_plots(self,err):

        save_dir = "data/PCE/"

        mean_err = err[0].T
        std_err = err[1].T
        r,lon,lat,v,fpa,psi,s,m = mean_err
        n_std = 3 # plots the n_std*std_err lines
        dr,dlon,dlat,dv,dfpa,dpsi,ds,dm = std_err*n_std

        rp = 3397e3
        r2d = 180.0/np.pi

        plt.figure(10)
        plt.plot(self.IV,r)
        plt.plot(self.IV,r+dr,'k--')
        plt.plot(self.IV,r-dr,'k--')
        plt.ylabel('Altitude Error (m)')
        plt.xlabel('Energy')
        plt.savefig(save_dir + 'Altitude Error N{} {} Order{}.png'.format(self.samples.shape[1],self.method,self.order))

        plt.figure(11)
        plt.plot(self.IV,v)
        plt.plot(self.IV,v+dv,'k--')
        plt.plot(self.IV,v-dv,'k--')
        plt.ylabel('Velocity Error (m/s)')
        plt.xlabel('Energy')
        plt.savefig(save_dir + 'Velocity Error N{} {} Order{}.png'.format(self.samples.shape[1],self.method,self.order))

        plt.figure(12)
        plt.plot(self.IV,lon*rp)
        plt.plot(self.IV,(lon+dlon)*rp,'k--')
        plt.plot(self.IV,(lon-dlon)*rp,'k--')
        plt.ylabel('Downrange Error (m)')
        plt.xlabel('Energy')
        plt.savefig(save_dir + 'Downrange Error N{} {} Order{}.png'.format(self.samples.shape[1],self.method,self.order))

        plt.figure(13)
        plt.plot(self.IV,lat*rp)
        plt.plot(self.IV,(lat+dlat)*rp,'k--')
        plt.plot(self.IV,(lat-dlat)*rp,'k--')
        plt.ylabel('Crossrange Error (m)')
        plt.xlabel('Energy')
        plt.savefig(save_dir + 'Crossrange Error N{} {} Order{}.png'.format(self.samples.shape[1],self.method,self.order))

        plt.figure(14)
        plt.plot(self.IV,fpa*r2d)
        plt.plot(self.IV,(fpa+dfpa)*r2d,'k--')
        plt.plot(self.IV,(fpa-dfpa)*r2d,'k--')
        plt.ylabel('FPA Error (deg)')
        plt.xlabel('Energy')
        plt.savefig(save_dir + 'FPA Error N{} {} Order{}.png'.format(self.samples.shape[1],self.method,self.order))

    def E(self):
        return cp.E(self.model,self.dist)

    def Std(self):
        return cp.Std(self.model,self.dist)

    def Cov(self):
        return np.array([cp.Cov(model,self.dist) for model in self.models])

    def Sobol(self):
        "Returns the 1st order, 2nd order, and total indices"
        return [cp.Sens_m(self.poly, self.dist),cp.Sens_m2(self.poly, self.dist),cp.Sens_t(self.poly, self.dist)]

    def PlotE(self,bounds=True):
        """ Plots the expected trajectory components of the PCE model. """

        E = self.E()

        xf = Trigger(E, self.lonTarget, minAlt=6e3, maxVel=485)
        i = E.tolist().index(list(xf))
        rp = 3397
        r2d = 180./np.pi
        r,lon,lat,v,fpa,psi,s,m = E.T#[:,:i]
        h = (r-3397e3)/1000

        if bounds:
            P = self.Cov()
            for hi,loni,lati,vi,fpai,psii,Pi in zip(h,lon,lat,v,fpa,psi,P)[::5]:
                # radius is in meters, but we plot in km so we must convert
                Pi[:,0] /= 1000
                Pi[0,:] /= 1000
                draw.cov([vi,hi], submatrix(Pi,[3,0]), ci=[0.99], fignum=3,legend=False)
                # draw.cov([lati*rp,loni*rp], submatrix(Pi,[2,1])*rp**2, ci=[0.99], fignum=4,legend=False)
                # conv = np.array([[1,r2d],[r2d,r2d**2]])
                # draw.cov([vi,fpai*r2d], submatrix(Pi,[3,4])*conv, ci=[0.99], fignum=5,legend=False)
                # draw.cov([psii*r2d,fpai*r2d], submatrix(Pi,[5,4])*r2d**2, ci=[0.99], fignum=5,legend=False)


        plt.figure(3)
        plt.title("Expected trajectory")
        plt.plot(v,h)
        plt.xlabel('Velocity')
        plt.ylabel('Altitude')
        Parachute.Draw(figure=3)

        plt.figure(4)
        plt.title("Expected trajectory")
        plt.plot(lat*rp,lon*rp)
        plt.ylabel('Downrange (km)')
        plt.xlabel('Crossrange (km)')

        plt.figure(5)
        plt.title("Expected trajectory")
        plt.plot(v,fpa*r2d)
        plt.xlabel('Velocity')
        plt.ylabel('FPA (deg)')

    def Sim(self, switches, banks, control_inputs, samples=None):
        """ Performs vectorized integration of the EDL model """
        if samples is None:
            samples = self.samples[:]
        edl = EDL(samples,Energy=True)
        reference_sim = Simulation(cycle=Cycle(1),output=False,**EntrySim(Vf=460))

        bankProfile = lambda t: profile(t, switch=switches, bank=banks,order=2)
        t = np.linspace(0,300,5000)
        # bankProfile = lambda **d: profile(d['time'], switch=switches, bank=banks,order=2)
        from Utils.smooth import smooth
        smoothProfile = smooth(t,bankProfile(t))
        bankProfileSmooth = lambda **d: smoothProfile(d['time'])
        x = InitialState()
        output = reference_sim.run(x,[bankProfileSmooth])
        # reference_sim.plot()
        # reference_sim.show()

        Xf = output[-1,:]
        self.lonTarget = np.radians(Xf[5])

        # Closed loop statistics generation
        refs = reference_sim.getFBL()

        nmpc = NMPC(fbl_ref=refs, debug=False)
        nmpc.dt = control_inputs[0]
        nmpc.Q = np.array([[control_inputs[1],0],[0,control_inputs[2]]])
        optSize = samples.shape[1]
        x = np.tile(x,(optSize,1)).T
        X = [x]
        U = []
        energy0 = edl.energy(x[0],x[3],False)[0]
        energyf = Xf[1]*0.5

        energy = energy0
        E = [energy]
        dt = 1
        while energy > energyf:

            Xc = X[-1]
            energys = edl.energy(Xc[0],Xc[3],False)
            lift,drag = edl.aeroforces(Xc[0],Xc[3],Xc[7])
            u = nmpc.controller(energy=energys, current_state=Xc,lift=lift,drag=drag,rangeToGo=None,planet=edl.planet)
            u.shape = (1,optSize)
            U.append(u)
            u = np.vstack((u,np.zeros((2,optSize))))
            de = -np.mean(drag)*np.mean(Xc[3]) * dt
            if (energy + de) <  energyf:
                de = energyf - energy
            eom = edl.dynamics(u)
            X.append(RK4(eom, X[-1], np.linspace(energy,energy+de,10),())[-1])
            energy += de
            E.append(energy)
            # if len(E)>1000:
            #     break
        if self.IV is None:
            self.IV = np.array(E)
        else:
            self.RIV = np.array(E)

        X = np.array(X)
        U = np.array(U)
        return X,U

def Cost(Xf,lonTarget):
    if not Xf.shape[0] in (6,8):
        Xf = Xf.T
    # h   = edl.altitude(Xf[0], km=True) # altitude, km
    rp = 3397
    DR = Xf[1]*rp # downrange, km
    CR = Xf[2]*rp # -crossrange, km

    J = np.sqrt((DR-lonTarget*3397)**2+CR**2).mean() # Norm squared, to be differentiable for finite differencing
    return J

def Trigger(traj, targetLon, minAlt=0e3, maxVel=600):
    for state in traj:
        alt = state[0]-3397e3
        vel = state[3]
        longitude = state[1]
        if alt < minAlt or (vel<maxVel and longitude>=targetLon):
            return state
    return traj[-1]# No better trigger point so final point is used

def test():
    dist = getUncertainty()['parametric']

    for N in [5,10]:
    # for N in [20,50,100,200,500,10000]:
        for method in  ("S"):
        # for method in  ("S","H","L"):
            pce = PCE(verbose=False)
            # pce.sample(dist, method='E', order=4)
            pce.sample(dist, method=method, N=N, order=2)
            pce.build()
            # import pdb
            # pdb.set_trace()
            # pce.PlotE()

            # t0 = time.time()
            # X = pce.eval(2000,'S')
            # t1 = time.time()

            Xf = np.array([Trigger(traj, pce.lonTarget, minAlt=6e3, maxVel=485) for traj in pce.evals.transpose((2,0,1))]).T # Parachute deployment
            J = Cost(Xf,pce.lonTarget)
            print "Cost with {} samples = {}".format(N,J)

            # t2 = time.time()
            # print "Trigger logic: {} s".format(t2-t1)

            # Plot
            for order in [2,3,4,5]: # build different order models from the same sampled data
                pce.change_order(order)
                X = pce.eval(50000,'S')
                # print X.shape
                Xf = np.array([Trigger(traj, pce.lonTarget, minAlt=6e3, maxVel=485) for traj in X.transpose((2,0,1))]).T # Parachute deployment
                J = Cost(Xf,pce.lonTarget)
                print "Cost with {} samples and {} order PCE = {} ".format(N,order,J)
                # pce.error_plots(pce.error())
                # plt.close("all")



    # Xi = Xf
    # Pf = np.cov(Xf)
    # Pf[:,0] /= 1000
    # Pf[0,:] /= 1000

    # h   = (Xi[0]-3397e3)/1000

    # plt.figure(4)
    # plt.scatter(Xi[2]*3397,Xi[1]*3397,c=h)
    # draw.cov([Xi[2].mean()*3397,Xi[1].mean()*3397], submatrix(Pf,[2,1])*3397**2, ci=[0.99], fignum=4, show=False, legend=False, xlabel='', ylabel='', legtext=None, linespec=None)
    #
    # # theta = np.linspace(0,2*np.pi,100)
    # # x = np.cos(theta)
    # # y = np.sin(theta)
    # # for r in [1,2,10]:
    # #     plt.plot(x*r,pce.lonTarget*3397 + y*r,label="{} km".format(r))
    # # plt.legend()
    # plt.xlabel('Crossrange (km)')
    # plt.ylabel('Downrange (km)')
    # plt.colorbar()
    # plt.axis('equal')
    #
    # plt.figure(3)
    # plt.plot(Xi[3],h,'o')
    # draw.cov([Xi[3].mean(),h.mean()], submatrix(Pf,[3,0]), ci=[0.99], fignum=3, show=False, legend=False, xlabel='', ylabel='', legtext=None, linespec=None)
    # plt.xlabel('Velocity (m/s)')
    # plt.ylabel('Altitude (km)')

    # Parachute.Draw(figure=2)

    # plt.show()

if __name__ == "__main__":
    # optimize()
    test()
