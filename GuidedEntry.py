import numpy as np
from scipy.integrate import odeint, trapz
from scipy import linalg
from scipy.interpolate import interp1d
from scipy.io import savemat, loadmat

from Utils.redirect import stdout_redirected
from Utils.loggingUtils import loggingInit,loggingFinish
loggingInit()

from EntryGuidance.EntryEquations import Entry
from EntryGuidance.Planet import Planet
from EntryGuidance.EntryVehicle import EntryVehicle
from EntryGuidance.Triggers import DeployParachute, findTriggerPoint, AltitudeTrigger, VelocityTrigger
from EntryGuidance.ParametrizedPlanner import Optimize, HEPBankReducedSmooth, OptimizeSmooth

from EntryGuidance.Simulation import Simulation, SRP

def Simulate(sample=None):
    if sample is not None:
        CD,CL,rho0,sh = sample
        entry = Entry(PlanetModel = Planet(rho0=rho0,scaleHeight=sh), VehicleModel = EntryVehicle(CD=CD,CL=CL),Trigger=VelocityTrigger(1200))

    else:
        entry = Entry(Trigger=DeployParachute)
    r0, theta0, phi0, v0, gamma0, psi0,s0 = (3540.0e3, np.radians(-90.07), np.radians(-43.90),
                                             5505.0,   np.radians(-14.15), np.radians(4.99),   780e3)
    x0 = np.array([r0, theta0, phi0, v0, gamma0, psi0, s0, entry.vehicle.mass])
    time = np.linspace(0,500,1500)

    n = 2
    # hep,ts = Optimize(x0,n)
    # hep,sol = OptimizeSmooth(x0)
    # return sol.fun
    # ts = sol.x
    hep = lambda x,t: HEPBankReducedSmooth(t,106,133)
    ts = np.zeros(3)
    # ts = np.hstack((np.zeros(3-n),ts))

    with stdout_redirected():    
        X = odeint(entry.dynamics(hep), x0, time)
    
    idx = findTriggerPoint(X,time)
    istart = np.argmax(X[0:idx,3])
    range = [entry.planet.range(*x0[[1,2,5]],lonc=np.radians(lon),latc=np.radians(lat),km=True) for lon,lat in zip(X[0:idx,1],X[0:idx,2])]

    # #In reality, need to compute the range to go to target. The initial condition in entry is set to the range to ignition?
    # entry.ignite(AltitudeTrigger(0.0)) # Initializes the srp phase and sets a condition for terminating the simulation
    
    
    
    energy = entry.energy(X[istart:idx,0],X[istart:idx,3])
    eInterp = np.linspace(0,1,1000)
    tInterp = interp1d(energy,time[istart:idx],'cubic')(eInterp)
    xInterp = interp1d(energy,X[istart:idx,:],'cubic',axis=0)(eInterp)
    return xInterp



# energy = entry.energy(X[0:idx,0],X[0:idx,3])
# eSwitch = interp1d(time[0:idx],energy,'cubic')(ts)        
    
# r,theta,phi,v,gamma,psi,s = X[0:idx,0], np.degrees(X[0:idx,1]), np.degrees(X[0:idx,2]), X[0:idx,3], np.degrees(X[0:idx,4]), np.degrees(X[0:idx,5]), (s0-X[0:idx,6])/1000
# h = [entry.altitude(R,km=True) for R in r]
# bank = [np.degrees(hep(xx,tt)) for xx,tt in zip(X[0:idx,:],time[0:idx])]
# range = [entry.planet.range(*x0[[1,2,5]],lonc=np.radians(lon),latc=np.radians(lat),km=True) for lon,lat in zip(theta,phi)]
# energy = energy[0:idx] #entry.energy(r,v)
# L,D = entry.aeroforces(r,v)
# Dbar = trapz(D,energy)
# Lbar = trapz(L,energy)


# loggingFinish(states = ['time','energy','bank','altitude','radius','longitude','latitude','velocity','fpa', 'heading', 'DR', 'CR','lift', 'drag'],
              # units =  ['s',   '-',     'deg', 'km',      'm',     'deg',      'deg',     'm/s',     'deg', 'deg',     'km', 'km','m/s^2','m/s^2'], 
              # data = np.c_[time[0:idx], energy, bank, h,   r,      theta,       phi,      v,         gamma, psi,       range,     L,      D],
              # summary = ([('state','-','trajSummary'),
                         # ('lbar','m/s^2',Lbar),
                         # ('dbar','m/s^2',Dbar),
                         # ('lodbar','-',Lbar/Dbar),
                         # ('ts1','s',ts[0]),('ts2','s',ts[1]),('ts3','s',ts[2]),
                         # ('es1','-',eSwitch[0]),('es2','-',eSwitch[1]),('es3','-',eSwitch[2])
                         # ]))

if __name__ == '__main__':
    from argparse import ArgumentParser
    import multiprocessing as mp
    import chaospy as cp
    import os

    # Parse Arguments and Setup Pool Environment
    mp.freeze_support()
    # pool = mp.Pool(mp.cpu_count()/2.)
    pool = mp.Pool(4)
    parser = ArgumentParser('Guided Entry Simulation')
    parser.add_argument('--type',   type=str,           default='baseline', help='Type of simulation(s) to run, [baseline, mc, qmc, pce]')
    parser.add_argument('--no_save',action='store_true',                    help='Flag to turn off saving the results.')
    parser.add_argument('--n',      type=int,           default=1,          help='Number of monte carlo cases to run, integer')
    parser.add_argument('--name',   type=str,           default=None,          help='Filename to save, string [optional]')
    parser.add_argument('--dir',    type=str,           default=None,          help='Directory in which to save with no leading or trailing slashes, string [optional]')

    
    args = parser.parse_known_args()[0]
    print "Running {0} simulation".format(args.type)
    
    n = args.n;
    if args.dir is not None:
        saveDir = './data/{0}/'.format(args.dir)
    else:
        saveDir = './data/temp/'
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)        
        
    # Define Uncertainty Joint PDF
    CD          = cp.Uniform(-0.10, 0.10)   # CD
    CL          = cp.Uniform(-0.10, 0.10)   # CL
    rho0        = cp.Normal(0, 0.0333)      # rho0
    scaleHeight = cp.Uniform(-0.05,0.05)    # scaleheight
    pdf         = cp.J(CD,CL,rho0,scaleHeight)
    
    

    
    if args.type == 'baseline':
        #single case
        sample = np.array([0]*4)
        states = Simulate(sample)
        index = {'r':0,'lon':1, 'lat':2,'v':3,'fpa':4,'heading':5,'rtg':6}
        p = pdf.pdf(sample)
        if not args.no_save:
            savemat(saveDir+'Baseline',{'states':states, 'samples':sample,'index':index})

    elif args.type == 'mc':
        samples = pdf.sample(n)    
        p = pdf.pdf(samples)

        stateTensor = pool.map(Simulate,samples.T)
        savemat(saveDir+'MC',{'states':stateTensor, 'samples':samples})

    else: # Polynomial Chaos Expansion !!!!
        if args.type == 'qmc':
            #Quasi MonteCarlo with 1/4 number of MC samples and PCE built from it
            samples = pdf.sample(n,'S')
            stateTensor = pool.map(Simulate,samples.T)
            if not args.no_save:
                savemat(saveDir+'SobolMC',{'states':stateTensor, 'samples':samples})
            
            # data = loadmat('./data/SobolMC')
            # samples = data['samples']
            # stateTensor = data['states']
            polynomials = cp.orth_ttr(order=2, dist=pdf)
            PCE = cp.fit_regression(polynomials, samples, stateTensor)
        elif args.type == 'pce':
            #Quadrature based PCE
            polynomials = cp.orth_ttr(order=2, dist=pdf)
            samples,weights = cp.generate_quadrature(order=2, domain=pdf, rule="Gaussian")
            stateTensor = pool.map(Simulate,samples.T)
            PCE = cp.fit_quadrature(polynomials,samples,weights,stateTensor)
            
        data = loadmat(saveDir+'MC') #Load the MC samples for an apples-to-apples comparison
        pceTestPoints = data['samples']    
        stateTensorPCE = np.array([PCE(*point) for point in pceTestPoints.T])
        Expectation = cp.E(poly=PCE,dist=pdf)
        if not args.no_save:      
            savemat(saveDir+'PCE2',{'states':stateTensorPCE, 'samples':pceTestPoints,'mean':Expectation})
        # savemat('./data/PCE2',{'states':stateTensorPCE[:,:,:,0], 'samples':pceTestPoints,'mean':Expectation})
    
    
    
    