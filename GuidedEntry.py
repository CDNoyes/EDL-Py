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
from EntryGuidance.Triggers import BiasDeployParachute, DeployParachute, findTriggerPoint, AltitudeTrigger
from EntryGuidance.HighElevationPlanner import Optimize, HEPBankReducedSmooth, OptimizeSmooth
#Consider using an fsm

def Simulate(sample):
    CD,CL,rho0,sh = sample
    entry = Entry(PlanetModel = Planet(rho0=rho0,scaleHeight=sh), VehicleModel = EntryVehicle(CD=CD,CL=CL),Trigger=BiasDeployParachute)
    # entryNom = Entry()
    r0, theta0, phi0, v0, gamma0, psi0,s0 = (3540.0e3, np.radians(-90.07), np.radians(-43.90),
                                             5505.0,   np.radians(-14.15), np.radians(4.99),   780e3)
    x0 = np.array([r0, theta0, phi0, v0, gamma0, psi0, s0])
    # x0Nom = np.array([r0, theta0, phi0, v0, gamma0, psi0, s0])
    time = np.linspace(0,500,1500)

    n = 2
    # hep,ts = Optimize(x0,n)
    hep,ts = OptimizeSmooth(x0)
    # hep = lambda x,t: HEPBankReducedSmooth(t,105,135)
    # ts = np.zeros(3)
    # ts = np.hstack((np.zeros(3-n),ts))

    with stdout_redirected():    
        X = odeint(entry.dynamics(hep), x0, time)

    idx = findTriggerPoint(X,time)
    
    istart = np.argmax(X[0:idx,3])
    # range = [entry.planet.range(*x0[[1,2,5]],lonc=np.radians(lon),latc=np.radians(lat),km=True) for lon,lat in zip(X[0:idx,1],X[0:idx,2])]

    # tInterp = np.linspace(0,time[idx-1],500)
    energy = entry.energy(X[istart:idx,0],X[istart:idx,3])
    eInterp = np.linspace(0,1,1000)
    
    # xInterp = interp1d(time[0:idx],X[0:idx,:],'cubic',axis=0)(tInterp)
    xInterp = interp1d(energy,X[istart:idx,:],'cubic',axis=0)(eInterp)
    # xInterp = interp1d(range,X[0:idx,:],'cubic',axis=0)(rInterp)
    # xInterp = interp1d(-X[0:idx,3],X[0:idx,:],'cubic',axis=0)(vInterp)
    return xInterp
    # return X[0:idx,:]




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
    import multiprocessing as mp
    import chaospy as cp


    CD          = cp.Uniform(-0.10, 0.10)   # CD
    CL          = cp.Uniform(-0.10, 0.10)   # CL
    rho0        = cp.Normal(0, 0.0333)      # rho0
    scaleHeight = cp.Uniform(-0.05,0.05)    # scaleheight
    pdf = cp.J(CD,CL,rho0,scaleHeight)
    mp.freeze_support()
    # pool = mp.Pool(mp.cpu_count()/2.)
    pool = mp.Pool(2)
    
    if 1:
        #single case
        sample = np.array([0]*4)
        states = Simulate(sample)
        index = {'r':0,'lon':1, 'lat':2,'v':3,'fpa':4,'heading':5,'rtg':6}
        savemat('./data/Baseline',{'states':states, 'samples':sample,'index':index})

    elif 0: #RS Monte Carlo
        samples = pdf.sample(1200)    

        stateTensor = pool.map(Simulate,samples.T)
        savemat('./data/MC',{'states':stateTensor, 'samples':samples})

    else: # Polynomial Chaos Expansion !!!!
        if 0:
            #Quasi MonteCarlo with 1/4 number of MC samples and PCE built from it
            samples = pdf.sample(300,'S')
            stateTensor = pool.map(Simulate,samples.T)
            savemat('./data/SobolMC',{'states':stateTensor, 'samples':samples})
            
            # data = loadmat('./data/SobolMC')
            # samples = data['samples']
            # stateTensor = data['states']
            polynomials = cp.orth_ttr(order=2, dist=pdf)
            PCE = cp.fit_regression(polynomials, samples, stateTensor)
        else:
            #Quadrature based PCE
            polynomials = cp.orth_ttr(order=2, dist=pdf)
            samples,weights = cp.generate_quadrature(order=2, domain=pdf, rule="Gaussian")
            stateTensor = pool.map(Simulate,samples.T)
            PCE = cp.fit_quadrature(polynomials,samples,weights,stateTensor)
            
        data = loadmat('./data/MC') #Load the MC samples for an apples-to-apples comparison
        pceTestPoints = data['samples']    
        # pceTestPoints = pdf.sample(1200)
        stateTensorPCE = np.array([PCE(*point) for point in pceTestPoints.T])
        Expectation = cp.E(poly=PCE,dist=pdf)
        savemat('./data/PCE2',{'states':stateTensorPCE, 'samples':pceTestPoints,'mean':Expectation})
        # savemat('./data/PCE2',{'states':stateTensorPCE[:,:,:,0], 'samples':pceTestPoints,'mean':Expectation})
    
    
    
    