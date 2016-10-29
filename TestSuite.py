''' Defines tests '''
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint, trapz
from scipy.optimize import minimize, differential_evolution
import chaospy as cp
from cubature import cubature as cuba

from EntryGuidance.EntryEquations import Entry, System, EDL
from EntryGuidance.Uncertainty import getUncertainty

def Optimize():
    ''' Optimizes filter gain of a 1st order fading memory filter in an RSOCP formulation '''
    
    
    
    perturb = getUncertainty()['parametric']

    bounds = [(0,1)]
    
    sol = differential_evolution(OptCostRS, args = (pdf, system),bounds=bounds, tol=1e-1, disp=True)
    
    
    
def OptCost(sample, gain):
    ''' Standard cost function. For a fixed sample we can optimize the gain '''
    
    system = System(sample)
    system.setFilterGain(gain)
    r0, theta0, phi0, v0, gamma0, psi0,s0 = (3540.0e3, np.radians(-90.07), np.radians(-43.90),
                                             5505.0,   np.radians(-14.15), np.radians(4.99),   780e3)
                                             
                                             
    x0_true = np.array([r0, theta0, phi0, v0, gamma0, psi0, s0, system.truth.vehicle.mass])
    
    x0_nav = np.array([r0, theta0, phi0, v0, gamma0, psi0, s0, system.nav.vehicle.mass])
    
    RL = 1.0
    RD = 1.0
    
    X0 = np.hstack((x0_true, x0_nav, RL, RD))
        
    time = np.linspace(0,250,1500)
    
    u = 0,0,0
    
    X = odeint(system.dynamics(u), X0, time)
    
    Ltrue,Dtrue     = system.truth.aeroforces(X[:,8],X[:,11])
    Lmodel,Dmodel   = system.model.aeroforces(X[:,8], X[:,11])
    L = Lmodel*X[:,16]
    D = Dmodel*X[:,17]
    err = (D-Dtrue)**2

    return trapz(err, time) 
    
def OptCostRS(gain, pdf):

    polynomials = cp.orth_ttr(order=2, dist=pdf)
    samples,weights = cp.generate_quadrature(order=2, domain=pdf, rule="Gaussian")
    stateTensor = [OptCost(s,gain) for s in samples.T]
    # stateTensor = pool.map(OptCost,samples.T)
    PCE = cp.fit_quadrature(polynomials,samples,weights,stateTensor)
    
    x0 = np.array([-0.10,-0.10,-0.2,-0.05])
    xf = np.array([0.10,0.10,0.2,0.05])
    P,err = cuba(PCE,ndim=4,fdim=1,xmin=x0,xmax=xf,vectorized=True, adaptive='p')
    
    return P

def testFilters(sample=None):
    if sample is None:
        perturb = getUncertainty()['parametric']
        sample = perturb.sample()
    
    # sample[2] = 0
    # sample[3] = 0
    # sample = [s*10 for s in sample]
    print sample
    system = System(sample)
    
    r0, theta0, phi0, v0, gamma0, psi0,s0 = (3540.0e3, np.radians(-90.07), np.radians(-43.90),
                                             5505.0,   np.radians(-14.15), np.radians(4.99),   780e3)
                                             
                                             
    x0_true = np.array([r0, theta0, phi0, v0, gamma0, psi0, s0, system.truth.vehicle.mass])
    
    x0_nav = np.array([r0, theta0, phi0, v0, gamma0, psi0, s0, system.nav.vehicle.mass])
    
    RL = 1.0
    RD = 1.0
    
    X0 = np.hstack((x0_true, x0_nav, RL, RD))
        
    time = np.linspace(0,250,1500)
    
    u = 0,0,0
    
    X = odeint(system.dynamics(u), X0, time)
    
    Ltrue,Dtrue     = system.truth.aeroforces(X[:,8],X[:,11])
    Lmodel,Dmodel   = system.model.aeroforces(X[:,8], X[:,11])
    L = Lmodel*X[:,16]
    D = Dmodel*X[:,17]
    # for i,x in enumerate(X0):
        # print "state {}: {}".format(i,x)
    
    # print "delta CL: {}".format(sample[1])
    # print "delta CD: {}".format(sample[0])
    
    # plt.figure()
    # plt.plot(time,X[:,16],label = 'RL')
    # plt.plot(time,X[:,17],label = 'RD')
    # plt.plot(time,(1+sample[1])*np.ones_like(time),label = 'RL true')
    # plt.plot(time,(1+sample[0])*np.ones_like(time),label = 'RD true')
    
    # plt.legend(loc='best')
    
    plt.figure()
    plt.plot(time,Ltrue, label='Lift, Truth Model')
    plt.plot(time,L,'--',label='Lift model corrected')
    plt.plot(time,Lmodel,label='Uncorrected model')
    # plt.plot(time,Dtrue, label='Drag, Truth Model')
    # plt.plot(time,D,'--',label='Drag model corrected by filter')
    plt.legend(loc='best')

    # plt.figure()
    # plt.plot(X[:,3],X[:,0])
    # plt.plot(X[:,11],X[:,8])
    
    plt.show()
    
    return

def testCuba():    
    # from cubature import cubature as cuba


    CD          = cp.Uniform(-0.10, 0.10)   # CD
    CL          = cp.Uniform(-0.10, 0.10)   # CL
    rho0        = cp.Normal(0, 0.0333)      # rho0
    scaleHeight = cp.Uniform(-0.05,0.05)    # scaleheight
    pdf = cp.J(CD,CL,rho0,scaleHeight)

    def PDF(x,*args,**kwargs):
        return pdf.pdf(np.array(x).T)

    x0 = np.array([-0.10,-0.10,-0.2,-0.05])
    xf = np.array([0.10,0.10,0.2,0.05])
    P,err = cuba(PDF,ndim=4,fdim=1,xmin=x0,xmax=xf,vectorized=True, adaptive='p')

    print P
    print err
    
if __name__ == '__main__':
    perturb = getUncertainty()['parametric']
    sample = perturb.sample()
    
    # print OptCost(sample)
    print OptCostRS(1, perturb)
    # testFilters(sample)