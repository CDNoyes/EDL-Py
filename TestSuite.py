''' Defines tests '''
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import chaospy as cp

from EntryGuidance.EntryEquations import Entry, System, EDL
from EntryGuidance.Uncertainty import getUncertainty

def testFilters():

    perturb = getUncertainty()['parametric']
    sample = perturb.sample()
    
    sample[2] = 0
    sample[3] = 0
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
    # for i,x in enumerate(X0):
        # print "state {}: {}".format(i,x)
    
    print "delta CL: {}".format(sample[1])
    print "delta CD: {}".format(sample[0])
    
    plt.figure()
    plt.plot(time,X[:,16],label = 'RL')
    plt.plot(time,X[:,17],label = 'RD')
    plt.plot(time,(1+sample[1])*np.ones_like(time),label = 'RL true')
    plt.plot(time,(1+sample[0])*np.ones_like(time),label = 'RD true')
    
    plt.legend(loc='best')
    
    plt.figure()
    plt.plot(X[:,3],X[:,0])
    plt.plot(X[:,11],X[:,8])
    
    plt.show()
    
    return

def testCuba():    
    from cubature import cubature as cuba


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