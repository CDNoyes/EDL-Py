import numpy as np
import chaospy as cp
from scipy.integrate import odeint
from functools import partial
import matplotlib.pyplot as plt

from pyaudi import abs, sqrt
# from numpy import abs, sqrt
from EntryEquations import Entry, EDL
from Triggers import DeployParachute, findTriggerPoint
from InitialState import InitialState
from Utils import DA as da 
# from HPC import profile

def profile(T, switch, bank, order=1):
    """ Returns the bank angle for a profile at time T given N switch points and N+1 bank angles with smoothness=order """
    
    if (len(switch)+1) != len(bank):
        print "Improper inputs to bank profile"
        return 0 
        
    # DO THIS ONLY IF USING LINEAR PROFILE
    if order == 1:
        maxRate=np.radians(20)
        newswitch = []
        newbank = [bank[0]]
        cbank = da.const(bank)
        for i,s in enumerate(switch): 
            newswitch.extend( [s, s + abs(cbank[i+1]-cbank[i])/maxRate] ) # Can use bank instead of cbank to get dependence on bank inputs but we cannot use it in a comparison against time
            newbank.extend([bank[i+1],bank[i+1]])
        bank=newbank 
        switch=newswitch
    # #########################################
    # cswitch = da.const(switch) # Constant portion for comparing with time 
    
    n = len(bank)
    bank_out = []
    try:
        T[0]
        isScalar = False
    except:
        T = [T]
        isScalar = True
        
    for t in T:    
        if t <= switch[0]:
            bank_out.append(bank[0])
        else:
            for i,s in enumerate(switch[::-1]):
                if t>s:
                    tswitch = s #switch[n-2-i]
                    bankcur = bank[n-2-i]
                    banknew = bank[n-1-i]
                    break
            if order == 2:
                bank_out.append(maneuver(t,tswitch,bankcur,banknew))                      # Use this for smooth maneuvers. Doesn't work properly with DA variables for some reason.
            elif order == 1:    
                bank_out.append(bankcur + (t-tswitch)*maxRate*da.sign(banknew-bankcur))   # Results in piecewise linear, continous maneuvers
            elif order == 0:
                bank_out.append(banknew)                                                  # Results in discontinuous maneuvers
            
    if isScalar:
        return bank_out[0]
    else:
        return bank_out
    
def maneuver(t, t0, bank_current, bank_desired, maxRate=np.radians(20), maxAcc=np.radians(5)):
    """ Optimal maneuver from one bank angle to another subject to constraint on rate and acceleration. """
    
    dt = maxRate/maxAcc                                         # Amount of time to go from 0 bank rate to max or vice versa
    tm = sqrt(abs(bank_current-bank_desired)/maxAcc)            # Amount of time to reach the midpoint of the maneuver assuming max acc the whole time
    
    if tm <= dt:                                                # No max rate because the angles are too close together
        t1a = t0 + tm
        t1v = t1a                                               # Never enter the middle phase 
        t1d = t1a + tm
        dbank = abs(bank_current-bank_desired)/2
        maxRate = maxAcc*tm                                     # The maximum rate achieved during the maneuver
    else:
        dbank = 0.5*(maxRate**2)/maxAcc                         # Angle traversed during max accel for dt seconds
        
        t1a = t0 + dt
        t1v = t0 + abs(bank_current-bank_desired)/maxRate
        t1d = t1v + dt
    
    # s = np.sign(bank_desired-bank_current)
    s = (bank_desired-bank_current)/abs(bank_desired-bank_current)

    if t >= t0 and t <= t1a:                                                # Max acceleration phase
        bank = (bank_current + 0.5*s*maxAcc*(t-t0)**2)
        
    elif t > t1a and t <= t1v:                                              # Max velocity phase ( if present )
        bank = (bank_current + s*dbank + s*maxRate*(t-t1a))
        
    elif t > t1v and t <= t1d:
        bank = (bank_current + s*dbank + s*maxRate*(t-t1a) - s*0.5*maxAcc*(t-t1v)**2 ) # Max deceleration
        
    elif t > t1d:
        bank = (bank_desired)                                               # Post-arrival
    
    return bank
        
def OptimizeSRP():
    """ Not named very well. Computes the three switch bank profile that delivers the entry vehicle to a prescribed altitude, downrange, crossrange at a given velocity trigger point. """
    
    from scipy.optimize import differential_evolution, minimize
    from Simulation import Simulation, Cycle, EntrySim

    sim = Simulation(cycle=Cycle(1),output=False,**EntrySim())
    x0 = InitialState()

    if 0:
        # bounds = [(0,250),(100,500)]
        bounds = [(0,250),(100,400),(100,550)]
        sol = differential_evolution(SRPCost,args = (sim,x0), bounds=bounds, tol=1e-1, disp=True, polish=False)
    else:
        sol = minimize(SRPCost, [68.5, 128.3, 189.2], args=(sim,x0), method='Nelder-Mead', tol=1e-2, options={'disp':True})
        
    bankProfile = lambda **d: profile(d['time'], switch=sol.x, bank=[np.radians(-30),np.radians(75),np.radians(-75),np.radians(30)],order=2)
    
    print sol.x
    output = sim.run(x0,[bankProfile])
    
    sim.plot()
    sim.show()
    
    return sim,sol 
    
def checkFeasibility(T,sign=-1):
    
    sig = sign*np.diff(T) 
    cost = (sum(sig+np.abs(sig)) + sum(np.abs(T)-T))*1e5
    if cost:
        return max(cost,1e7)
    else:
        return 0    
    
def SRPCost(p, sim, x0, sample=None):

    dr_target = 900
    cr_target = 0

    
    J = checkFeasibility(p)
    if J > 300:
        return J
    
    bankProfile = lambda **d: profile(d['time'], switch=p, bank=[np.radians(-30),np.radians(75),np.radians(-75),np.radians(30)],order=2)
                                                
    output = sim.run(x0,[bankProfile],sample)

    Xf = output[-1,:]
    # data = np.c_[self.times, energy, bank, h,   r,      theta,       phi,      v,         gamma, psi,       range,     L,      D]
    hf = Xf[3]
    # fpaf = Xf[8]
    dr = Xf[10]
    cr = Xf[11]
    
    J = -hf + (0*(dr_target-dr)**2 + 1*(cr_target-cr)**2)**0.5 

    return J

def OptimizeSRPRS():
    from Uncertainty import getUncertainty
    from Simulation import Simulation, Cycle, EntrySim
    from scipy.optimize import differential_evolution, minimize
    
    sim = Simulation(cycle=Cycle(1),output=False,**EntrySim()) 
    perturb = getUncertainty()['parametric']
    if 0:
        sol = minimize(SRPCostRS, [ 165.4159422 ,  308.86420218,  399.53393904], args=(sim, perturb), method='Nelder-Mead', tol=1e-2, options={'disp':True})
    else:
        bounds = [(0,250),(100,400),(250,500)]
        sol = differential_evolution(SRPCostRS, args=(sim, perturb, x0), bounds=bounds, tol=1e-3, disp=True, polish=False)
    
    print sol.x
    
    bankProfile = lambda **d: HEPBank(d['time'], *(sol.x))
    
    r0, theta0, phi0, v0, gamma0, psi0, s0 = (3540.0e3, np.radians(-90.07), np.radians(-43.90),
                                             5505.0,   np.radians(-14.15), np.radians(4.99),   1000e3)
                                             
    x0 = np.array([r0, theta0, phi0, v0, gamma0, psi0, s0, 8500.0])
    output = sim.run(x0,[bankProfile])
    
    sim.plot()
    sim.show()
    
    return # Optimal came out to be 
    # [ 162.43219422  294.49990189  429.22841173] using Nelder-Mead with nominal optimal as guess and 
    # [ 162.4368125   294.35875742  461.31910219] with differential evolution, each with a cost of 43.8 roughly. Nominal solution has a cost of 44.6
    
def SRPCostRS(p, sim, pdf, x0):

    polynomials = cp.orth_ttr(order=2, dist=pdf)
    samples,weights = cp.generate_quadrature(order=2, domain=pdf, rule="Gaussian")
    stateTensor = [SRPCost(p, sim, x0, s) for s in samples.T]
    # stateTensor = pool.map(OptCost,samples.T)
    PCE = cp.fit_quadrature(polynomials,samples,weights,stateTensor)
    
    # print "PCE Expectation: {} ".format(cp.E(poly=PCE,dist=pdf))
    return cp.E(poly=PCE,dist=pdf)
    
    
def ExpandBank():
    
    hep = HEPBankReducedSmooth
    
    t = np.linspace(0,350,3500)
    t1 = cp.Uniform(0,150)
    t2 = cp.Uniform(100, 260)
    
    t1 = cp.Normal(70,1)
    t2 = cp.Normal(115,1)
    pdf = cp.J(t1,t2)
    polynomials = cp.orth_ttr(order=2, dist=pdf) #No good for dependent
    # polynomials = cp.orth_bert(N=2,dist=pdf) 
    # polynomials = cp.orth_gs(order=2,dist=pdf) 
    # polynomials = cp.orth_chol(order=2,dist=pdf) 
    
    if 0:
        nodes, weights = cp.generate_quadrature(order=3, domain=pdf, rule="Gaussian")
        # nodes, weights = cp.generate_quadrature(order=2, domain=pdf, rule="C")
        # nodes, weights = cp.generate_quadrature(order=9, domain=pdf, rule="L")
        print nodes.shape
        samples = np.array([hep(t,*node) for node in nodes.T])
        hepPCE = cp.fit_quadrature(polynomials,nodes,weights,samples)
    else:
        nodes = pdf.sample(4,'S')
        samples = np.array([hep(t,*node) for node in nodes.T])
        hepPCE = cp.fit_regression(polynomials, nodes, samples,rule='T')
    return hepPCE


    
    # TODO: evaluate expansions at various design points based on gradient and hessian info
    
def testExpansion():
    '''
    Conclusions from running this over and over with different inputs:
        Order is very important. This problem works best by far with order 2
        Polynomial type had very little impact
        Quadrature performed best with Gaussian, order must >= polynomial order. The order of quadrature (and number of uncertain inputs) determines how many samples you will have to run.
        Collocation worked best with T(ikhonov Regularization), and had similar performance with quadrature for a similar number of evaluations. 
        However, the 1-D test in TestSuite.py seems to show clearly superior results using quadrature, perhaps because it is only 1-D and very smooth.

    '''
    import matplotlib.pyplot as plt
    model = ExpandBank()
    hep = lambda t: HEPBankReducedSmooth(t,71,113)

    t = np.linspace(0,350,3500)
    ti = np.linspace(0,350,3500)
    test = model(*[71,113])
    
    plt.plot(ti,hep(ti))  
    plt.plot(t,test)
    plt.show()

if __name__ == '__main__':
    # from Uncertainty import getUncertainty
    # from Simulation import Simulation, Cycle, EntrySim

    # sim = Simulation(cycle=Cycle(1),output=False,**EntrySim())
    sim,sol = OptimizeSRP()
    # print sol.x
    # perturb = getUncertainty()['parametric']
    # p = np.array([ 165.4159422 ,  308.86420218,  399.53393904])
    # print "RS cost of nominal optimized profile: {}".format(SRPCostRS(p, sim, perturb))
    # OptimizeSRPRS()
    
    # testExpansion()

    # import numpy as np
    
    # t = np.linspace(9.8, 19,500)
    # t = np.linspace(0, 25,5000)
    # b = HEPBankSmooth(t,10,50,200,minBank = np.radians(15.), maxBank = np.radians(85.))    
    # b = HEPNR(t,5,15,minBank = np.radians(15.), maxBank = np.radians(85.))    
    # b = [maneuver(T, 0, np.radians(-80), np.radians(-80)) for T in t]
    # b = np.degrees(b)
    
    # t -= t[0]
    # print 
    # for deg in [3,9]:
        # p = np.polyfit(t,b,deg)
        # print p
        # bp = np.polyval(p,t)
        # plt.plot(t,np.abs(b-bp),'--',label=str(deg))
    
    
    # plt.plot(t,b,'k',label='Truth')
    # plt.legend(loc='best')
    # plt.show()
    