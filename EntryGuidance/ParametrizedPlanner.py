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


def HEPNR(T,t1,t2,minBank = np.radians(15.), maxBank = np.radians(85.)):
    """ A version of the 3 switch planner with no reversals planned (and thus only two switches). May be useful in schemes like Apollo. """
    isScalar = False
    bank = []
    if not isinstance(T,(list,np.ndarray)):
        T = [T]
        isScalar = True

    for t in T:
        if t <= t1:
            bank.append(minBank)
        elif t > t1 and t <= t2:
            bank.append(maneuver(t, t1, minBank,maxBank))
        elif t > t2:
            bank.append(maneuver(t, t2, maxBank,minBank))
            
    if isScalar:
            try:
                return bank[0]
            except:
                print "Bank angle comp failed"
                return -1.
    else:
        return bank
    
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

def HEPBankReduced(T,t1,t2,minBank = np.radians(15.), maxBank = np.radians(85.)):
    maxRate = np.radians(20.)
    ti1 = t1 + 2*(maxBank)/maxRate
    ti2 = t2 + (maxBank+minBank)/maxRate
    isScalar = False
    bank = []
    if isinstance(T,(float,int,np.float32,np.float64)):
        T = [T]
        isScalar = True

    for t in T:

        if t <= t1:
            bank.append(maxBank)
        elif t >= t1 and t <= ti1:
            bank.append(maxBank-maxRate*(t-t1))
        elif t >= ti1 and t <= t2:
            bank.append(-maxBank)
        elif t >= t2 and t <= ti2:
            bank.append(-maxBank + maxRate*(t-t2))
        else:
            bank.append(minBank)
    if isScalar:
        try:
            return bank[0]
        except:
            print "Bank angle comp failed"
            return -1.
    else:
        return bank

        
def HEPBankReducedSmooth(T,t1,t2,minBank = np.radians(15.), maxBank = np.radians(85.)):
    maxRate = np.radians(20)
    maxAcc = np.radians(5)
    dt = maxRate/maxAcc                                     # Amount of time to go from 0 bank rate to max or vice versa
    dbank = 0.5*(maxRate**2)/maxAcc                         # Angle traversed during max accel for dt seconds
    t1a = t1+dt                                             # Acceleration phase
    t1v = t1 + 2*maxBank/maxRate                            # Max velocity phase
    t1d = t1v + dt                                          # Max deceleration phase    #Future: check that this is less than t2
    if t1d >= t2:
        noMax1 = True
    else:
        noMax1 = False
    
    t2a = t2 + dt
    t2v = t2 + (minBank+maxBank)/maxRate
    t2d = t2v + dt
    
    isScalar = False
    bank = []
    if isinstance(T,(float,int,np.float32,np.float64)):
        T = [T]
        isScalar = True
        
    for t in T:
        if t <= t1:
            bank.append(maxBank)
        elif t > t1 and t <= t1a: #Negative acceleration phase
            bank.append(maxBank-0.5*maxAcc*(t-t1)**2)
        elif t > t1a and t <= t1v: #Max velocity phase
            bank.append(maxBank-dbank-maxRate*(t-t1a))
        elif t > t1v and t <= t1d:
            bank.append(dbank-maxBank - maxRate*(t-t1v) + 0.5*maxAcc*(t-t1v)**2 )
        elif t > t1d and t <= t2:
            bank.append(-maxBank)
        elif t > t2 and t <= t2a:
            bank.append(-maxBank + 0.5*maxAcc*(t-t2)**2)
        elif t > t2a and t <= t2v:
            bank.append(dbank-maxBank + maxRate*(t-t2a))
        elif t > t2v and t <= t2d:
            bank.append(minBank-dbank + maxRate*(t-t2v) - 0.5*maxAcc*(t-t2v)**2)
        else:
            bank.append(minBank)
        
    if isScalar:
        try:
            return bank[0]
        except:
            print "Bank angle comp failed"
            return -1.
    else:
        return bank
        
def HEPBank(T,t1,t2,t3,minBank = np.radians(15.), maxBank = np.radians(85.)):

    maxRate = np.radians(20.)
    ti1 = t1 + (maxBank+minBank)/maxRate
    ti2 = t2 + 2*(maxBank)/maxRate
    ti3 = t3 + (maxBank+minBank)/maxRate
    isScalar = False
    bank = []
    if isinstance(T,(float,int,np.float32,np.float64)):
        T = [T]
        isScalar = True

    for t in T:
        if t < t1 and t >= 0:
            bank.append(-minBank) 
        elif t >= t1 and t <= ti1:
            bank.append(-minBank+maxRate*(t-t1))
        elif t >= ti1 and t <= t2:
            bank.append(maxBank)
        elif t >= t2 and t <= ti2:
            bank.append(maxBank-maxRate*(t-t2))
        elif t >= ti2 and t <= t3:
            bank.append(-maxBank)
        elif t >= t3 and t <= ti3:
            bank.append(-maxBank + maxRate*(t-t3))
        else:
            bank.append(minBank)
    if isScalar:
        try:
            return bank[0]
        except:
            print "Bank angle comp failed"
            return -1.
    else:
        return bank

def HEPBankSmooth(T, t1, t2, t3, minBank=np.radians(15.), maxBank=np.radians(85.)):
    
    maxRate = np.radians(20)
    maxAcc = np.radians(5)
    dt = maxRate/maxAcc                                     # Amount of time to go from 0 bank rate to max or vice versa
    dbank = 0.5*(maxRate**2)/maxAcc                         # Angle traversed during max accel for dt seconds
    
    t1a = t1 + dt
    t1v = t1 + (minBank+maxBank)/maxRate
    t1d = t1v + dt
    
    t2a = t2 + dt                                           # Acceleration phase
    t2v = t2 + 2*maxBank/maxRate                            # Max velocity phase
    t2d = t2v + dt                                          # Max deceleration phase    #Future: check that this is less than t3
    
    t3a = t3 + dt
    t3v = t3 + (minBank+maxBank)/maxRate
    t3d = t3v + dt

    isScalar = False
    bank = []
    if isinstance(T,(float,int,np.float32,np.float64)):
        T = [T]
        isScalar = True

    for t in T:
        if t <= t1:
            bank.append(-minBank)
        elif t > t1 and t <= t1a: #Positive acceleration phase
            bank.append(-minBank + 0.5*maxAcc*(t-t1)**2)
        elif t > t1a and t <= t1v: #Max velocity phase (if it exists)
            bank.append(-minBank+dbank+maxRate*(t-t1a))
        elif t > t1v and t <= t1d:    
            bank.append(maxBank - dbank + maxRate*(t-t1v) - 0.5*maxAcc*(t-t1v)**2)
        elif t > t1d and t <= t2:
            bank.append(maxBank)
        elif t > t2 and t <= t2a: #Negative acceleration phase
            bank.append(maxBank-0.5*maxAcc*(t-t2)**2)
        elif t > t2a and t <= t2v: #Max velocity phase
            bank.append(maxBank-dbank-maxRate*(t-t2a))
        elif t > t2v and t <= t2d:
            bank.append(dbank-maxBank - maxRate*(t-t2v) + 0.5*maxAcc*(t-t2v)**2 )
        elif t > t2d and t <= t3:
            bank.append(-maxBank)
        elif t > t3 and t <= t3a:
            bank.append(-maxBank + 0.5*maxAcc*(t-t3)**2)
        elif t > t3a and t <= t3v:
            bank.append(dbank-maxBank + maxRate*(t-t3a))
        elif t > t3v and t <= t3d:
            bank.append(minBank-dbank + maxRate*(t-t3v) - 0.5*maxAcc*(t-t3v)**2)
        else:
            bank.append(minBank)
    if isScalar:
        try:
            return bank[0]
        except:
            print "Bank angle comp failed"
            return -1.
    else:
        return bank    
        
def checkFeasibility(T,sign=-1):
    
    sig = sign*np.diff(T) #np.array([T[0]-T[1],T[1]-T[2]])
    cost = (sum(sig+np.abs(sig)) + sum(abs(T)-T))*1e5
    if cost:
        return max(cost,1e7)
    else:
        return 0
        


    
def Optimize(x0,n=3,iv='time'):
    
    from scipy.optimize import minimize, minimize_scalar
       
    if (n == 3):
        guess = [50,100,133]
        bankFun = HEPBank
    elif (n==2):
        guess = [106,133]
        bankFun = HEPBankReducedSmooth
    else:   
        print "Optimize: improper input for n."
        return None
    
    if iv.lower() == 'time':
        getIV = lambda x,t: t
        check = checkFeasibility
        
    elif 'vel' in iv.lower():
        if n == 2:
            guess = [-4700,-2100]
        elif n == 3:
            guess = [-5100,-4500,-2500]
        getIV = lambda x,t: -x[3]
        check = lambda v: checkFeasibility(-v,sign=1)
    
    entry = Entry(Trigger = partial(DeployParachute,{'velBias':30}))
    sol = minimize(HEPCost,np.array(guess),args = (x0, entry, Target(), bankFun, getIV, check), method='Nelder-Mead',tol=1e-5,options={'disp':True})
    # sol = minimize(HEPCost,np.array(guess),args = (x0, entry, Target(), bankFun, getIV, check), method='Powell',tol=1e-5,options={'disp':True})
    print "The {0} switching {1}s are {2}".format(n,iv,sol.x)
    return (lambda x,t: bankFun(getIV(x,t),*sol.x)), sol.x

def OptimizeSmooth(x0):
    from scipy.optimize import minimize, differential_evolution
    
    entry = Entry(Trigger=partial(DeployParachute,{'velBias':30}))
    bankFun = HEPBankReducedSmooth
    # bankFun = HEPBank

    # guess = [108,133]
    getIV = lambda x,t: t
    check = lambda T: True
    
    
    bounds = [(0,250),(100,350)]
    sol = differential_evolution(HEPCost,args = (x0, entry, Target(), bankFun, getIV, check),bounds=bounds, tol=1e-1, disp=True)

    print "The 2 switching times are {} with final cost {}".format(sol.x,sol.fun)

    return (lambda x,t: bankFun(getIV(x,t),*sol.x)), sol

def OptimizeSRP():
    """ Not named very well. Computes the three switch bank profile that delivers the entry vehicle to a prescribed altitude, downrange, crossrange at a given velocity trigger point. """
    
    from scipy.optimize import differential_evolution, minimize
    from Simulation import Simulation, Cycle, EntrySim

    sim = Simulation(cycle=Cycle(1),output=False,**EntrySim())
    x0 = InitialState()

    if 1:
        bounds = [(0,250),(100,500)]
        # bounds = [(0,250),(100,400),(100,550)]
        sol = differential_evolution(SRPCost,args = (sim,x0), bounds=bounds, tol=1e-1, disp=True, polish=False)
    else:
        sol = minimize(SRPCost,[ 165.4159422 ,  308.86420218,  399.53393904], args=(sim,x0), method='Nelder-Mead', tol=1e-5, options={'disp':True})
        
    # bankProfile = lambda **d: HEPBankReducedSmooth(d['time'],*sol.x)
    # bankProfile = lambda **d: HEPBankSmooth(d['time'], *sol.x, minBank=np.radians(30))
    bankProfile = lambda **d: HEPNR(d['time'], *sol.x, minBank=np.radians(30))
    
    print sol.x
    output = sim.run(x0,[bankProfile])
    
    sim.plot()
    sim.show()
    
    return sim,sol 
    
def SRPCost(p, sim, x0, sample=None):

    dr_target = 900
    cr_target = 0

    
    J = checkFeasibility(p)
    if J > 300:
        return J
    
    # bankProfile = lambda **d: HEPBankSmooth(d['time'],*p, minBank=np.radians(30))
    bankProfile = lambda **d: HEPNR(d['time'],*p, minBank=np.radians(30))
                                                
    output = sim.run(x0,[bankProfile],sample)

    Xf = output[-1,:]
    # data = np.c_[self.times, energy, bank, h,   r,      theta,       phi,      v,         gamma, psi,       range,     L,      D]
    hf = Xf[3]
    # fpaf = Xf[8]
    dr = Xf[10]
    cr = Xf[11]
    
    J = -hf + (0*(dr_target-dr)**2 + 0*(cr_target-cr)**2)**0.5 

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

def DragProfile(n, m):
    ''' Uses symbolic math to simplify the drag profile prior to use in DragPlanner 
        n is the degree of approximating polynomial
        m is the number of mesh splits - i.e. m=1 means two polynomials of order n are used
    '''
    import sympy as sym
    
    coeff, Ak, Bk = DragKnots(n,m,1)
    Abc, Bbc = DragBC(n,m)
       
    # Concatenate these matrices to form the complete set of linear constraints
    A = sym.Matrix(Ak+Abc)
    B = sym.Matrix(Bk+Bbc)
    
    s = sym.linsolve((A,B),coeff)
    print s
    print s.free_symbols
    # The symbolic variables to be replaced are V0,Vf,D0,Df, the m switches vsi, and the design variable(s)
    v0 = sym.Symbol('v0')
    vi = sym.Symbol('vi')
    vf = sym.Symbol('vf')
    
    D0 = sym.Symbol('D0')
    Di = sym.Symbol('Di')
    Df = sym.Symbol('Df')
    vs = [sym.Symbol('vs{}'.format(i)) for i in range(1,m+1)]
    # vars = [v0, vi, vf, D0, Di, Df] + vs
    vars = [v0, vf, D0, Df] + vs
    design = [sy for sy in s.free_symbols if not sy in vars]
    print design
    sfun = sym.lambdify([vars+design], list(s), "numpy") # we may not want numpy if we don't end up using the vectorization
    
    return sfun
    
def sampleDragProfiles(n,m):
    
    ndesign = int(np.ceil(n*(m+1) - 2 - 2*m))
    print "Number of free variables: {}".format(ndesign)
    coeff = DragProfile(n,m)
    
    Vscale = 500
    V0 = 5000/Vscale 
    Vi = 3700/Vscale # has to be greater than vs1 with current implementation
    Vf = 500/Vscale
    Vs = list(np.linspace(V0,Vf,m+2))
    
    sf = 300e3 
    gamma0 = np.radians(-8.2)
    hf = 1.4e3
    h0 = 23.16e3
    D0 = 44
    Dj = 81
    Df = 4.5 
    
    
    v = np.linspace(V0,Vf,200)
    for _ in range(5000):
        C = (-1+2*np.random.random(ndesign))/200
        # Di = [c for c in coeff([V0, Vi, Vf, D0, Dj, Df] + Vs[1:-1] + list(C))[0]]
        Di = [c for c in coeff([V0, Vf, D0, Df] + Vs[1:-1] + list(C))[0]]
       
        # Reconstruct the drag profile from coefficients          
        D = np.zeros_like(v)
        for j, Vsj in enumerate(Vs[:-1]):
            D[v<=Vsj] = sum([Di[n*j + i]*v[v<=Vsj]**i for i in range(n)])
            
            
        if np.all(np.array(D)>0) and np.all(np.array(D)<(Dj+5)):
            print C
            plt.plot(v*Vscale, D)
    
    
    # Plot a true optimal drag profile for comparison
    vals = '1.30575967020474e-07	-1.41198624451222e-05	0.000600124327551634	-0.0138971614274694	0.198664313828484	-1.85640234286750	11.6152838992996	-48.8247269033992	136.234831743230	-246.631224418153	281.651083129391	-177.907118367548'
    das = [float(vv) for vv in vals.split()[::-1]]
    Dtrue0 = 50.0267
    Dtrue = Dtrue0
    for i,val in enumerate(das):
        Dtrue += val*v**(i+1)
        
    plt.plot(v*Vscale,Dtrue,'k--')    
    plt.show()
    
def DragBC(n, m):
    ''' Sets up symbolic math for the two altitude boundary conditions '''
    import sympy as sym
    v0 = sym.Symbol('v0')
    vi = sym.Symbol('vi')
    vf = sym.Symbol('vf')
    
    D0 = sym.Symbol('D0')
    Di = sym.Symbol('Di')
    Df = sym.Symbol('Df')
    
    A0 = [v0**i for i in range(n)] + [0]*n*m
    Ai = [vi**i for i in range(n)] + [0]*n*m
    Af = [0]*n*m + [vf**i for i in range(n)]
    A = [A0,Af]
    b = [D0,Df]
    
    return A,b
    
    
def DragKnots(n, m, nderivs=1):
    ''' Solves the linear system defining the continuity conditions at m knot points. '''

    import sympy as sym

    ds = sym.symbols(['d{}'.format(i) for i in range(n*(m+1))])
    if m == 0:
        return ds, [], []
        
    A = []    
    for k in range(1,m+1):    
        vs = sym.Symbol('vs{}'.format(k)) # kth split velocity
    
    
        A1p = [vs**i for i in range(n)]
        A1n = [-a for a in A1p]
        A1 = [0]*n*(k-1) + A1p + A1n + [0]*n*(m-k)
        Atemp = [A1]

        for _ in range(nderivs):
            Atemp.append([sym.diff(a,vs) for a in Atemp[-1]])

        A.extend(Atemp) 
    
    b = [0]*(nderivs+1)*m
    
    return ds, A, b
    
def DragPlanner():
    # from scipy.integrate import cumtrapz
    from Utils.trapz import cumtrapz
    from pyaudi import gdual_double as gd
    from InitialState import InitialState
    from Utils import DA as da
    edl = EDL()
    x0 = InitialState()
    
    # vars = ['D1','D2','D3','D4','D5']
    # vals = (-8.2931,9.7341,-1.5803,0.1057,-0.0033)
    # D0 = 4.2643
    vals = '1.30575967020474e-07	-1.41198624451222e-05	0.000600124327551634	-0.0138971614274694	0.198664313828484	-1.85640234286750	11.6152838992996	-48.8247269033992	136.234831743230	-246.631224418153	281.651083129391	-177.907118367548'
    vals = [float(v) for v in vals.split()[::-1]]
    D0 = 50.0267
    
    use_da = False
    if use_da:
        from pyaudi import log, asin, acos, sin, cos

        vars = ['D{}'.format(i+1) for i in range(len(vals))]
        
        bounds = [(-0.6,0.1),(-1,6),(-26,6),(-15,41),(-13,18)]
        das = [gd(val,x,2) for x,val in zip(vars,vals)]
    else:
        from numpy import log, sin, cos
        from numpy import arcsin as asin
        from numpy import arccos as acos
        das = vals
        
    # Get the independent variable and corresponding drag profile
    Vf = 470  # Final velocity is sort of a design variable as well.
    V0 = 5400 # Velocity is not monotonic from V0 so this is the second time that the velocity reaches V0
    V = np.linspace(Vf, V0, 500)
    Vscaled = V/500
    M = V/235 # Assume M is linear in V, decent approximation. This could work instead of Vscaled
    D = D0
    for i,val in enumerate(vals):
        D += das[i]*Vscaled**(i+1)
        
    # Estimate altitude and energy
    cd,cl = edl.vehicle.aerodynamic_coefficients(M)
    L = cl*D/cd
    beta = edl.vehicle.BC(mass=x0[7], Mach=M)
    if use_da:
        h = np.array([-edl.planet.scaleHeight*log(2*Dd*betad/(Vd**2)/edl.planet.rho0) for Dd,Vd,betad in zip(D,V,beta)])
    else:
        h = -edl.planet.scaleHeight*log(2*D*beta/(V**2)/edl.planet.rho0)
    r = edl.radius(h)
    E = edl.energy(r, V, False)
    
    # Get the remaining quantities (range, fpa, bank)
    dh = np.diff(h)
    dE = np.diff(E)
    sfpa = -(dh/dE)*D[:-1]
    
    if use_da:
        fpa = np.array([asin(sfpa_) for sfpa_ in sfpa])
        cfpa = np.array([cos(fpa_) for fpa_ in fpa])
    else:
        fpa = asin(sfpa)
        cfpa = cos(fpa)
        
    dfpa = np.diff(fpa)
    cbank = ( (dfpa/dE[1:]) + cfpa[1:]/r[2:]/V[2:]/D[2:] - edl.gravity(r)[2:]*cfpa[1:]/(D[2:]*V[2:]**2) )*(-D[2:]*V[2:]**2/L[2:])
    if use_da:
        bank = np.array([acos(c) for c in np.clip(da.const(cbank,array=True),0,1)])
    else:
        bank = acos(np.clip(cbank,0,1))
    
    downrange = cumtrapz(-cfpa/D[:-1], E[:-1], initial=0)
    downrange_approx = cumtrapz(-1/D[:-1], E[:-1], initial=0) # assume cos(fpa) = 1 (we could also assu,e cos(fpa)=constant=cos(fpa0)
    downrange -= downrange[-1]
    downrange_approx -= downrange_approx[-1]
    
    if use_da:
        plt.figure()
        plt.plot(V,da.const(h, array=True)/1000,'k--')
        pts = np.random.random([20,5])
        for i in range(pts.shape[1]):
            # pts[:,i] = bounds[i][0] + pts[:,i]*bounds[i][1] 
            pts[:,i] *= vals[i]/(100)        # each component can vary by 1% of the nominal value
        hnew = da.evaluate(h, vars, pts)
        for hn in hnew:
            if hn.max() < 90e3:
                plt.plot(V, hn/1000)
        # print hnew.shape
        plt.figure()
        plt.plot(V[1:], da.const(downrange/1000, array=True),'k--')
        plt.plot(V[1:], da.const(downrange_approx/1000, array=True),'b--')
        # snew = da.evaluate(downrange, vars, pts)
        # for sn in snew:
            # plt.plot(V[1:], sn/1000)
        # plt.plot(da.const(E[1:]), da.const(fpa, array=True)*180/np.pi)
        # plt.figure()
        plt.plot(V[1:], da.const(fpa, array=True)*180/np.pi)
        # plt.figure()
        # plt.plot(V[2:], da.const(cbank, array=True))
    else:
        plt.figure()
        plt.plot(V,h/1000)
        plt.figure()
        plt.plot(V[:-1],downrange/1000)
        plt.plot(V[:-1],downrange_approx/1000)
        # plt.plot(E[:-1],downrange/1000)
    plt.figure()
    plt.plot(V[2:], bank*180/np.pi)

    plt.show()

    
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
    # sim,sol = OptimizeSRP()
    # print sol.x
    # perturb = getUncertainty()['parametric']
    # p = np.array([ 165.4159422 ,  308.86420218,  399.53393904])
    # print "RS cost of nominal optimized profile: {}".format(SRPCostRS(p, sim, perturb))
    # OptimizeSRPRS()
    
    # testExpansion()
    # DragProfile(3,0)
    # sampleDragProfiles(7,0,4)
    sampleDragProfiles(7,0)
    # DragPlanner()
    # 
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
    