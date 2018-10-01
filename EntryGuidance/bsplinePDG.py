""" SRP Guidance based on differential algebra, differential flatness, and b-splines """
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splrep, splev, splder, BSpline
from scipy.integrate import trapz,cumtrapz
from scipy.io import savemat
from pyaudi import gdual_double as gd
import time

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from Utils import DA as da
from SRP import SRP_Riccati
from TrajPlot import TrajPlot as traj3d
import time

def optimize():
    from scipy.optimize import minimize
    N = 30 # Number of discretization points

    m0 = 8500.
    ve = 290.*9.81
    x0 = np.array([-3200., 400, 3200, 625., -80, -270.])
    xf = np.array([-3200., 400, 2600, 625., -80, -270.])*0
    # x = np.array([np.linspace(x0i,xfi,N) for x0i,xfi in zip(x0,xf)]).T
    tf = 14.5
    # tf = 20
    srp = SRP_Riccati()
    x,_ = srp.solve(tf, N, max_iter=1)  # Warm start turns out to be essential for convergence in some cases
    order = 3  # cubic splines
    t = np.linspace(0, 1, N)  # Normalized time

    spl = [splrep(t,x[:,i]) for i in range(3)]

    tknots = spl[0][0]
    c = [sp[1][1:-(order+2)] for sp in spl]#  Remove x0 and xf from constraints by removing the initial and final coefficients from the optimization problem.
    c = np.concatenate(c, axis=0)
    c = np.append(c, [tf])
    t0 = time.time()
    result = minimize(cost, c, args=(tknots,x0,xf), method='SLSQP', constraints=constraint_dict(args=(tknots,x0,xf)), options={'disp':True,'maxiter':1000})    
    print ("NLP time: {} s".format(time.time()-t0))
    
    tfine = np.linspace(0,1,50)
    splines = coeff2spl(result.x,tknots,x0,xf)
    P,V,A,T,mu,eta = getStates(splines, tfine, result.x[-1])
    if False: # show the initial guess 
        # splines = coeff2spl(c,tknots,x0,xf)
        # P,V,A,T,mu,eta = getStates(splines, tfine, tf)
    tfine *= result.x[-1]
    m = np.exp(np.log(m0) + cumtrapz(-T/ve,tfine,initial=0))
    print ("Prop used: {} kg".format(m0-m[-1]))
    plt.figure()
    plt.plot(tfine,m)
    plt.xlabel('Time (s)')
    plt.ylabel('Mass (kg)')
    plt.figure()
    plt.plot(tfine,P)
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.figure()
    plt.plot(tfine,V)
    plt.legend(('In-plane','Out-of-plane','Vertical'))
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.figure()
    plt.plot(tfine,T)
    plt.xlabel('Time (s)')
    plt.plot(tfine,40*np.ones_like(tfine),'k--')
    plt.plot(tfine,70*np.ones_like(tfine),'k--')
    plt.axis([0,tf+2,0,80])
    plt.figure()
    plt.plot(tfine,np.degrees(mu),label='pitch')
    plt.plot(tfine,np.degrees(eta),label='yaw')
    plt.xlabel('Time (s)')
    plt.ylabel('(deg)')
    plt.legend()
    traj3d(*(P.T),T=A*20)
    plt.show()

    # savemat('fuel_opt_srp.mat',{key:val for key,val in zip(['P','V','A','T','mu','eta'],[P,V,A,T,mu,eta])})

def coeff2spl(c, t, x0, xf):
    """ Converts the flat coefficient matrix into 3 tck tuples """
    n = len(t)-6 # 6 = order + 3
    z = np.zeros((4,)) # order + 1
    return [(t,np.concatenate(([x0[0]],c[0:n],[xf[0]],z)),3), (t,np.concatenate(([x0[1]],c[n:2*n],[xf[1]],z)),3), (t,np.concatenate(([x0[2]],c[2*n:3*n],[xf[2]],z)),3)]


def cost_minimum_energy(t, a):
    return trapz(a**2,t)

def cost_minimum_fuel(t, a):
    return trapz(a,t)

def cost(c, t, x0, xf):
    splines = coeff2spl(c,t,x0,xf)
    P,V,A,T,mu,eta = getStates(splines, t,c[-1])
    # J = cost_minimum_energy(t*c[-1],T)
    J = cost_minimum_fuel(t*c[-1],T)
    return J

def constraints(c, t, x0, xf):
    """ Define any constraints that cannot be satisfied as bounds on the spline coefficients """
    splines = coeff2spl(c,t,x0,xf)
    P,V,A,T,mu,eta = getStates(splines,t,c[-1])
    Tmax = 70
    Tmin = 40
    # Have to written as >= 0 for SLSQP
    n = 2
    return np.array([Tmax**n-T**n,            # Max thrust constraint
                     T**n-Tmin**n,            # Min thrust constraint
                     # P[:,2],                  # Altitude constraint
                    -P[:,0]**2 - P[:,1]**2 + (P[:,2]*np.tan(85*np.pi/180.))**2      # Glide slope constraint
                    ]).flatten()



def constraints_eq(c, t, x0, xf):
    splines = coeff2spl(c,t,x0,xf)
    P,V,A,T,mu,eta = getStates(splines,t,c[-1])

    con = np.array([V[0,:]-x0[3:6],
                    V[-1,:]-xf[3:6],
                    ]).flatten()

    # con = np.append(con,[mu[-1]-np.pi/2])

    return con

def constraint_dict(args):
    c = {'type':'ineq','fun':constraints,'args':args}
    ceq = {'type':'eq','fun':constraints_eq,'args':args}
    return [c,ceq]

def constraint_jac(c,t,x0,xf):
    cvars = ["c{}".format(i) for i in range(c)]
    cda = da.make(c,cvars,1,True)

    con = constraints(cda,t,x0,xf)
    return da.jacobian(con)

def getStates(splines, t, tf):
    """ Input is a list of the spline tck for each of x,y,z """

    Z = np.array([splev(t, spl, i) for i in [0,1,2] for spl in splines]).T   # x,y,z and their first and second derivatives - will not work for DA

    # splines = [BSpline(*spl) for spl in splines]
    # splines += [splder(spl) for spl in splines]        # Add derivative
    # splines += [spl.derivative() for spl in splines[3:]]    # Add second derivative
    # print (splines[0].k)
    # print len(splines[4].t)
    # print (splines[8].k)

    # For each spline, we need to add two zeros to the end of c and two zeros to the beginning of t for each derivative taken

    # Z = np.array([da.splev(t,spl.tck) for spl in splines])     # Evaluate all the splines

    # print Z.shape

    g = np.tile([0,0,3.71],(len(t),1))  # Matrix of gravity vectors -  note: we don't have to assume g is constant

    # States
    P = Z[:,0:3] # Positions
    V = Z[:,3:6]/tf # Velocities
    A = Z[:,6:9]/(tf**2) + g # Thrust acceleration

    # Controls
    T = np.linalg.norm(A, axis=1)      # Thrust acceleration magnitude
    mu = np.arctan2(A[:,2], np.linalg.norm(A[:,0:2], axis=1)) # Pitch angle
    eta = np.arctan2(A[:,1], A[:,0]) # Yaw angle

    return P,V,A,T,mu,eta

# ##################################### #
#          Functionality Tests          #
# ##################################### #

def testBsplineDA():
    x = 1
    t = np.linspace(0,1,15)
    y = [ti**3 + x for ti in t]
    dy = 3*t**2
    k = 3
    spl = splrep(t,y,k=k)
    
    tstar = [np.sum(spl[0][i+1:i+k-1])/(k-1) for i in range(len(spl[0]))]

    da_names = ['c{}'.format(i) for i in range(len(spl[1]))]
    # print "Unique Knots: {}".format(sorted(list(set(spl[0]))))
    # print "Knots: {} ({} elements)".format(spl[0],len(spl[0]))
    # print "Coeff: {} ({} elements)".format(spl[1],len(spl[1]))

    daSpline = BSpline(spl[0],da.make(spl[1],da_names,2),spl[2])
    # print daSpline.c # Good first step, creating the spline out of dual variables
    # print daSpline.derivative().tck # Small changes to scipy allow this to work as well

    tfine = np.linspace(0,0.98,7)
    B = da.splev(tfine,daSpline) # The final piece is evaluating the daSplines - wrote my own method for this 
    print("My eval w/ expansion = {}".format(B))
    print("My eval = {}".format(da.const(B)))


    # spl = splder(spl)

    # print "Spline Derivative:"
    # print "Knots: {} ({} elements)".format(spl[0],len(spl[0]))
    # print "Coeff: {} ({} elements)".format(spl[1],len(spl[1]))    # da.splder(daSpline)
    # print "K = {}".format(spl[2])
    # print " "
    # print "Estimated Derivative:"


    # print daSpline([0.1, 0.5]) # No luck evaluating, which is crucial
    print ("True eval = {}".format(splev(tfine,spl)))
    
    tfine = np.linspace(0,1,150)
    yfine = splev(tfine,spl)
    # dyfine = splev(tfine,spl,1) # First deriative

    plt.plot(t,y,'o',tfine,yfine,tstar,spl[1])
    # plt.plot(t,dy,'^',tfine,dyfine)
    plt.show()

def testStepSpline():
    x = [0]*10 +[0.5]*3+ [1]*10
    t = np.linspace(0,1,len(x))
    spl = splrep(t,x)

    teval = np.linspace(0,1,100)
    xeval = splev(teval,spl)

    plt.plot(t,x,teval,xeval,'k--')
    plt.show()

def testOptSpline():
    """ Demonstrate optimization of spline coefficients with fixed knots to approximate a curve """
    from scipy.optimize import minimize
    from scipy.integrate import trapz

    t = np.linspace(0,2*np.pi,500)
    f = np.clip(np.sin(t),-0.7,0.6)

    order = 3
    knots = np.linspace(0,2*np.pi,10) # Only internal knots - need to transform to full knot set
    c = np.ones((len(knots)+order+1,)) # Initial guess
    cexact = splrep(knots,np.clip(np.sin(knots),-0.7,0.6),k=order)
    print (cexact[0])
    print (cexact[1])
    
    def cost(c,knots,order,t,f):
        fc = splev(t, (knots,c,order))
        return trapz((fc-f)**2, t)

    # cost(cexact[1],cexact[0],3,t,f)
    copt = minimize(cost, c, args=(cexact[0],order,t,f),method='BFGS')
    print(copt.x )
    fc = splev(t, (cexact[0],copt.x,order))
    fe = splev(t, cexact)
    plt.plot(t,f,t,fc,'--',cexact[0],copt.x,'o')
    plt.show()

if __name__ == "__main__":
    # testBsplineDA()
    # testStepSpline()
    # testOptSpline()
    optimize()
