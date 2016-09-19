from numpy import sin, cos
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import root


def sat(x,xmin,xmax):
    return np.fmax(xmin,np.fmin(x,xmax))


def sec(x):
    return 1.0/cos(x)


def getProblem(vertical=False):
    problem = {}
    # Problem Info #
    isp = 268
    
    problem['x0'] = -3200*(not vertical)
    problem['xf'] = 0
    
    problem['y0'] = 2000
    problem['yf'] = 0
    
    problem['u0'] = 625*(not vertical)
    problem['uf'] = 0
    
    problem['v0'] = -270
    problem['vf'] = 0
    
    problem['udotf'] = 0
    # Equivalent constraint on mu:
    problem['muf'] = np.pi/2.0
    
    problem['m0'] = 8500
    problem['ve'] = isp*9.81
    thrust = 600000
    problem['Tmax'] = thrust
    problem['Tmin'] = thrust*0.1  # 10% throttle

    if vertical:
        fpa0 = -np.pi/2.0
    else:
        V0 = (problem['u0']**2 + problem['v0']**2)**0.5
        fpa0 = np.arcsin(problem['v0']/V0)
    problem['mu0'] = np.pi+fpa0
    problem['mudotmax'] = 40*np.pi/180  # 40 deg/s

    problem['gain'] = 1000
    problem['tf'] = None  
    
    # problem['sol0'] = np.array([0, 0, 3, 0, -1, -10, 12])  # The solution to the initial problem beginning the homotopy.
    problem['sol0'] = np.array([0, 0, 2, -2, -1, 17, 0.001])
    problem['solv'] = np.array([0, 0.0431, 0, -2.6119, -0.8889, 0, 9.458]) # Costates of the corresponding vertical solution
    problem['h'] = 0
    return problem


# #################### #
# Pontryagin Approach  #
# #################### #

def  EoM(X,t, lx,ly,lu0,lv0, problem, output=False):
    # Defines the equations of motion for 2-D powered flight and the associated costates.
    # Only two costates needs to be integrated numerically.

    t = t*problem['tf']  # Create the true time
    
    g = 3.7

    x, y, u, v, m, mu, lm, lmu = X
    lu = lu0-lx*t
    lv = lv0-ly*t
    target = np.pi+np.arctan2(lv, lu)

    # if not hasattr(EoM,'reached'):
    if t == 0:
        setattr(EoM,'reached',False)
        setattr(EoM,'sign',np.sign(lmu))

    # if (not EoM.reached) and (np.sign(lmu) != EoM.sign):
    if (not EoM.reached) and (EoM.sign*lmu <= 0.004):
        EoM.reached = True

    tTurn = np.abs(np.pi/2.0-mu)/problem['mudotmax']
    if (t+tTurn) >= problem['tf']:
        target = np.pi/2.0
        dmu = np.sign(target-mu)*problem['mudotmax']
    else:
        if EoM.reached:
            if np.abs(lu) <= 1e-8:
                dmu = sat((target - mu) * 100, -problem['mudotmax'], problem['mudotmax'])
            else:
                dmu = (lx*lv/lu**2 - ly/lu)*cos(mu)**2 + sat((target-mu)*100,-problem['mudotmax'],problem['mudotmax'])
        else:
            dmu = -EoM.sign*np.sign(lu*sin(mu)-lv*cos(mu))*problem['mudotmax']

    s = (lu*cos(mu)+lv*sin(mu)) - m*lm/problem['ve']
    if s <= 0:
        T = problem['Tmax']
    else:
        T = problem['Tmin']
    
    dx = u
    dy = v
    du = T*cos(mu)/m
    dv = T*sin(mu)/m-g
    dm = -T/problem['ve']
    dlm = T*(lu*cos(mu)+lv*sin(mu))/m**2
    dlmu = T*(lu*sin(mu)-lv*cos(mu))/m
    
    if output:
        return T, dmu
    else:
        return np.array([dx, dy, du, dv, dm, dmu, dlm, dlmu])*problem['tf']


def EoM_Simple(X,t,problem):
    return t*np.ones_like(X)*np.sqrt(2*problem['tf'])

    
def EoM_Homotopy(X,t, lx,ly,lu0,lv0, problem):
    h = problem['h']
    return (1-h)*EoM_Simple(X,t,problem)+h*EoM(X,t, lx,ly,lu0,lv0, problem)
    
    
def SimpleCost(guess,problem):
    g1 = guess[0:6] - problem['sol0'][0:6]

    lx,ly,lu0,lv0,lm0,lmu0,tf = guess
    problem['tf'] = tf
    x0 = np.array([lx,ly,lu0,lv0,lm0,lmu0,1,1])
    X = odeint(EoM_Homotopy, x0, np.linspace(0,1,50), args=(lx, ly, lu0, lv0, problem))

    g = np.hstack((g1,X[-1,4]-lm0-tf))
    return g


def HomotopyCost(guess,problem):
    h = problem['h']
    return (1-h)*SimpleCost(guess,problem) + h*PMPCost(guess,problem,EoM_Homotopy)


def PMPCost(guess,problem,dynamics=EoM):

    lx,ly,lu0,lv0,lm0,lmu0,tf = guess
    problem['tf'] = tf
    x0 = np.array([problem['x0'],problem['y0'],problem['u0'],problem['v0'],problem['m0'],problem['mu0'],lm0,lmu0])*problem['h'] + \
         np.array([problem['xf'], problem['yf'], problem['uf'], problem['vf'], problem['m0']*0.5, problem['muf'], lm0, lmu0]) * (1-problem['h'])
    X = odeint(dynamics, x0, np.linspace(0,1,500), args=(lx, ly, lu0, lv0, problem))

    imax = X.shape[0]
    for i in range(imax):
        if np.all(X[i, :] == 0.0):
            print "Debug: Stripping Bad Values"
            imax = i
            break

    Xf = X[imax-1,:]
    xf,yf,uf,vf,mf,muf,lmf,lmuf = Xf
    luf = lu0-lx*tf
    lvf = lv0-ly*tf
    dx, dy, du, dv, dm, dmu, dlm, dlmu = EoM(Xf, 1, lx, ly, lu0, lv0, problem, output=False)

    Hf = lx*uf + ly*vf + luf*du + lvf*dv + lmf*dm + lmuf*dmu 

    g = [
      # Equality Constraints
        (xf-problem['xf'])/100,
        (yf-problem['yf'])/100,
        (uf-problem['uf'])/10,
        (vf-problem['vf'])/10,
        1*(muf-problem['muf']),   # Ensures a vertical landing, could be automatically satisfied

        1*Hf,                     # Free tf, transversality condition
        lmf + 1]                  # Free final mass, so the associated costate is fixed at the final time

    return np.array(g)


def PMPOpt():
    import matplotlib.pyplot as plt

    vertical = not True
    problem = getProblem(vertical=vertical)
    problem['h'] = 1
    if vertical:
        guess = problem['solv']
    else:
        # guess = np.array([0.08042227,   0.12706492,   2.66632126,  -0.36258304,  -0.75506248,  -9.26403713,  12.26226516])
        guess = np.array([0.0829,   0.1328,   2.6765,  -0.3318,  -0.7549,  -9.1742,  12.21])

    # guess = [g*1.01 for g in guess]

    sol = root(PMPCost, guess, args=(problem,EoM),tol=1e-4)
    xsol = sol['x']
    print sol['message']
    print "Number of function evaluations: {}\n".format(sol['nfev'])
    print xsol

    # xsol = guess
    print PMPCost(xsol, problem)
    t, x,y, u,v, udot,vdot, m,T, mu,mudot, lx,ly,lu,lv,lm,lmu, s, H,Hp,Hv,Hm,Hmu = PMPParse(xsol,problem)

    if np.abs(lu[0]) > 1e-12:
        mu_dot_opt = (lx*lv/lu**2 - ly/lu)*cos(mu)**2
    else:
        mu_dot_opt = np.zeros_like(t)
    print lv[-1]
    print (-mudot[-1]*lmu[-1]+lm[-1]*problem['Tmin']/problem['ve'])/(problem['Tmin']/m[-1] - 3.7)
    print("Prop used: {} kg".format(m[0]-m[-1]))

    plt.figure()
    plt.plot(x,y)
    plt.plot(0,0,'kx')
    # plt.plot(problem['x0'],problem['y0'])
    plt.title('Positions')
    
    plt.figure()
    plt.plot(u,v)
    plt.plot(0,0,'kx')
    plt.title('Velocities')
    
    plt.figure()
    plt.plot(udot,vdot)
    plt.plot(udot[-1],vdot[-1],'x')
    plt.plot(udot[0],vdot[0],'o')
    plt.title('Accelerations')
    
    plt.figure()
    plt.plot(t,m)
    plt.title('Mass')
    
    plt.figure()
    plt.plot(t,mu*180/np.pi)
    plt.plot(t,np.arctan2(lv,lu)*180/np.pi + 180)
    plt.title('Thrust Angle')
    
    plt.figure()
    plt.plot(t,T/problem['Tmax'])
    plt.axis([0, t[-1], 0, 1.1])
    plt.title('Throttle')
    
    plt.figure()
    plt.plot(t,mudot*180/np.pi)
    plt.plot(t,mu_dot_opt*180/np.pi)
    plt.legend(('Proportional','Formula'))
    plt.title('Mu dot')
    
    plt.figure()
    for l in [lx,ly,lu,lv,lm,lmu]:
        plt.plot(t,l)
    plt.title('Costates')

    plt.figure()
    # plt.plot(t,s)
    plt.plot(t,lu*sin(mu)-lv*cos(mu))
    plt.title('Switching fun')

    plt.figure()
    plt.plot(t,H)
    # for h in [Hp,Hv,Hm,Hmu]:
    #     plt.plot(t,h)
    plt.title('Hamiltonian')
    # plt.legend(('H','pos','vel','mass','mu'))

    plt.show()    


def PMPParse(guess,problem):
    print "Parsing solution:"
    g = 3.7
    lx,ly,lu0,lv0,lm0,lmu0,tf = guess
    problem['tf'] = tf
    x0 = np.array([problem['x0'],problem['y0'],problem['u0'],problem['v0'],problem['m0'],problem['mu0'],lm0,lmu0])*problem['h'] + \
         np.array([problem['xf'], problem['yf'], problem['uf'], problem['vf'], problem['m0']*0.5, problem['muf'], lm0, lmu0]) * (1-problem['h'])
    tau = np.linspace(0, 1, 500)
    t = tau*tf  # Turn tau into true time

    X = odeint(EoM, x0, tau, args = (lx,ly,lu0,lv0,problem))

    imax = len(t)
    for i in range(len(t)):
        if np.all(X[i, :] == 0.0):
            imax = i
            print "Removing {} elements from final solution.".format(len(t)-imax)
            break

    t = t[0:imax]
    x, y, u, v, m, mu, lm, lmu= X[0:imax, 0], X[0:imax, 1], X[0:imax, 2], X[0:imax, 3], X[0:imax, 4], X[0:imax, 5], X[0:imax, 6], X[0:imax, 7]
    
    lu = lu0-lx*t
    lv = lv0-ly*t
    mu_dot = np.ones_like(t)
    T = np.ones_like(t)
    s = (lu*cos(mu)+lv*sin(mu)) - m*lm/problem['ve']

    for i in range(len(t)):
        T[i], mu_dot[i] = EoM(X[i,:],tau[i], lx, ly, lu[0], lv[0], problem, output=True)

    udot = T*cos(mu)/m
    vdot = T*sin(mu)/m-g

    H = lx * u + ly * v + lu * udot + lv * vdot + lm * -T/problem['ve'] + lmu*mu_dot
    Hp = lx * u + ly * v
    Hv = lu * udot + lv * vdot
    Hm = lm * -T/problem['ve']
    Hmu = lmu*mu_dot

    return t,x,y,u,v,udot,vdot,m,T,mu,mu_dot, lx*np.ones_like(t),ly*np.ones_like(t),lu,lv,lm,lmu, s, H, Hp,Hv,Hm,Hmu


def testSimple():
    problem = getProblem()
    guess = problem['sol0']
    print SimpleCost(guess,problem)
    sol = root(SimpleCost, guess, args=(problem), tol=1e-12)
    xsol = sol['x']
    print sol['message']
    print "Number of function evaluations: {}\n".format(sol['nfev'])
    print xsol
    return None


def Homotopy():

    problem = getProblem()
    guess = problem['sol0']
    print PMPCost(guess,problem)

    guess = np.array([0.08042227,   0.12706492,   2.66632126,  -0.36258304,  -0.75506248,  -9.26403713,  12.26226516])
    problem['h'] = 1
    print PMPCost(guess,problem)

    h = 0
    Iu = 15  # Reasonable number of iterations
    dh = 0.00000001 # Initial stepsize
    dh_max = 0.1
    dh_min = 0.0001
    guess = problem['sol0']
    # h += dh
    while True:

        problem['h'] = h
        print "Current step: {}".format(h)
        print "Current stepsize: {}".format(dh)
        print "Current guess: {}".format(guess)
        sol = root(PMPCost, guess, args=(problem), tol=1e-4)
        print sol['message']

        if np.any(np.abs(sol['x'])>1e3):
            print "Homotopy aborted because the solution is diverging."
            return sol['x']

        if "converged" in sol['message']:
            if np.abs(h - 1) <= 1e-6:
                print "Final Solution: {}".format(sol['x'])
                return sol['x']
            guess = sol['x']
            print "Number of function evaluations: {}".format(sol['nfev'])
            print "New guess: {}\n".format(guess)
            # Update the stepsize:
            if sol['nfev'] <= Iu:
                print 'Increasing stepsize due to low iterations'
                dh *= 1.8
            elif sol['nfev'] >= 75:
                dh *= 0.5
            dh = sat(dh,dh_min,np.min((dh_max,1-h)))
            h += dh

        else:
            if dh == dh_min:
                print "Homotopy aborted because the last step failed and stepsize is equal to minimum stepsize."
                return sol['x']
            print "Decreasing stepsize from {} to {} and rerunning current guess.\n".format(dh,dh/2.0)
            h = np.max((0,h-0.5*dh))
            dh *= 0.5
            dh = sat(dh,dh_min,dh_max)

def testODE():
    fun = lambda x,t: 0

    sol = odeint(fun,1,0.1)
    print sol

if __name__=='__main__':
    PMPOpt()
    # testSimple()
    # Homotopy()
    # testODE()