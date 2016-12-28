# ######################################### #
# Homotopy Optimal Powered Descent Guidance #
# ######################################### #



from numpy import sin, cos
from numpy.linalg import norm
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import root, differential_evolution, minimize_scalar
import pyOpt


def sat(x,xmin,xmax):
    return np.fmax(xmin,np.fmin(x,xmax))


def sec(x):
    return 1.0/cos(x)


def getProblem():
    problem = {}
    # Problem Info #
    isp = 268
    
    problem['x0'] = -3200
    problem['xf'] = 0
    
    problem['y0'] = 2000
    problem['yf'] = 0
    
    problem['u0'] = 625
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

    problem['mudotmax'] = 40*np.pi/180  # 40 deg/s

    problem['gain'] = 1000
    problem['tf'] = None  
    
    # problem['sol0'] = np.array([0, 0.013355, 0, -2.433418, -0.756649, 0, 9.7562])  # Best known sol for h = 0
    # problem['sol1'] = np.array([0.080372, 0.126816, 2.667427, -0.361754, -0.755004, -9.279743, 12.319111]) #  Best known solution for h = 1

    problem['sol0'] = np.array([0, 0.013355, 0, -2.433418, -0.756649, 9.7562])  # Best known sol for h = 0
    # problem['sol1'] = np.array([0.080372, 0.126816, 2.667427, -0.361754, -0.755004, 12.319111]) #  Best known solution for h = 1
    problem['sol1'] = np.array([0.08066016,   0.12672358,   2.67065948, - 0.36033732, - 0.75492235,    12.35255585])
    problem['bounds'] = [(0,0.1), (0,0.2), (0,3), (-2.5,0), (-0.8,-0.7), (9,13)]
    problem['h'] = 0
    return problem


def getInitialState(guess,problem):
    lx,ly,lu0,lv0,lm0,tf = guess
    lmu0 = 0
    h = problem['h']
    c1 = sin(h * np.pi / 2)
    c2 = cos(h * np.pi / 2)
    y0 = np.sqrt(problem['y0']**2 + (c2*problem['x0'])**2)
    v0 = -np.sqrt(problem['v0']**2 + (c2*problem['u0'])**2)
    mu0 = np.pi + np.arctan2(v0, problem['u0']*c1)
    x0 = np.array([problem['x0']*c1, y0, problem['u0']*c1, v0, problem['m0'], mu0, lm0, lmu0,0])

    return x0

# #################### #
# Pontryagin Approach  #
# #################### #


def  EoM(X, t, lx,ly,lu0,lv0, problem, output=False):
    # Defines the equations of motion for 2-D powered flight and the associated costates.
    # Only two costates needs to be integrated numerically.

    t = t*problem['tf']  # Create the true time
    
    g = 3.7

    x, y, u, v, m, mu, lm, lmu, e = X
    lu = lu0-lx*t
    lv = lv0-ly*t
    target = np.pi+np.arctan2(lv, lu)
    dmu = sat((target*0.9975-mu)*20 + e*1,-problem['mudotmax'],problem['mudotmax'])
    
    # if t == 0:
        # setattr(EoM,'reached',False)
        # setattr(EoM,'sign',np.sign(lmu))

    # if (not EoM.reached)  and EoM.sign*(target-mu)>=1: #and (EoM.sign*lmu <= -0.001)
        # EoM.reached = True

    tTurn = np.abs(np.pi/2.0-mu)/problem['mudotmax']
    if (t+tTurn) >= problem['tf']:
        target = np.pi/2.0
        dmu = np.sign(target-mu)*problem['mudotmax']
    # else:
        # if EoM.reached:
            # if np.abs(lu) <= 1e-5:
                # dmu = 0
            # else:
                # dmu = (lx*lv/lu**2 - ly/lu)*cos(mu)**2 + sat((target-mu)*0,-problem['mudotmax'],problem['mudotmax'])
        # else:
            # dmu = -EoM.sign*np.sign(lu*sin(mu)-lv*cos(mu))*problem['mudotmax']

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
    dlmu = 0*T*(lu*sin(mu)-lv*cos(mu))/m
    
    if output:
        return T, dmu
    else:
        return np.array([dx, dy, du, dv, dm, dmu, dlm, dlmu,(target-mu)])*problem['tf']


def PMPCost(guess,problem,opt=False):

    lx,ly,lu0,lv0,lm0,tf = guess
    # lmu0 = 0
    problem['tf'] = tf
    x0 = getInitialState(guess,problem)
    X = odeint(EoM, x0, np.linspace(0,1,500), args=(lx, ly, lu0, lv0, problem))

    imax = X.shape[0]
    for i in range(imax):
        if np.all(X[i, :] == 0.0):
            print "Debug: Stripping Bad Values"
            imax = i
            break

    Xf = X[imax-1,:]
    xf,yf,uf,vf,mf,muf,lmf,lmuf,e = Xf # 
    # luf = lu0-lx*tf
    # lvf = lv0-ly*tf
    # dx, dy, du, dv, dm, dmu, dlm, dlmu,edot = EoM(Xf, 1, lx, ly, lu0, lv0, problem, output=False)/tf

    # Hf = lx*uf + ly*vf + luf*du + lvf*dv + lmf*dm #+ lmuf*dmu

    g = [
      # Equality Constraints
        (xf-problem['xf'])/1,
        (yf-problem['yf'])/1,
        (uf-problem['uf'])/1,
        (vf-problem['vf'])/1,
        1*(muf-problem['muf']),   # Ensures a vertical landing, could be automatically satisfied

        # 1*Hf,                     # Free tf, transversality condition
        lmf + 1]                  # Free final mass, so the associated costate is fixed at the final time
    if opt:
        # return sum(np.fabs(g))
        return sum(np.array(g)) # Norm of g, squared
    else:
        return np.array(g)**2


def PMPCostScalar(guess, problem):
    J, c = PMPCost(guess, problem,opt=True)

    return J,c,0


def PMPDE(h, guess):
    problem = getProblem()
    problem['h'] = h

    # bounds = [(0.8*val, 1.2*val) for val in guess]
    bounds = problem['bounds']

    sol = differential_evolution(PMPCost, bounds, args=(problem, True), disp=True, tol=0.01, polish=False)
    # sol = minimize(PMPCost, guess, args=(problem, True), method='Nelder-Mead')

    return sol.x, problem


def PMPOpt():
    problem = getProblem()
    problem['h'] = 1
    guess = problem['sol{}'.format(problem['h'])]


    opt = pyOpt.Optimization('Optimal PDG', lambda c: PMPCostScalar(c, problem))

    opt.addVar('lambdaX',  'c', lower=0, upper=.25, value=guess[0])
    opt.addVar('lambdaY',  'c', lower=0, upper=.25, value=guess[1])
    opt.addVar('lambdaU',  'c', lower=0, upper=3, value=guess[2])
    opt.addVar('lambdaV',  'c', lower=-3, upper=0, value=guess[3])
    opt.addVar('lambdaM',  'c', lower=-1, upper=0, value=guess[4])
    # opt.addVar('lambdaMu', 'c', lower=-10, upper=0, value=guess[5])
    opt.addVar('tf',       'c', lower=guess[6]-1, upper=guess[6]+1, value=guess[6])

    opt.addCon('xf','e')
    opt.addCon('yf','e')
    opt.addCon('vf','e')
    opt.addCon('uf','e')
    opt.addCon('lambdaMf','e')

    opt.addObj('J')

    # optimizer = pyOpt.ALGENCAN()
    # optimizer.setOption('epsfeas',1e-2)
    # optimizer.setOption('epsopt',1e-1)

    optimizer = pyOpt.SLSQP()
    optimizer.setOption('ACC',1e-1)

    fopt, copt, info = optimizer(opt, sens_step=1e-5)

    print opt.solution(0)
    print info

    return copt, problem

def min_time(tf,lam,problem):
    guess = [l for l in lam]
    guess.append(tf)
    return PMPCost(guess,problem,opt=True)


def PMPSolve():
    problem = getProblem()
    problem['h'] = 1
    guess = problem['sol1']
    lam = guess[0:5]
    # guess[-1] -= .2
    # sol = root(PMPCost, guess, args=(problem),tol=1e-4,options={'eps': 1e-8})

    solmin = minimize_scalar(min_time, bracket=(10,13), args=(lam,problem))
    tf = solmin.x
    xsol = [l for l in lam]
    xsol.append(tf)
    # xsol = sol['x']
    # print sol['message']
    # print "Number of function evaluations: {}\n".format(sol['nfev'])
    # print "Solution: {}".format(xsol)

    # xsol = guess
    print "Constraint Satisfaction and Optimality Measures:\n{}".format(PMPCost(xsol, problem))
    return xsol, problem


def ShowSolution(sol,problem):
    import matplotlib.pyplot as plt

    t, x,y, u,v, udot,vdot, m,T, mu,mudot, lx,ly,lu,lv,lm,lmu, s, H,Hp,Hv,Hm,Hmu = PMPParse(sol,problem)

    if np.abs(lu[0]) > 1e-12:
        mu_dot_opt = (lx*lv/lu**2 - ly/lu)*cos(mu)**2
    else:
        mu_dot_opt = np.zeros_like(t)
    # print lv[-1]
    # print (-mudot[-1]*lmu[-1]+lm[-1]*problem['Tmin']/problem['ve'])/(problem['Tmin']/m[-1] - 3.7)
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

    # plt.figure()
    # plt.plot(t,s)
    # plt.plot(t,lu*sin(mu)-lv*cos(mu))
    # plt.title('Switching fun')

    # plt.figure()
    # plt.plot(t,H)
    # for h in [Hp,Hv,Hm,Hmu]:
    #     plt.plot(t,h)
    # plt.title('Hamiltonian')
    # plt.legend(('H','pos','vel','mass','mu'))

    plt.show()    



def PMPParse(guess,problem):
    print "Parsing solution with h = {}:".format(problem['h'])
    g = 3.7
    lx,ly,lu0,lv0,lm0,tf = guess
    lmu0 = 0
    problem['tf'] = tf

    x0 = getInitialState(guess,problem)

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


def Homotopy():
    import matplotlib.pyplot as plt

    problem = getProblem()
    guess = problem['sol0']
    print PMPCost(guess, problem)

    # guess = np.array([0.08042227,   0.12706492,   2.66632126,  -0.36258304,  -0.75506248,  -9.26403713,  12.26226516])
    # problem['h'] = 1
    # print PMPCost(guess,problem)

    h = 0
    hf = 1  # Final value of h (should be 1 for full homotopy solution)
    Ilow = 15  # Reasonable number of iterations % for root, 15 and 75 work well; for neldermead,200,500
    Ihigh = 1500
    dh = 0.01  # Initial stepsize
    dh_max = 0.1
    dh_min = 0.0001
    guess = problem['sol0']
    h += dh
    method =  'hybr' #'lm' #
    history = {'h': [0], 'sol': [guess]}

    while True:

        problem['h'] = h
        print "Current step: {}".format(h)
        print "Current stepsize: {}".format(dh)
        print "Current guess: {}".format(guess)
        sol = root(PMPCost, guess, args=(problem), tol=1e-4, method=method, options={'eps': 1e-10, 'diag':(.1,.1,1,1,1,1)})
        # sol = minimize(PMPCost, guess, args=(problem, True), method='TNC',bounds=problem['bounds'])

        print sol['message']

        # if np.any(np.abs(sol['x'])>1e3):
        #     print "Homotopy aborted because the solution is diverging."
        #     return sol['x'], problem

        if sol['success']:
            history['h'].append(h)
            history['sol'].append(sol['x'])
            if np.abs(h - hf) <= 1e-6:
                print "Final Solution (h={}): {}".format(h,sol['x'])
                return sol['x'], problem
            guess = sol['x']
            print "Number of function evaluations: {}".format(sol['nfev'])
            print "New guess: {}\n".format(guess)
            # Update the stepsize:
            if sol['nfev'] <= Ilow:
                print 'Increasing stepsize due to low iterations'
                dh *= 1.8
            elif sol['nfev'] >= Ihigh:
                dh *= 0.5
            dh = sat(dh,dh_min,np.min((dh_max,hf-h)))
            h += dh

        else:
            if dh == dh_min:
                print "Homotopy aborted because the last step failed and stepsize is equal to minimum stepsize."
                return sol['x'], problem
            print "Decreasing stepsize from {} to {} and rerunning current guess.\n".format(dh,dh/2.0)
            h = np.max((0,h-0.5*dh))
            dh *= 0.5
            dh = sat(dh,dh_min,dh_max)

        sols = np.vstack(history['sol'])
        plt.plot(history['h'], sols)




def test_root():

    for method in ['hybr','lm','broyden1','broyden2','anderson']:
        print method
        sol = root(nonsmooth_fun, [30,10], tol=1e-5, method=method)
        print sol['x']
    return None


def nonsmooth_fun(x):
    if x[1] < 0 :
        g2 = x[1]*(x[1]-1)
    else:
        g2 = -2*x[1]*(x[1]+1)
    return np.fabs(x[0]**2-9), g2

def FD(guess):
    problem = getProblem()
    problem['h'] = 1
    dtf = 1e-6
    # guess = [-0.13605246,  0.27659359,  2.09970539,  0.28434616, -0.68900922,  9.95517178]

    sol = PMPCost(guess, problem)

    guess[5] += dtf
    sol2 = PMPCost(guess,problem)

    dgdx = (sol2-sol)/dtf
    print "Gradient wrt tf: {}".format(dgdx)

if __name__ == '__main__':
    # Sol, prob = PMPSolve()
    # sol = Sol #Sol['x']

    sol, prob = Homotopy()

    # sol, prob = PMPOpt()
    # prob = getProblem()

    # prob['h'] = 1
    # sol = prob['sol{}'.format(prob['h'])]
    # test_root()
    # sol, prob = PMPDE(1, 0)
    ShowSolution(sol, prob)
    print "Best solution:"
    FD(prob['sol1'])
    print "Local Minimum:"
    FD(sol)
