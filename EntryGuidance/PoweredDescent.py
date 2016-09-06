from numpy import sin,cos
import numpy as np
from scipy.integrate import odeint, ode
from scipy.optimize import root

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
    
    problem['m0'] =  8500
    problem['ve'] = isp*9.81
    thrust = 600000
    problem['Tmax'] = thrust
    problem['Tmin'] = thrust*0.1  # 10% throttle
    
    V0 = (problem['u0']**2 + problem['v0']**2)**0.5
    fpa0 = np.arcsin(problem['v0']/V0)
    problem['mu0'] = np.pi+fpa0
    problem['mudotmax'] = 40*np.pi/180  # 40 deg/s

    problem['gain'] = 50

    problem['tf'] = 12.28  # An initial guess at the final time
    return problem



# #################### #
# Pontryagin Approach  #
# #################### #

def  EoM(X,t, lx,ly,lu0,lv0, problem,output=False):
    # Defines the equations of motion for 2-D powered flight and the associated costates.
    # Only one costate needs to be integrated numerically.

    g = 3.7

    x, y, u, v, m, mu, lm = X
    
    lu = lu0-lx*t
    lv = lv0-ly*t

    tTurn = np.abs(np.pi/2.0-mu)/problem['mudotmax']
    if (t+tTurn) >= problem['tf']:
        target = np.pi/2.0
        dmu = np.sign(target-mu)*problem['mudotmax']
    else:
        target = np.pi+np.arctan2(lv, lu)
        dmu = sat(problem['gain']*(target-mu), -problem['mudotmax'], problem['mudotmax'])

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

    if output:
        return T, dmu
    else:
        return np.array([dx, dy, du, dv, dm, dmu, dlm])
    


def PMPCost(guess,problem):

    lx,ly,lu0,lv0,lm0,tf = guess
    problem['tf'] = tf
    x0 = np.array([problem['x0'],problem['y0'],problem['u0'],problem['v0'],problem['m0'],problem['mu0'],lm0])
    X = odeint(EoM, x0, np.linspace(0,tf,500), args=(lx, ly, lu0, lv0, problem))

    imax = X.shape[0]
    for i in range(imax):
        if np.all(X[i, :] == 0.0):
            imax = i
            break

    Xf = X[imax-1,:]
    xf,yf,uf,vf,mf,muf,lmf = Xf
    luf = lu0-lx*tf
    lvf = lv0-ly*tf
    dx, dy, du, dv, dm, dmu, dlm = EoM(Xf, tf, lx, ly, lu0, lv0, problem, output=False)

    H = lx*uf + ly*vf + luf*du + lvf*dv + lmf*dm # Ignores the lambda_mu term since we can set the final mu_dot to 0
    
    g = [
      # Equality Constraints
      xf-problem['xf'],
      yf-problem['yf'],
      uf-problem['uf'],
      vf-problem['vf'],
      # H,                      # Free tf, transversality condition
      muf-problem['muf'],   # Ensures a vertical landing, could be automatically satisfied
      lmf + 1]

    return g


def PMPOpt():
    import matplotlib.pyplot as plt

    problem = getProblem()
    # guess = [0.0793,0.1265,2.66,-0.3638,-0.755,12.28]
    guess = [g*1.0 for g in [0.0793,0.1265,2.66,-0.3638,-0.755,12.9]]
    sol = root(PMPCost, guess, args=(problem))
    xsol = sol['x']
    print sol['message']
    print "Number of function evaluations: {}".format(sol['nfev'])

    # xsol = guess

    t, x,y, u,v, udot,vdot, m,T, mu,mudot, lx,ly,lu,lv,lm, s, H = PMPParse(xsol,problem)


    print("Prop used: {} kg".format(m[0]-m[-1]))

    plt.figure()
    plt.plot(x,y)
    plt.title('Positions')
    
    plt.figure()
    plt.plot(u,v)
    plt.title('Velocities')
    
    plt.figure()
    plt.plot(udot,vdot)
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
    plt.title('Mu dot')
    
    plt.figure()
    for l in [lx,ly,lu,lv,lm]:
        plt.plot(t,l)
    plt.title('Costates')

    # plt.figure()
    # plt.plot(t,s)
    # plt.title('Switching fun')

    plt.figure()
    plt.plot(t,H)
    plt.title('Hamiltonian (Approx, no lambda mu term)')

    plt.show()    


def PMPParse(guess,problem):

    g = 3.7
    lx,ly,lu0,lv0,lm0,tf = guess
    problem['tf'] = tf
    x0 = np.array([problem['x0'],problem['y0'],problem['u0'],problem['v0'],problem['m0'],problem['mu0'],lm0])
    t = np.linspace(0, tf, 500)

    X = odeint(EoM, x0, t, args = (lx,ly,lu0,lv0,problem))

    imax = len(t)
    for i in range(len(t)):
        if np.all(X[i, :] == 0.0):
            imax = i
            print "Removing {} elements from final solution.".format(len(t)-imax)
            break

    t = t[0:imax]
    x, y, u, v, m, mu, lm= X[0:imax, 0], X[0:imax, 1], X[0:imax, 2], X[0:imax, 3], X[0:imax, 4], X[0:imax, 5], X[0:imax, 6]
    
    lu = lu0-lx*t
    lv = lv0-ly*t
    mu_dot = np.zeros_like(t)
    T = np.zeros_like(t)
    s = (lu*cos(mu)+lv*sin(mu)) - m*lm/problem['ve']

    for i in range(len(t)):
        T[i], mu_dot[i] = EoM(X[i,:],t[i], lx, ly, lu[0], lv[0], problem, output=True)

    udot = T*cos(mu)/m
    vdot = T*sin(mu)/m-g

    H = lx * u + ly * v + lu * udot + lv * vdot + lm * -T/problem['ve']


    return t,x,y,u,v,udot,vdot,m,T,mu,mu_dot, lx*np.ones_like(t),ly*np.ones_like(t),lu,lv,lm, s, H
    
# ########################### #
# Polynomial Based Approaches #
# ########################### #

if __name__=='__main__':
    PMPOpt()