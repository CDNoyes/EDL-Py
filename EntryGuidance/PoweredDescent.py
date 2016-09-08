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

    problem['tf'] = None  
    return problem



# #################### #
# Pontryagin Approach  #
# #################### #

def  EoM(X,t, lx,ly,lu0,lv0, problem,output=False):
    # Defines the equations of motion for 2-D powered flight and the associated costates.
    # Only one costate needs to be integrated numerically.

    g = 3.7

    x, y, u, v, m, mu, lm, lmu = X
    
    lu = lu0-lx*t
    lv = lv0-ly*t

    # if np.abs(lmu) <= 1:
        # target = np.pi+np.arctan2(lv, lu)
        # dmu = sat(problem['gain']*(target-mu), -problem['mudotmax'], problem['mudotmax'])
    # else:
        # dmu = -problem['mudotmax']*np.sign(lmu)
        
    tTurn = np.abs(np.pi/2.0-mu)/problem['mudotmax']
    if (t+tTurn-0.03) >= problem['tf']:
        target = np.pi/2.0
        # dmu = np.fmin(0,np.sign(target-mu)*problem['mudotmax'])
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
    dlmu = T*(lu*sin(mu)-lv*cos(mu))/m
    
    if output:
        return T, dmu
    else:
        return np.array([dx, dy, du, dv, dm, dmu, dlm, dlmu])
    


def PMPCost(guess,problem):

    lx,ly,lu0,lv0,lm0,lmu0,tf = guess
    problem['tf'] = tf
    x0 = np.array([problem['x0'],problem['y0'],problem['u0'],problem['v0'],problem['m0'],problem['mu0'],lm0,lmu0])
    X = odeint(EoM, x0, np.linspace(0,tf+0.03,500), args=(lx, ly, lu0, lv0, problem))

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
    dx, dy, du, dv, dm, dmu, dlm, dlmu = EoM(Xf, tf, lx, ly, lu0, lv0, problem, output=False)
    dx0, dy0, du0, dv0, dm0, dmu0, dlm0, dlmu0 = EoM(Xf, tf, lx, ly, lu0, lv0, problem, output=False)

    Hf = lx*uf + ly*vf + luf*du + lvf*dv + lmf*dm + lmuf*dmu 
    H0 = lx*problem['u0'] + ly*problem['v0'] + luf*du0 + lvf*dv0 + lmf*dm0 + lmuf*dmu 
    
    g = [
      # Equality Constraints
      xf-problem['xf'],
      yf-problem['yf'],
      uf-problem['uf'],
      vf-problem['vf'],
      0*Hf,                      # Free tf, transversality condition
      0*(muf-problem['muf']),   # Ensures a vertical landing, could be automatically satisfied
      lmf + 1]

    return g


def PMPOpt():
    import matplotlib.pyplot as plt

    problem = getProblem()
    # guess = [0.0793,0.1265,2.66,-0.3638,-0.755,12.28]
    # guess = [0.0804911135750877, 0.126960707251939, 2.67099166231944, -0.363124805249400,-0.754774141665489,-9.26285943305668,12.263]
    guess = [0.08042227,   0.12706492,   2.66632126,  -0.36258304,  -0.75506248,  -9.26403713,  12.26226516]

    # guess = [g*1.0 for g in [0.0799,0.1265,2.66,-0.3638,-0.755,-9.305,12.28]]
    
    sol = root(PMPCost, guess, args=(problem))
    xsol = sol['x']
    print sol['message']
    print "Number of function evaluations: {}".format(sol['nfev'])
    print xsol
    # xsol = guess
    # problem['tf'] = guess[-1]
    
    t, x,y, u,v, udot,vdot, m,T, mu,mudot, lx,ly,lu,lv,lm,lmu, s, H,Hp,Hv,Hm,Hmu = PMPParse(xsol,problem)

    mu_dot_exp = sat(-T/(m*problem['ve'])*(lu*cos(mu)+lv*sin(mu))/(lu*sin(mu)-lv*cos(mu)),-problem['mudotmax'],-problem['mudotmax'])

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
    plt.plot(t,mu_dot_exp*180/np.pi)
    plt.legend(('True','Hdot=0'))
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
    for h in [Hp,Hv,Hm,Hmu]:
        plt.plot(t,h)
    plt.title('Hamiltonian')
    plt.legend(('H','pos','vel','mass','mu'))
    

    plt.show()    


def PMPParse(guess,problem):
    print "Parsing solution"
    g = 3.7
    lx,ly,lu0,lv0,lm0,lmu0,tf = guess
    problem['tf'] = tf
    x0 = np.array([problem['x0'],problem['y0'],problem['u0'],problem['v0'],problem['m0'],problem['mu0'],lm0,lmu0])
    t = np.linspace(0, tf+0.03, 500)

    X = odeint(EoM, x0, t, args = (lx,ly,lu0,lv0,problem))

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
        T[i], mu_dot[i] = EoM(X[i,:],t[i], lx, ly, lu[0], lv[0], problem, output=True)

    udot = T*cos(mu)/m
    vdot = T*sin(mu)/m-g

    H = lx * u + ly * v + lu * udot + lv * vdot + lm * -T/problem['ve'] + lmu*mu_dot
    Hp = lx * u + ly * v
    Hv = lu * udot + lv * vdot
    Hm = lm * -T/problem['ve']
    Hmu = lmu*mu_dot

    return t,x,y,u,v,udot,vdot,m,T,mu,mu_dot, lx*np.ones_like(t),ly*np.ones_like(t),lu,lv,lm,lmu, s, H, Hp,Hv,Hm,Hmu
    
# ########################### #
# Polynomial Based Approaches #
# ########################### #

if __name__=='__main__':
    PMPOpt()