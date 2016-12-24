import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
# from scipy.optimize import minimize, differential_evolution
import pyOpt
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d

def ChebyshevDiff(n):
# %Returns the (Sorted) Chebyshev Differentiation Matrix, D, with n nodes 
# %using the collocated points t.
# %
# %Unsorted collocation points are used to compute the standard
# %differentiation matrix, then the matrix is converted to its sorted form.
# %See Ross, Fahroo (2002) for details. 
 
    t = cos(pi*np.arange(0,n+1)/n) # %Unsorted collocation points
    D = np.empty((n+1,n+1))
     
    for j in range(n+1):
        for k in range(n+1):
            if j == 0 and j == k:
                D[k,j] = (2*n**2+1)/6.
            elif j == n and j ==k:
                D[k,j] = -(2*n**2+1)/6.
            elif j == k:
                D[k,j] = -t[k]/2./(1-t[k]**2)
            else:
                if k == 0 or k == n:
                    ck = 2.
                else:
                    ck = 1.

                if j == 0 or j == n:
                    cj = 2.
                else:
                    cj = 1.
                D[k,j] = (ck/cj)*( ( (-1)**(j+k) ) / (t[k]-t[j]) )
                
    return -D

def Cost(c,problem):

    g = 3.7
    ve = problem['ve']
    D = problem['D']
    N = problem['N']
    x = np.hstack((problem['x0'], c[0:N], problem['xf']))
    y = np.hstack((problem['y0'], c[N:N*2], problem['yf']))
    tf = c[-1]
    D = D*2/tf
    tol = 1    

    tau = -cos(pi*np.arange(0,N+2)/(N+1))
    t = (tau+1)*0.5*tf
    u = np.dot(D,x)
    v = np.dot(D,y)
    udot = np.dot(D,u)
    vdot = np.dot(D,v)
    
    if np.any(np.iscomplex(vdot)) or np.any(np.iscomplex(udot)):
        mu = np.arctan((vdot+g)/udot)
    else:
        mu = np.arctan2(vdot+g,udot)
    mudot = np.dot(D,mu)
    r = -(g+vdot)/(ve*sin(mu))
    m = problem['m0']*np.exp(cumtrapz(r,t,initial=0))
    T = m*(udot)/(cos(mu))
    T = interp1d(tau,T)(np.linspace(-1,1,problem['nConstraint'])) # Interpolate to finer grid for better constraint satisfaction
    mudot = interp1d(tau,mudot)(np.linspace(-1,1,problem['nConstraint'])) # Interpolate to finer grid for better constraint satisfaction
    
    g = [ 
          #Six Equality Constraints - Independent of the order of the solution
          u[0]-problem['u0'],
          v[0]-problem['v0'],
          u[-1]-problem['uf'],
          v[-1]-problem['vf'],
          mu[0] - problem['mu0'],
          udot[-1]-problem['udotf']] #Ensures a vertical landing
     
      #Inequality Constraints on thrust magnitude - Dependent on order of solution
    g.extend(mudot - problem['mudotmax'])
    g.extend(problem['mudotmin'] - mudot)
    g.extend(T-problem['Tmax'])    
    g.extend(problem['Tmin']-T)
            
            
    fail = 0
    return -m[-1], g, fail
    
    
    
def Opt():
    problem = {}
    # Problem Solution Info #
    order = 5 # Order should be kept relatively low <=6, if more accuracy is required, increase the number of partitions
    N = order-1
    problem['N'] = N
    problem['nConstraint'] = 10
    problem['nDivisions'] = 1 # number of segments, each fitted with its own polynomial of the specified order 
    # Problem Info #
    isp = 290
    
    problem['x0'] = -3200
    problem['xf'] = 0
    
    problem['y0'] = 2000
    problem['yf'] = 0
    
    problem['u0'] = 625
    problem['uf'] = 0
    
    problem['v0'] = -270
    problem['vf'] = 0
    
    problem['udotf'] = 0
    
    problem['m0'] =  8500
    problem['ve'] = isp*9.81
    thrust = 600000
    problem['Tmax'] = thrust
    problem['Tmin'] = thrust*0.1 # 10% throttle
    
    V0 = (problem['u0']**2 + problem['v0']**2)**0.5
    fpa0 = np.arcsin(problem['v0']/V0)
    problem['mu0'] = np.pi+fpa0
    problem['mudotmax'] = 40*np.pi/180 # 40 deg/s
    problem['mudotmin'] = -problem['mudotmax']
    
    # Initial Guess
    tf = 12
    x = np.linspace(problem['x0'],problem['xf'],order+1)
    y = np.linspace(problem['y0'],problem['yf'],order+1)
    # tau = -cos(pi*np.arange(0,N+2)/(N+1))
    # t = (tau+1)*0.5*tf
    # x = interp1d(np.linspace(0,tf,order+1),x)(t)
    # y = interp1d(np.linspace(0,tf,order+1),y)(t)
    c0 = np.hstack((x[1:-1],y[1:-1],tf))
    
    # Form D
    problem['D'] = ChebyshevDiff(order)

    opt = pyOpt.Optimization('Flat Pseudospectral PDG',lambda c: Cost(c,problem))
    
    
    # Add the design variables
    for i,xi in enumerate(x[1:-1]):
        opt.addVar('x{}'.format(i+1), 'c', lower = problem['x0'], upper = problem['xf'], value = xi)
    
    for i,xi in enumerate(y[1:-1]):
        opt.addVar('y{}'.format(i+1), 'c', lower = problem['yf'], upper = problem['y0'], value = xi)
    
    opt.addVar('tf','c', lower = 5, upper = 50, value=tf)
    
    # Add the objective and constraints
    opt.addObj('J')
    
    for i in range(1,7):
        opt.addCon('g{}'.format(i),'e')
          
          
    for i in range(1,4*problem['nConstraint'] + 0*order):
        opt.addCon('h{}'.format(i),'i')
        
    # optimizer = pyOpt.COBYLA()
    # optimizer = pyOpt.ALPSO()
    optimizer = pyOpt.ALGENCAN()
    # optimizer = pyOpt.SLSQP()
    # optimizer = pyOpt.SDPEN()
    # optimizer = pyOpt.PSQP()
    # optimizer = pyOpt.SOLVOPT()
    
    
    sens_type = 'CS' # Differencing Type, options ['FD', CS']
    # optimizer.setOption('MAXIT',100) #SLSQP option
    # optimizer.setOption('MIT',200) # PSQP
    # fopt,copt,info = optimizer(opt,sens_type=sens_type)
    fopt,copt,info = optimizer(opt)

    print info
    print opt.solution(0)

    t,x,y,u,v,udot,vdot,m,T,mu,mudot = Parse(copt,problem)


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
    plt.title('Thrust Angle')
    
    plt.figure()
    plt.plot(t,T)
    plt.title('Thrust')
    
    plt.figure()
    plt.plot(t,x)
    
    plt.figure()
    plt.plot(t,y)
    
    plt.show()    
   
def Parse(c,problem):

    g = 3.7
    ve = problem['ve']
    D = problem['D']
    N = problem['N']
    x = np.hstack((problem['x0'], c[0:N], problem['xf']))
    y = np.hstack((problem['y0'], c[N:N*2], problem['yf']))
    tf = c[-1]
    D = D*2/tf

    tau = -cos(pi*np.arange(0,N+2)/(N+1))
    t = (tau+1)*0.5*tf
    u = np.dot(D,x)
    v = np.dot(D,y)
    udot = np.dot(D,u)
    vdot = np.dot(D,v)
        
    mu = np.arctan2(vdot+g,udot)
    mudot = np.dot(D,mu)
    r = -(g+vdot)/(ve*sin(mu))
    m = problem['m0']*np.exp(cumtrapz(r,t,initial=0))
    T = m*(udot)/(cos(mu))

    tinterp = np.linspace(0,tf)
    X = interp1d(t,np.vstack((x,y,u,v,udot,vdot,m,T,mu,mudot)).T,'cubic',axis=0)(tinterp)
    x,y,u,v,udot,vdot,m,T,mu,mudot = X[:,0], X[:,1], X[:,2], X[:,3], X[:,4], X[:,5], X[:,6], X[:,7], X[:,8],X[:,9]
    return tinterp,x,y,u,v,udot,vdot,m,T,mu,mudot
    
if __name__ == '__main__':    
    Opt()
    