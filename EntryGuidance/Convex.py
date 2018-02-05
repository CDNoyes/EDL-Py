import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
import time
from scipy.integrate import odeint
from scipy.interpolate import interp1d

def LTV(x0, A, B, tf, f_ref, x_ref, trust_region=0.5, P=0, xf=0, umax=3):
    """ Solves a convex LTV subproblem

    tf - known final time
    A - state linearization around x_ref
    B - control linearization around x_ref
    f_ref - the original nonlinear dynamics evaluated along x_ref
    x_ref - the trajectory used to linearize the problem, typically the current iterate

    x0 - initial condition


    """

    n = A[0].shape[0]
    m = 1
    N = A.shape[0]
    T = range(N-1)
    dt = float(tf)/(N-1)
    assert(dt>0)

    x = cvx.Variable(n, N)
    u = cvx.Variable(m, N-1)
    if P > 0:
        bc = [x[:,0] == x0, x[:,-1] <= xf+P*1, x[:,-1] >= xf-P*1]
    else:
        bc = [x[:,-1] == xf, x[:,0] == x0]

    states = []
    for t,a,b,f,xr in zip(T,A,B,f_ref,x_ref):
        cost = dt*cvx.abs(u[:,t]) #dt*cvx.sum_squares(u[:,t])#
        constr = [x[:,t+1] == x[:,t] + f*dt + (dt*a)*(x[:,t]-xr) + dt*b*u[:,t],
                  cvx.norm(u[:,t], 'inf') <= umax,
                  cvx.norm((x[:,t]-xr)) < trust_region]
        states.append( cvx.Problem(cvx.Minimize(cost), constr) )

    # sums problem objectives and concatenates constraints.
    prob = sum(states) #+ P*cvx.Problem(cvx.Minimize(cvx.sum_squares(x[:,-1])))
    prob.constraints += bc

    t0 = time.time()
    prob.solve()
    t1 = time.time()
    print "status:        ", prob.status
    print "optimal value: ", np.around(prob.value,3)
    print "solution time:  {} s\n".format(t1-t0)

    return x.value.A, u.value.A


def test():
    """ Solves a control constrained VDP problem to compare with GPOPS """

    mu = 1
    def dynamics(x):
        # returns f,g evaluated at x (vectorized)
        return np.array([x[1],-x[0] + mu*(1-x[0]**2)*x[1]]),np.vstack((np.zeros_like(x[0]),np.ones_like(x[0]))).squeeze()

    def dyn(x,t,u): # integrable function
        f,g = dynamics(x)
        return f + g*u(t)

    def jac(x):
        x1,x2=x
        shape = [x.shape[0]]
        shape.extend(x.shape)
        A = np.zeros(shape)
        # B = np.zeros((2,1,shape[2])))

        B = np.vstack((np.zeros_like(x1),np.ones_like(x1))).squeeze()

        A[0,1,:] = np.ones_like(x[0])
        A[1,0,:] = -np.ones_like(x[0]) -2*mu*x1*x2
        A[1,1,:] = mu*(1-x[0]**2)

        return np.moveaxis(A, -1, 0),np.moveaxis(B, -1, 0)

    def integrate(x0, u, tf, N, return_u=False):
        # X = [x0]
        t = np.linspace(0,tf,u.shape[0]+1)
        dt = t[1]
        U = np.append(u,u[-1])
        ut = interp1d(t,U,kind='cubic',assume_sorted=True,fill_value=u[-1],bounds_error=False)

        t_fine = np.linspace(0,tf,N)
        X = odeint(dyn,x0,t_fine,args=(ut,))
        # for ut in u:
            # X.append(odeint(dyn, X[-1], [0,dt] , args=(ut,))[-1])
        if return_u:
            return np.asarray(X),ut(t_fine).T
        else:
            return np.asarray(X)






    X = []
    X_cvx = []
    U = []
    tf = 5
    x0 = [-3,0]
    N = 200
    u = np.zeros((N-1,1))
    # ti = [0,2,3,4,5,6,8,10,12]
    # ui = [0,0,0,3,3,0,0,0,0]
    t = np.linspace(0,tf,N)
    # u = interp1d(ti,ui, kind='cubic')(t[:-1])
    # x = integrate(x0, u, tf=tf, N=N).T
    # A,B = jac(x)

    iters = 5

    # P = np.logspace(-1, iters-1,iters)
    P = np.linspace(1,0.1,iters)
    for it in range(iters):
        U.append(u)

        x = integrate(x0, u, tf=tf, N=N).T
        f,g = dynamics(x)
        A,B = jac(x)
        # F = f + g*u

        X.append(x)
        umax = 3 #+ 3*(iters-1-it)
        x_approx,u = LTV(x0, A, B, tf, f.T, x.T, trust_region=4, P=P[it], umax=umax )
        X_cvx.append(x_approx)

        u = u.T


    U.append(u)
    x = integrate(x0, u, tf=tf, N=1*N).T # Can increase fidelity here or after optimal solution found
    X.append(x)

    try:
        print "Performing final iteration - hard enforcement of constraints, and additional discretization points"
        f,g = dynamics(x)
        A,B = jac(x)
        x_approx,u = LTV(x0, A, B, tf, f.T, x.T,trust_region=2,P=0,umax=3)

        u = u.T
        print u.shape
        x,u = integrate(x0, u, tf=tf, N=1*N,return_u=True)
        X.append(x.T)
        X_cvx.append(x_approx)
        X_cvx.append(x_approx) # just so theyre the same length
        print u.shape
        U.append(u)
    except:
        print "Failed to solve final iteration."

    for i,xux in enumerate(zip(X,U,X_cvx)):
        x,u,xc = xux
        plt.figure(1)
        plt.plot(x[0],x[1],label=str(i))
        plt.title('Integration')
        plt.figure(5)
        plt.plot(xc[0],xc[1],label=str(i))
        plt.title('Discretization')
        plt.figure(2)
        plt.step(np.linspace(0,tf,u.shape[0]),u)

    plt.figure(3)
    plt.plot(X[-1][0],X[-1][1],label='Integration')
    plt.plot(X_cvx[-1][0],X_cvx[-1][1],label='Convex discretization')
    plt.legend()

    plt.figure(4)
    plt.step(np.linspace(0,tf,U[-1].shape[0]),U[-1])
    plt.title('optimal control')

    plt.figure(1)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test()
