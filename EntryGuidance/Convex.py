import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
import time
from scipy.integrate import odeint


def LTV(x0, A, B, tf, f_ref, x_ref, trust_region=0.5, P=0, xf=0, umax=3):
    """ Solves a convex LTV subproblem

    tf - known final time
    A - state linearization around x_ref
    B - control linearization around x_ref
    f_ref - the original nonlinear dynamics evaluated along x_ref
    x_ref - the trajectory used to linearize the problem, typically the current iterate

    x0 - initial condition


    """

    # import pdb
    # pdb.set_trace()

    n = A[0].shape[0]
    m = 1
    N = A.shape[0]
    T = range(N-1)
    dt = float(tf)/(N-1)
    assert(dt>0)

    x = cvx.Variable(n, N)
    u = cvx.Variable(m, N-1)
    if P > 0:
        bc = [x[:,0] == x0]
    else:
        bc = [x[:,-1] == xf, x[:,0] == x0]

    states = []
    for t,a,b,f,xr in zip(T,A,B,f_ref,x_ref):
        cost = dt*cvx.abs(u[:,t])
        constr = [x[:,t+1] == x[:,t] + f*dt + (dt*a)*(x[:,t]-xr) + dt*b*u[:,t],
                  cvx.norm(u[:,t], 'inf') <= umax,
                  cvx.norm((x[:,t]-xr)) < trust_region]
        states.append( cvx.Problem(cvx.Minimize(cost), constr) )

    # sums problem objectives and concatenates constraints.
    prob = sum(states) + P*cvx.Problem(cvx.Minimize(cvx.sum_squares(x[:,-1])))
    prob.constraints += bc

    t0 = time.time()
    prob.solve()
    t1 = time.time()
    print "status:           ", prob.status
    print "optimal value:    ", prob.value
    print "solution time: {}s".format(t1-t0)

    return x.value.A, u.value.A

def Fixed(x0, f_ref, g_ref, tf, umax=1):
    """ Solves a convex subproblem with the nonlinear dynamics evaluated at some reference values

    dx = f(x) + g(x)u dt

    tf - known final time
    f_ref - the original nonlinear dynamics evaluated along x_ref
    g_ref - the original nonlinear dynamics evaluated along x_ref
    References should be evaluated at N equally spaced time points

    x0 - initial condition


    """
    n = f_ref[0].shape[0]
    m = 1 #g_ref[0].shape[1]
    N = f_ref.shape[0]
    T = range(N)
    dt = float(tf)/N
    assert(dt>0)

    x = cvx.Variable(n, N+1)
    u = cvx.Variable(m, N)

    states = []
    for t,f,g in zip(T,f_ref,g_ref):
        cost = dt*cvx.abs(u[:,t])
        print g
        constr = [x[:,t+1] == x[:,t] + dt*( f + g*u[:,t]),
                  cvx.norm(u[:,t], 'inf') <= umax]
        states.append( cvx.Problem(cvx.Minimize(cost), constr) )
    # sums problem objectives and concatenates constraints.
    prob = sum(states)
    # prob.constraints += [x[:,0] == x0]
    prob.constraints += [x[:,T] == 0, x[:,0] == x0]
    t0 = time.time()
    prob.solve()
    t1 = time.time()
    print "status:           ", prob.status
    print "optimal value:    ", prob.value
    print "solution time: {}s".format(t1-t0)

    return x.value.A,u.value.A


def test():
    """ Solves a control constrained VDP problem to compare with GPOPS """

    x = np.random.random((100,2)).T # two states at 100 times
    mu = 1
    def dynamics(x):
        # returns f,g evaluated at x (vectorized)
        return np.array([x[1],-x[0] + mu*(1-x[0]**2)*x[1]]),np.vstack((np.zeros_like(x[0]),np.ones_like(x[0]))).squeeze()

    def dyn(x,t,u):
        f,g = dynamics(x)

        return f + g*u

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

    def integrate(x0, u, tf, N):
        X = [x0]
        dt = np.linspace(0,tf,N-1)[1]
        for ut in u:
            X.append(odeint(dyn, X[-1], [0,dt] , args=(ut,))[-1])

        return np.array(X)






    X = []
    X_cvx = []
    U = []
    tf = 5
    x0 = [-3,0]
    N = 200
    u = np.zeros((N-1,1))
    # x = integrate(x0, u, tf=tf, N=N).T
    A,B = jac(x)

    iters = 5
    for _ in range(iters):
        U.append(u)

        x = integrate(x0, u, tf=tf, N=N).T
        f,g = dynamics(x)
        A,B = jac(x)
        # print x.shape
        # print f.shape
        # print g.shape

        X.append(x)
        umax = 3 + 3*(iters-1-_)
        print umax
        x_approx,u = LTV(x0, A, B, tf, f.T, x.T, trust_region=5, P=0, umax=umax )
        X_cvx.append(x_approx)
        # x_approx,u = LTV(x0, A, B, tf, f.T, x.T, trust_region=3, P=2**_, umax=umax )
        # print x_approx.shape
        # print u.shape
        u = u.T
        # x_approx,u = Fixed(x0, f.T, g.T, tf,umax=3) # check converge of x-x_approx at some point
        # import pdb
        # pdb.set_trace()

    U.append(u)
    x_approx,u = LTV(x0, A, B, tf, f.T, x.T,trust_region=1,P=0,umax=3)

    u = u.T
    x = integrate(x0, u, tf=tf, N=N).T
    X.append(x)
    X_cvx.append(x_approx)
    U.append(u)

    for i,xu in enumerate(zip(X,U)):
        x,u=xu
        plt.figure(1)
        plt.plot(x[0],x[1],label=str(i))

        plt.figure(2)
        plt.plot(u)

    plt.figure(3)
    plt.plot(X[-1][0],X[-1][1])
    plt.plot(X_cvx[-1][0],X_cvx[-1][1])

    plt.figure(1)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test()
