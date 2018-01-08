import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
import time
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from Mesh import Mesh

def LTV(x0, A, B, f_ref, x_ref, mesh, trust_region=0.5, P=0, xf=0, umax=3):

    """ Solves a convex LTV subproblem

    A - state linearization around x_ref
    B - control linearization around x_ref
    f_ref - the original nonlinear dynamics evaluated along x_ref
    x_ref - the trajectory used to linearize the problem, typically the current iterate

    x0 - initial condition
    mesh - Mesh.Mesh instance

    """

    n = A[0].shape[0]
    m = 1
    N = mesh.n_points
    T = range(N)

    # x = cvx.Variable(N,n)
    x = np.array([cvx.Variable(n) for _ in range(N)]).T
    u = cvx.Variable(N,m)



    X = mesh.chunk(x)
    A = mesh.chunk(A)
    B = mesh.chunk(B)
    F = mesh.chunk(f_ref)
    Xr = mesh.chunk(x_ref)
    U = mesh.chunk(u)

    if P > 0:
        bc = [x[0] == x0, x[-1] <= xf+P*1, x[-1] >= xf-P*1]
    else:
        bc = [x[-1] == xf, x[0] == x0]


    constr = []
    for t,xr in zip(T,x_ref):
        constr += [cvx.norm((x[t]-xr)) < trust_region]

    # Control constraints
    Cu = cvx.abs(u) <= umax

    # Lagrange cost and ode constraints
    states = []
    for d,xi,f,a,b,xr,ui,w in zip(mesh.diffs,X,F,A,B,Xr,U,mesh.weights):
        L = cvx.abs(ui)                                         # Lagrange integrands for a single mesh
        cost = w*L                                              # Clenshaw-Curtis quadrature
        dx = d.dot(xi)
        ode =  [dxi == fi + ai*(xii-xri) + bi*uii  for xii,fi,ai,bi,xri,uii,dxi in zip(xi,f,a,b,xr,ui,dx) ] # Differential equation constraints
        states.append(cvx.Problem(cvx.Minimize(cost), ode))

    # sums problem objectives and concatenates constraints.
    prob = sum(states) #+ P*cvx.Problem(cvx.Minimize(cvx.sum_squares(x[:,-1])))
    prob.constraints.append(Cu)
    prob.constraints += bc
    prob.constraints += constr

    t0 = time.time()
    prob.solve()
    t1 = time.time()
    print "status:        ", prob.status
    print "optimal value: ", np.around(prob.value,3)
    print "solution time: {}s\n".format(t1-t0)

    try:
        x_sol = np.array([xi.value.A for xi in x]).squeeze()
        u_sol = u.value.A.squeeze()
        return x_sol.T, u_sol
    except:
        return x_ref.T,np.zeros_like(T)


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

    def integrate(x0, u, t, return_u=False):

        ut = interp1d(t,u,kind='cubic',assume_sorted=True,fill_value=u[-1],bounds_error=False)
        X = odeint(dyn,x0,t,args=(ut,))

        if return_u:
            return np.asarray(X),u
        else:
            return np.asarray(X)






    X = []
    X_cvx = []
    U = []

    tf = 10
    mesh = Mesh(tf=tf)
    t = mesh.times
    x0 = [3,1]
    u = np.zeros_like(t)
    # x = integrate(x0, u, t).T
    # A,B = jac(x)

    # plt.plot(x[0],x[1])
    # plt.show()
    iters = 10

    # P = np.logspace(-1, iters-1,iters)
    P = np.linspace(1,0.01,iters)
    for it in range(iters):
        U.append(u)

        x = integrate(x0, u, t).T
        f,g = dynamics(x)
        A,B = jac(x)

        X.append(x)
        umax = 3 #+ 3*(iters-1-it)
        x_approx,u = LTV(x0, A, B, f.T, x.T, mesh, trust_region=4, P=P[it], umax=umax )
        X_cvx.append(x_approx)
        # u = u.T
        # import pdb
        # pdb.set_trace()


    U.append(u)
    x = integrate(x0, u, t).T # Can increase fidelity here or after optimal solution found
    X.append(x)

    try:
        print "Performing final iteration - hard enforcement of constraints, and additional discretization points"
        f,g = dynamics(x)
        A,B = jac(x)
        x_approx,u = LTV(x0, A, B, f.T, x.T, mesh, trust_region=2,P=0,umax=3)

        u = u.T
        print u.shape
        x,u = integrate(x0, u, t,return_u=True)
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
        plt.plot(u)

    plt.figure(3)
    plt.plot(X_cvx[-1][0],X_cvx[-1][1],'o-',label='Convex discretization')
    plt.plot(X[-1][0],X[-1][1],label='Integration')
    plt.legend()

    plt.figure(4)
    plt.plot(U[-1])
    plt.title('optimal control')

    plt.figure(1)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test()
