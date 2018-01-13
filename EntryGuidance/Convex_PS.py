import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
import time
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from Mesh import Mesh

def LTV(x0, A, B, f_ref, x_ref, u_ref, mesh, trust_region=0.5, P=0, xf=0, umax=3):

    """ Solves a convex LTV subproblem

    A - state linearization around x_ref
    B - control linearization around x_ref
    f_ref - the original nonlinear dynamics evaluated along x_ref
    x_ref - the trajectory used to linearize the problem, typically the current iterate

    x0 - initial condition
    mesh - Mesh.Mesh instance

    """
    t0 = time.time()

    n = A[0].shape[0]
    m = 1
    N = mesh.n_points
    T = range(N)

    # x = cvx.Variable(N,n)
    x = np.array([cvx.Variable(n) for _ in range(N)]) # This has to be done to "chunk" it later
    stm = np.array([[cvx.Variable(n) for _ in range(n)] for i in range(N)])
    # stm = np.array([cvx.Variable(n,n) for _ in range(N)])
    u = cvx.Variable(N,m)
    v = cvx.Variable(N,m) # Virtual control


    X = mesh.chunk(x)
    A = mesh.chunk(A)
    B = mesh.chunk(B)
    F = mesh.chunk(f_ref)
    Xr = mesh.chunk(x_ref)
    U = mesh.chunk(u)
    Ur = mesh.chunk(u_ref)
    STM = mesh.chunk(stm)
    
    if P > 0:
        bc = [x[0] == x0, x[-1] <= xf+P*1, x[-1] >= xf-P*1]
    else:
        bc = [x[-1] == xf, x[0] == x0]

    stm_ic = [stm0 == I for stm0,I in zip(stm[0],np.eye(n))] 
    bc += stm_ic  

    constr = []
    for t,xr in zip(T,x_ref):
        constr += [cvx.norm((x[t]-xr)) < trust_region]

    # Control constraints
    Cu = cvx.abs(u) <= umax

    # Lagrange cost and ode constraints
    states = []
    for d,xi,f,a,b,xr,ur,ui,w,stmi in zip(mesh.diffs,X,F,A,B,Xr,Ur,U,mesh.weights,STM):
        L = cvx.abs(ui)**1                                         # Lagrange integrands for a single mesh
        cost = w*L                                                  # Clenshaw-Curtis quadrature
        dx = d.dot(xi)
        dstmi = d.dot(stmi).T
        ode =  [dxi == fi + ai*(xii-xri) + bi*(uii-uri)  for xii,fi,ai,bi,xri,uri,uii,dxi in zip(xi,f,a,b,xr,ur,ui,dx) ] # Differential equation constraints
        # stm_ode = [dstmii == ai*stmii for ai,stmii,dstmii in zip(a,stmi,dstmi)]
        ode_stm = []
        for ai,stmii,dstmii in zip(a,stmi,dstmi):
            for s,ds in zip(stmii,dstmii):

                ode_stm.append(ds == ai*s) 

        states.append(cvx.Problem(cvx.Minimize(cost), ode+ode_stm))

    # sums problem objectives and concatenates constraints.
    prob = sum(states)
    prob.constraints.append(Cu)
    prob.constraints += bc
    prob.constraints += constr

    t1 = time.time()
    prob.solve()
    t2 = time.time()
    print "status:        ", prob.status
    print "optimal value: ", np.around(prob.value,3)
    print "solution time: {}s".format(t2-t1)
    print "setup time: {}s\n".format(t1-t0)
    print stm[0][0].value.A
    print stm[0][1].value.A

    stm_sol = np.array([[s.value.A for s in S] for S in stm]).squeeze()
    import pdb 
    pdb.set_trace()
    
    try:
        x_sol = np.array([xi.value.A for xi in x]).squeeze()
        u_sol = u.value.A.squeeze()
        return x_sol.T, u_sol, stm_sol
    except:
        return x_ref.T,u_ref,np.zeros((N,n,n))


def test():
    """ Solves a control constrained VDP problem to compare with GPOPS """

    mu = 0.1
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
        u[-1]=u[-2]
        ut = interp1d(t,u,kind='cubic',assume_sorted=True,fill_value=u[-1],bounds_error=False)
        X = odeint(dyn,x0,t,args=(ut,))

        if return_u:
            return np.asarray(X),u
        else:
            return np.asarray(X)






    X = []
    X_cvx = []
    U = []
    STM = []

    tf = 5
    mesh = Mesh(tf=tf,orders=[3]*12)
    t = mesh.times
    x0 = [3,1]
    u = np.zeros_like(t)
    # x = np.vstack((np.linspace(x0[0],0,u.shape[0]),np.linspace(x0[1],0,u.shape[0])))


    iters = 2

    # P = np.logspace(-1, iters-1,iters)
    P = np.linspace(1,0.01,iters)
    for it in range(iters):
        U.append(u)

        # if it:
        x = integrate(x0, u, t).T
    # else:
        if not it:
            x_approx = x
        f,g = dynamics(x)
        F = f.T + g.T*u[:,None]
        A,B = jac(x)

        X.append(x)
        umax = 3
        x_approx,u, stm_approx = LTV(x0, A, B, F, x_approx.T, u, mesh, trust_region=8, P=0*P[it], umax=umax )
        X_cvx.append(x_approx)
        STM.append(stm_approx)


    for i in range(2):
        mesh.bisect()
        t_u = t
        t = mesh.times
        u = interp1d(t_u, u, kind='linear', axis=-1, copy=True, bounds_error=None, fill_value=np.nan, assume_sorted=True)(t)
        U.append(u)
        x = integrate(x0, u, t).T # Can increase fidelity here or after optimal solution found
        X.append(x)

        try:
            print "Performing final iteration - hard enforcement of constraints, and additional discretization points"
            f,g = dynamics(x)
            F = f.T + g.T*u[:,None]
            A,B = jac(x)
            x_approx,u, stm_approx = LTV(x0, A, B, F, x.T, u, mesh, trust_region=1,P=0,umax=umax)

            u = u.T
            x,u = integrate(x0, u, t,return_u=True)
            X.append(x.T)
            X_cvx.append(x_approx)
            X_cvx.append(x_approx) # just so theyre the same length
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
