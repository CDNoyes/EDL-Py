import sys
sys.path.append('./')
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
import time
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from Utils.RK4 import RK4, RK4_STM
from Mesh import Mesh
import Unscented

def LTV(x0, A, B, f_ref, x_ref, u_ref, mesh, trust_region=0.5, P=0, xf=0, umax=3, use_stm=False):

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

    x = np.array([cvx.Variable(n) for _ in range(N)])       # This has to be done to "chunk" it later
    if use_stm:
        stm = np.array([cvx.Variable(n,n) for _ in range(N)])
    else:
        stm = np.zeros((N,n,n))
    u = cvx.Variable(N,m) # Only works for m=1
    v = np.array([cvx.Variable(n) for _ in range(N)]) # Virtual controls
    # K = cvx.Variable(rows=1,cols=2) # Linear feedback gain

    # Alternatively, we could create meshes of variables directly
    X = mesh.chunk(x)
    A = mesh.chunk(A)
    B = mesh.chunk(B)
    F = mesh.chunk(f_ref)
    Xr = mesh.chunk(x_ref)
    U = mesh.chunk(u)
    Ur = mesh.chunk(u_ref)
    STM = mesh.chunk(stm)
    V = mesh.chunk(v)

    if P > 0: # Relaxed final conditions - we can make P a variable (or vector) and penalize it heavily in the cost function like v
        bc = [x[0] == x0, x[-1] <= xf+P*1, x[-1] >= xf-P*1]
    else:
        bc = [x[-1] == xf, x[0] == x0]
        # bc = [x[0] == x0]   # Initial condition only
        # bc = [x[-1] == xf]  # Final condition only

    if use_stm:
        stm_ic = [stm[0] == np.eye(n)]
        bc += stm_ic

    constr = []
    for t,xr in zip(T,x_ref):
        constr += [cvx.norm((x[t]-xr)) <= trust_region]

    # Control constraints
    Cu = cvx.abs(u) <= umax

    # Lagrange cost and ode constraints
    states = []
    for d,xi,f,a,b,xr,ur,ui,w,stmi,vi in zip(mesh.diffs,X,F,A,B,Xr,Ur,U,mesh.weights,STM,V):
        L = 1*cvx.abs(ui)**1                                          # Lagrange integrands for a single mesh
        cost = w*L                                                  # Clenshaw-Curtis quadrature

        # Estimated derivatives:
        dx = d.dot(xi)
        if use_stm:
            dstmi = d.dot(stmi)

        # Differential equation constraints
        ode =  [dxi == fi + ai*(xii-xri) + bi*(uii-uri) + vii  for xii,fi,ai,bi,xri,uri,uii,dxi,vii in zip(xi,f,a,b,xr,ur,ui,dx,vi) ]
        if use_stm:
            ode_stm = [dstmii == ai*stmii for ai,bi,stmii,dstmii in zip(a,b,stmi,dstmi)]
        else:
            ode_stm = []

        states.append(cvx.Problem(cvx.Minimize(cost), ode+ode_stm))

        # Mayer Cost, including penalty for virtual control
    Phi = cvx.Problem(cvx.Minimize(0*cvx.norm(u,'inf')))
    Penalty = cvx.Problem(cvx.Minimize(1e5*cvx.norm(cvx.vstack(*v),'inf') ))

    # sums problem objectives and concatenates constraints.
    prob = sum(states) + Phi + Penalty
    prob.constraints.append(Cu)
    prob.constraints += bc
    prob.constraints += constr

    t1 = time.time()
    prob.solve()
    t2 = time.time()
    print "status:        ", prob.status
    print "optimal value: ", np.around(prob.value,3)
    print "solution time:  {} s".format(np.around(t2-t1,3))
    print "setup time:     {} s".format(np.around(t1-t0,3))

    try:
        x_sol = np.array([xi.value.A for xi in x]).squeeze()
        u_sol = u.value.A.squeeze()
        v_sol = np.array([xi.value.A for xi in v]).squeeze()
        print "penalty value:  {}\n".format(np.linalg.norm(v_sol.flatten(),np.inf))

        if use_stm:
            stm_sol = np.array([s.value.A for s in stm]).squeeze()
        else:
            stm_sol = np.zeros((N,n,n))

        return x_sol.T, u_sol, stm_sol, prob.value
    except:
        return x_ref.T,u_ref,np.zeros((N,n,n)),None


def test():
    """ Solves a control constrained VDP problem to compare with GPOPS """

    mu = 1.
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

        B = np.vstack((np.zeros_like(x1),np.ones_like(x1))).squeeze()

        A[0,1,:] = np.ones_like(x[0])
        A[1,0,:] = -np.ones_like(x[0]) -2*mu*x1*x2
        A[1,1,:] = mu*(1-x[0]**2)

        return np.moveaxis(A, -1, 0),np.moveaxis(B, -1, 0)

    def hess(x):
        x1,x2=x
        try:
            N = x1.shape[0]
        except:
            N = 1
        H = np.zeros((N,2,2,2)) # Will be (N,2,3,3) if mu is included
        H[:,1,0,0] = -2*mu*x2
        H[:,1,0,1] = -2*mu*x1
        H[:,1,1,0] = -2*mu*x1
        return H.squeeze()

    def stm_dyn(stm,t,A):
        return np.asarray(A(t)).dot(stm)

    def integrate_stm(A,t):
        stm0 = np.eye(A[0].shape[0])

        Ai = interp1d(t,A, kind='cubic', axis=0, copy=True, bounds_error=None, fill_value=np.nan, assume_sorted=True)
        stm = RK4(stm_dyn, stm0, t, args=(Ai,))
        return stm

    def integrate(x0, u, t, return_stm=False):
        u[-1]=u[-2]
        ut = interp1d(t,u,kind='cubic',assume_sorted=True,fill_value=u[-1],bounds_error=False)
        if return_stm:
            X,STM = RK4_STM(dyn,x0,t,args=(ut,))
        else:
            X = odeint(dyn,x0,t,args=(ut,))

        if return_stm:
            return np.asarray(X),np.asarray(STM)
        else:
            return np.asarray(X)






    X = []
    X_cvx = []
    J_cvx = []
    U = []
    STM = []
    T = []

    umax = 3
    tf = 4
    mesh = Mesh(tf=tf,orders=[6]*5)
    t = mesh.times
    x0 = [3,-3]
    P0 = np.eye(2)*0.1              # Initial covariance
    sp,wm,wc = Unscented.Transform(x0,P0)
    print sp
    # Initial "guess" used for linearization
    u = np.zeros_like(t)
    # x = np.vstack((np.linspace(x0[0],0,u.shape[0]),np.linspace(x0[1],0,u.shape[0])))
    x = integrate(x0, u, t).T
    f,g = dynamics(x)
    F = f.T + g.T*u[:,None]
    A,B = jac(x)

    x_approx = x


    iters = 20                       # Maximum number of iterations
    bisectMax = 3                       # Maximum number of bisections of the mesh
    n_bisect = 0
    P = np.linspace(1,0.1,iters)
    trust_region = 5
    # Main Loop
    for it in range(iters):
        print "Iteration {}".format(it)

        x_approx, u, stm_approx, J_approx = LTV(x0, A, B, F, x_approx.T, u, mesh, trust_region=trust_region, P=0*P[it], umax=umax, use_stm=False)

        if J_approx is None: # Failed iteration
            trust_region *= 0.8
            print "New trust region = {}".format(trust_region)

            mesh.bisect()
            t_u = t
            t = mesh.times
            u = interp1d(t_u, u, kind='linear', axis=-1, copy=True, bounds_error=None, fill_value=np.nan, assume_sorted=True)(t)
            x = integrate(x0, u, t).T
            x_approx = x

        else:
            x = integrate(x0, u, t).T

            f,g = dynamics(x_approx)
            F = f.T + g.T*u[:,None]
            A,B = jac(x_approx)

            X_cvx.append(x_approx)
            J_cvx.append(J_approx)
            X.append(x)
            U.append(u)
            T.append(t)
            STM.append(stm_approx)
            # Check for convergence
            if len(J_cvx)>1 and np.abs(J_cvx[-1]-J_cvx[-2])<0.1:
                if n_bisect >= bisectMax:
                    break
                if it < iters-1:
                    mesh.bisect()
                    n_bisect += 1
                    t_u = t
                    t = mesh.times
                    print "Bisecting mesh -> {} collocation points".format(t.size)
                    # plt.figure(666)
                    # plt.plot(t_u,u,'o-',label='Original')
                    u = interp1d(t_u, u, kind='linear', axis=-1, copy=True, bounds_error=None, fill_value=np.nan, assume_sorted=True)(t)
                    # plt.plot(t,u,'x-',label='Resampled')
                    # plt.legend()
                    # plt.show()
                    x = integrate(x0, u, t).T
                    x_approx = x
                    f,g = dynamics(x_approx)
                    F = f.T + g.T*u[:,None]
                    A,B = jac(x_approx)


    x = integrate(x0, u, t)
    X.append(x.T)
    T.append(t)
    U.append(u)
    STM.append(stm_approx)

    # print len(X)
    # print len(U)
    # print len(T)
    # print len(X_cvx)
    A,B = jac(x.T)
    stm_true = integrate_stm(A,t)

    for i,xux in enumerate(zip(T,X,U,X_cvx)):

        t,x,u,xc = xux

        plt.figure(1)
        plt.plot(x[0],x[1],label=str(i))
        plt.title('State Iterations (Integration)')
        plt.figure(5)
        plt.plot(xc[0],xc[1],label=str(i))
        plt.title('State Iterations (Discretization)')
        plt.figure(2)
        plt.plot(t,u,label=str(i))
        plt.title('Control Iterations')

    plt.figure(3)
    plt.plot(X_cvx[-1][0],X_cvx[-1][1],'o-',label='Discretization')
    plt.plot(X[-1][0],X[-1][1],label='Integration')
    plt.title('Optimal Trajectory')
    plt.legend()

    plt.figure(4)
    plt.plot(T[-1],U[-1])
    plt.title('Optimal control')

    # plt.figure(6)
    # plt.plot(t,np.reshape(stm_true,(t.shape[0],-1)))
    # plt.plot(t,np.reshape(STM[-1],(t.shape[0],-1)),'--')

    plt.figure(7)
    plt.semilogy(J_cvx,'o-')
    plt.ylabel('Objective Function')
    plt.xlabel('Iteration')

    for fig in [1,2,5,3]:
        plt.figure(fig)
        plt.legend()
    plt.show()

if __name__ == "__main__":
    test()
