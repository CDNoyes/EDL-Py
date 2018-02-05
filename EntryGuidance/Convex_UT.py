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

def LTV(sigma_points, A_sp, B_sp, f_sp, x_sp, u_ref, mesh, trust_region=0.5, xf=0, umax=3):

    """ Solves a convex LTV subproblem

    A - state linearization around x_ref
    B - control linearization around x_ref
    f_sp - the original nonlinear dynamics evaluated along each x_ref in x_sp
    x_sp - the trajectories used to linearize the problem, the current iterate

    x0 - initial condition
    mesh - Mesh.Mesh instance

    """
    t0 = time.time()
    prob = None

    N = mesh.n_points
    T = range(N)

    n = len(sigma_points[0]) #.shape[1]
    m = 1


    # Define the controls, which are used across all sigma points
    u = cvx.Variable(N,m) # Only works for m=1
    v = np.array([cvx.Variable(n) for _ in range(N)]) # Virtual controls, might need a different one for each sp
    U = mesh.chunk(u)
    Ur = mesh.chunk(u_ref)
    V = mesh.chunk(v)

    x_var = np.array([[cvx.Variable(n) for _ in range(N)] for __ in sigma_points])       # This has to be done to "chunk" it later

    # Control constraints
    Cu = cvx.abs(u) <= umax

    # Iteration over each of the sigma points
    for x0, x, A, B, f_ref, x_ref in zip(sigma_points, x_var, A_sp, B_sp, f_sp, x_sp):

        # K = cvx.Variable(rows=1,cols=2) # Linear feedback gain
        # K = np.array([1,1]).T

        X = mesh.chunk(x)
        A = mesh.chunk(A)
        B = mesh.chunk(B)
        F = mesh.chunk(f_ref)
        Xr = mesh.chunk(x_ref)

        bc = [x[0] == x0]   # Initial condition only

        Ctr = []
        for t,xr in zip(T,x_ref):
            Ctr += [cvx.norm((x[t]-xr)) <= trust_region]

        # Lagrange cost and ode constraints
        states = []
        for d,xi,f,a,b,xr,ur,ui,w,vi in zip(mesh.diffs,X,F,A,B,Xr,Ur,U,mesh.weights,V): # Iteration over the segments of the mesh
            L = 0*cvx.abs(ui)**1                                          # Lagrange integrands for a single mesh
            cost = w*L                                                  # Clenshaw-Curtis quadrature

            # Estimated derivatives:
            dx = d.dot(xi)

            # Differential equation constraints
            ode =  [dxi == fi + ai*(xii-xri) + bi*(uii-uri) + vii  for xii,fi,ai,bi,xri,uri,uii,dxi,vii in zip(xi,f,a,b,xr,ur,ui,dx,vi) ] # Iteration over individual collocation points

            states.append(cvx.Problem(cvx.Minimize(cost), ode))


        # sums problem objectives and concatenates constraints.
        if prob is None:
            prob = sum(states)
        else:
            prob += sum(states)
        prob.constraints += Ctr # Apply the trust region constraints
        prob.constraints += bc  # Apply the initial boundary conditions

    Endpoint = []
    # Phi = cvx.Problem(cvx.Minimize(1*cvx.norm(x_var[0,-1],2)))                        # Mayer Cost
    Phi = cvx.Problem(cvx.Minimize(1*cvx.norm(u,'inf')))                        # Mayer Cost
    Penalty = cvx.Problem(cvx.Minimize(1e6*cvx.norm(cvx.vstack(*v),'inf') ))    # Penalty for virtual control
    prob.constraints.append(Cu)     # Apply the control constraints
    # import pdb
    # pdb.set_trace()
    prob.constraints.append(x_var[0,-1]==xf) # End point constraint, on mean or nominal trajectory
    prob += Phi + Penalty

    t1 = time.time()
    prob.solve() #solver='ECOS'
    t2 = time.time()
    print "status:        ", prob.status
    print "optimal value: ", np.around(prob.value,3)
    print "solution time:  {} s".format(np.around(t2-t1,3))
    print "setup time:     {} s".format(np.around(t1-t0,3))

    try:
        x_sol = [np.array([xi.value.A for xi in x]).squeeze() for x in x_var]
        u_sol = u.value.A.squeeze()
        v_sol = np.array([xi.value.A for xi in v]).squeeze()
        print "penalty value:  {}\n".format(np.linalg.norm(v_sol.flatten(),np.inf))

        stm_sol = np.zeros((N,n,n))
        return x_sol, u_sol, stm_sol, prob.value
    except:
        return x_ref,u_ref,np.zeros((N,n,n)),None





def test():
    """ Solves a control constrained VDP problem to compare with GPOPS """

    mu = 1.
    def dynamics(x):
        # returns f,g evaluated at x (vectorized)
        return np.array([x[1],-x[0] + mu*(1-x[0]**2)*x[1]]),np.stack((np.zeros_like(x[0]),np.ones_like(x[0]))).squeeze()

    def dyn(x,t,u): # integrable function
        f,g = dynamics(x)
        return f + g*u(t)

    def jac(x):
        x1,x2=x
        shape = [x.shape[0]]
        shape.extend(x.shape)
        A = np.zeros(shape)

        B = np.stack((np.zeros_like(x1),np.ones_like(x1))).squeeze()

        A[0,1,:] = np.ones_like(x[0])
        A[1,0,:] = -np.ones_like(x[0]) -2*mu*x1*x2
        A[1,1,:] = mu*(1-x[0]**2)

        return np.moveaxis(A, -2, 0),np.moveaxis(B, -2, 0)

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
            if len(x0.shape)>1:
                X = RK4(dyn,x0,t,args=(ut,))
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
    tf = 5
    mesh = Mesh(tf=tf,orders=[6]*5)
    t = mesh.times
    x0 = np.array([3,-3])
    P0 = np.eye(2)*0.1              # Initial covariance
    sp,wm,wc = Unscented.Transform(x0,P0)
    # Cerify mean and cov of initial points
    # print sp.T.dot(wm)
    e = (sp-np.array(x0))
    C = sum([wci*np.outer(ei,ei) for ei,wci in zip(e,wc)])
    # print sp
    # print wm

    # Initial "guess" used for linearization
    u = np.zeros_like(t)
    x = integrate(sp.T, u, t)
    x = np.transpose(x,(1,0,2))
    f,g = dynamics(x)
    F = f + g*u[:,None]
    A,B = jac(x)

    x = np.transpose(x,(2,1,0))
    x_approx = x

    A = np.moveaxis(A,-1,0)
    B = np.moveaxis(B,-1,0)
    F = np.transpose(F,(2,1,0))
    print sp.shape
    print x.shape
    print A.shape
    print B.shape
    print F.shape

    # sys.exit()

    iters = 20                       # Maximum number of iterations
    P = np.linspace(1,0.1,iters)
    trust_region = 4
    # Main Loop
    for it in range(iters):
        print "Iteration {}".format(it)

        x_approx, u, stm_approx, J_approx = LTV([sp[0]], [A[0]], [B[0]], [F[0]], [x_approx[0]], u, mesh, trust_region=trust_region, umax=umax)
        # x_approx, u, stm_approx, J_approx = LTV(sp, A, B, F, x_approx, u, mesh, trust_region=trust_region, umax=umax)

        # sys.exit()
        if J_approx is None: # Failed iteration
            trust_region *= 0.8
            print "New trust region = {}".format(trust_region)

            mesh.bisect()
            t_u = t
            t = mesh.times
            u = interp1d(t_u, u, kind='linear', axis=-1, copy=True, bounds_error=None, fill_value=np.nan, assume_sorted=True)(t)
            x = integrate(sp.T, u, t)
            x_approx = x

        else:
            x = integrate(sp.T, u, t)
            x = np.transpose(x,(1,0,2))
            x_approx = x #todo - remove this when done debugging
            f,g = dynamics(x_approx)
            F = f + g*u[:,None]
            A,B = jac(x_approx)
            x_approx = np.transpose(x_approx,(2,1,0))
            x = np.transpose(x,(2,1,0))

            A = np.moveaxis(A,-1,0)
            B = np.moveaxis(B,-1,0)
            F = np.transpose(F,(2,1,0))

            X_cvx.append(x_approx)
            J_cvx.append(J_approx)
            X.append(x)
            U.append(u)
            T.append(t)
            STM.append(stm_approx)

            if len(J_cvx)>1:
                if J_cvx[-1] > 1e-3:
                    rel_diff = np.abs(J_cvx[-1]-J_cvx[-2])/(J_cvx[-1])
                else: #near zero cost so we use the absolute difference instead
                    rel_diff = np.abs(J_cvx[-1]-J_cvx[-2])

                if rel_diff < 0.3: # check state convergence instead?
                    # In contrast to NLP, we only refine after the solution converges on the current mesh
                    if it < iters-1:
                        current_size = mesh.times.size
                        refined = mesh.refine(x_approx[0], F[0], tol=1e-6, rho=1)
                        if mesh.times.size > 1000 or not refined:
                            print 'Terminating'
                            break
                        t_u = t
                        print "Mesh refinement resulted in {} segments with {} collocation points\n".format(len(mesh.orders),t.size)
                        t = mesh.times
                        u = interp1d(t_u, u, kind='linear', axis=-1, copy=True, bounds_error=None, fill_value=np.nan, assume_sorted=True)(t)
                        x_approx = interp1d(t_u, x_approx, kind='linear', axis=1, copy=True, bounds_error=None, fill_value=np.nan, assume_sorted=True)(t)
                        print x_approx.shape
                        x_approx = np.transpose(x_approx,(2,1,0))

                        f,g = dynamics(x_approx)
                        F = f + g*u[:,None]
                        A,B = jac(x_approx)
                        x_approx = np.transpose(x_approx,(2,1,0))

                        A = np.moveaxis(A,-1,0)
                        B = np.moveaxis(B,-1,0)
                        F = np.transpose(F,(2,1,0))


    x = integrate(x0, u, t)
    plt.figure(1)
    plt.plot(x.T[0],x.T[1])
    plt.show()
    X.append(x.T)
    T.append(t)
    U.append(u)
    STM.append(stm_approx)
    X_sp = []
    x_bar = np.zeros_like(x)
    for sigma_pt,wmi in zip(sp,wm):
        x_sp = integrate(sigma_pt, u, t)
        x_bar += x_sp*wmi
        X_sp.append(x_sp)




    # print len(X)
    # print len(U)
    # print len(T)
    # print len(X_cvx)

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
    ti = mesh._times
    xcvx = interp1d(T[-1], X_cvx[-1].T, kind='linear', axis=0, assume_sorted=True)(ti).T
    plt.plot(xcvx[0],xcvx[1],'o',label='Discretization')
    # plt.plot(X_cvx[-1][0],X_cvx[-1][1],'o-',label='Discretization')
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

    plt.figure(8)
    for pt in X_sp:
        plt.plot(pt.T[0],pt.T[1])
    plt.plot(x_bar.T[0],x_bar.T[1],'k--')
    plt.title('Optimal Sigma Point Trajectories')

    for fig in [1,2,5,3]:
        plt.figure(fig)
        plt.legend()
    plt.show()

if __name__ == "__main__":
    test()
