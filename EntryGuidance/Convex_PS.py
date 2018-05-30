import sys
sys.path.append('./')
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
import time
from scipy.integrate import odeint
from scipy.interpolate import interp1d

from Utils.RK4 import RK4
from EntryGuidance.Mesh import Mesh

def LTV(x0, A, B, f_ref, x_ref, u_ref, mesh, trust_region=0.5, P=0, xf=0, umax=3):

    """ Solves a convex LTV subproblem

    A - state linearization around x_ref
    B - control linearization around x_ref
    f_ref - the original nonlinear dynamics evaluated along x_ref
    x_ref - the trajectory used to linearize the problem, the current iterate

    x0 - initial condition
    mesh - Mesh.Mesh instance

    """
    t0 = time.time()

    n = A[0].shape[0]
    m = 1
    N = mesh.n_points
    T = range(N)

    x = np.array([cvx.Variable(n) for _ in range(N)])       # This has to be done to "chunk" it later
    # u = cvx.Variable(N,m) # Only works for m=1
    u = np.array([cvx.Variable(m) for _ in range(N)])
    v = np.array([cvx.Variable(n) for _ in range(N)]) # Virtual controls
    # K = cvx.Variable(rows=1,cols=2) # Linear feedback gain
    K = np.array([1,1]).T
    # Alternatively, we could create meshes of variables directly
    X = mesh.chunk(x)
    A = mesh.chunk(A)
    B = mesh.chunk(B)
    F = mesh.chunk(f_ref)
    Xr = mesh.chunk(x_ref)
    U = mesh.chunk(u)
    Ur = mesh.chunk(u_ref)
    V = mesh.chunk(v)

    if P > 0: # Relaxed final conditions - we can make P a variable (or vector) and penalize it heavily in the cost function like v
        bc = [x[0] == x0, x[-1] <= xf+P*1, x[-1] >= xf-P*1]
    else:
        bc = [x[-1] == xf, x[0] == x0]
        # bc = [x[0] == x0]   # Initial condition only
        # bc = [x[-1] == xf]  # Final condition only

    constr = []
    for t,xr in zip(T,x_ref):
        # constr += [cvx.norm((x[t]-xr)) <= trust_region]
        # constr += [cvx.abs(x[t]-xr) <= trust_region]
        constr += [(x[t]-xr)**2 <= trust_region**2]
        constr += [cvx.abs(u[t]) <= umax]


    # Control constraints

    # Lagrange cost and ode constraints
    states = []
    for d,xi,f,a,b,xr,ur,ui,w,vi in zip(mesh.diffs,X,F,A,B,Xr,Ur,U,mesh.weights,V): # Iteration over the segments of the mesh
        L = np.array([cvx.abs(uii)**2 for uii in ui])               # Lagrange integrands for a single mesh
        cost = np.dot(w,L)                                                  # Clenshaw-Curtis quadrature

        # Estimated derivatives:
        dx = d.dot(xi)

        # Differential equation constraints
        ode =  [dxi == fi + ai*(xii-xri) + bi*(uii-uri) + vii  for xii,fi,ai,bi,xri,uri,uii,dxi,vii in zip(xi,f,a,b,xr,ur,ui,dx,vi) ] # Iteration over individual collocation points

        states.append(cvx.Problem(cvx.Minimize(cost), ode))

    # Mayer Cost, including penalty for virtual control
    # Phi = cvx.Problem(cvx.Minimize(0*cvx.norm(u,'inf')))
    Phi = 0
    Penalty = cvx.Problem(cvx.Minimize(1e5*cvx.norm(cvx.vstack(*v),'inf') ))

    # sums problem objectives and concatenates constraints.
    prob = sum(states) + Phi + Penalty
    prob.constraints += bc
    prob.constraints += constr

    t1 = time.time()
    prob.solve(solver='ECOS')
    t2 = time.time()

    print("status:        {}".format(prob.status))
    print("optimal value: {}".format(np.around(prob.value,3)))
    print("solution time:  {} s".format(np.around(t2-t1,3)))
    print("setup time:     {} s".format(np.around(t1-t0,3)))

    try:
        x_sol = np.array([xi.value.A for xi in x]).squeeze()
        u_sol = np.array([ui.value for ui in u]).squeeze()
        v_sol = np.array([xi.value.A for xi in v]).squeeze()
        print("penalty value:  {}\n".format(np.linalg.norm(v_sol.flatten(),np.inf)))

        return x_sol.T, u_sol, prob.value
    except Exception as e:
        print(e.message)
        return x_ref.T,u_ref,None





class OCP:
    """ Defines an abstract class for solving nonlinear optimal control
        problems via successive convexification
    """


    def __init__(self):
        pass

    def dyn(self):
        raise NotImplementedError
    def jac(self):
        raise NotImplementedError

    def integrate(self, x0, u, t):
        u[-1]=u[-2]
        ut = interp1d(t,u,kind='cubic',assume_sorted=True,fill_value=u[-1],bounds_error=False)
        X = odeint(self.dyn,x0,t,args=(ut,))
        return np.asarray(X)

    def solve(self):
        X = []
        X_cvx = []
        J_cvx = []
        U = []
        T = []

        umax = 3
        tf = 5
        mesh = Mesh(tf=tf, orders=[4]*3)
        t = mesh.times
        x0 = [2,-2]

        # Initial "guess" used for linearization
        u = np.zeros_like(t)
        x = self.integrate(x0, u, t).T
        ufun = lambda t: 0
        F = self.dyn(x,t,ufun).T
        A,B = self.jac(x)

        x_approx = x


        iters = 30                       # Maximum number of iterations
        P = np.linspace(1, 0.1, iters)
        # trust_region = 4
        trust_region = np.array([4,4])
        # Main Loop
        for it in range(iters):
            print("Iteration {}".format(it))

            x_approx, u, J_approx = LTV(x0, A, B, F, x_approx.T, u, mesh, trust_region=trust_region, P=0*P[it], umax=umax)

            if J_approx is None: # Failed iteration
                trust_region *= 0.8
                print("New trust region = {}".format(trust_region))

                mesh.bisect()
                t_u = t
                t = mesh.times
                u = interp1d(t_u, u, kind='linear', axis=-1, copy=True, bounds_error=None, fill_value=np.nan, assume_sorted=True)(t)
                x = self.integrate(x0, u, t).T
                x_approx = x

            else:
                x = self.integrate(x0, u, t).T

                ufun = interp1d(t, u, kind='linear', axis=-1, copy=True, bounds_error=None, fill_value=np.nan, assume_sorted=True)
                F = self.dyn(x_approx, t, ufun).T
                A,B = self.jac(x_approx)

                X_cvx.append(x_approx)
                J_cvx.append(J_approx)
                X.append(x)
                U.append(u)
                T.append(t)

                rel_diff = None
                if len(J_cvx)>1:
                    if J_cvx[-1] > 1e-3:
                        rel_diff = np.abs(J_cvx[-1]-J_cvx[-2])/(J_cvx[-1])
                    else: # near zero cost so we use the absolute difference instead
                        rel_diff = np.abs(J_cvx[-1]-J_cvx[-2])

                if rel_diff is None or rel_diff < 0.1: # check state convergence instead?
                        # In contrast to NLP, we only refine after the solution converges on the current mesh
                        if it < iters-1:
                            current_size = mesh.times.size
                            if it%2 and False:
                                _ = mesh.refine(u, np.zeros_like(u), tol=1e-2, rho=0) # Control based refinement
                            else:
                                refined = mesh.refine(x_approx.T, F, tol=1e-5, rho=3) # Dynamics based refinement for convergence check
                            if mesh.times.size > 1000:
                                print("Terminating because maximum number of collocation points has been reached.")
                                break
                            if not refined:
                                print('Terminating with optimal solution.')
                                break
                            t_u = t
                            print("Mesh refinement resulted in {} segments with {} collocation points\n".format(len(mesh.orders),t.size))
                            t = mesh.times
                            ufun = interp1d(t_u, u, kind='linear', axis=-1, copy=True, bounds_error=None, fill_value=np.nan, assume_sorted=True)
                            u = ufun(t)
                            x_approx = interp1d(t_u, x_approx, kind='linear', axis=-1, copy=True, bounds_error=None, fill_value=np.nan, assume_sorted=True)(t)
                            F = self.dyn(x_approx,t,ufun).T
                            A,B = self.jac(x_approx)



        x = self.integrate(x0, u, t)
        X.append(x.T)
        T.append(t)
        U.append(u)

        A,B = self.jac(x.T)

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


        plt.figure(7)
        plt.semilogy(J_cvx,'o-')
        plt.ylabel('Objective Function')
        plt.xlabel('Iteration')


        mesh.plot(show=False)

        for fig in [1,2,5,3]:
            plt.figure(fig)
            plt.legend()
        plt.show()

class TestClass(OCP):
    """ A very basic vanderpol oscillator for testing OCP solver """

    def __init__(self, mu=0):
        self.mu = mu

    def dynamics(self, x):
        # returns f,g evaluated at x (vectorized)
        return np.array([x[1],-x[0] + self.mu*(1-x[0]**2)*x[1]]),np.vstack((np.zeros_like(x[0]),np.ones_like(x[0]))).squeeze()

    def dyn(self, x, t, u): # integrable function
        f,g = self.dynamics(x)
        return f + g*u(t)

    def jac(self, x):
        x1,x2=x
        shape = [x.shape[0]]
        shape.extend(x.shape)
        A = np.zeros(shape)

        B = np.vstack((np.zeros_like(x1),np.ones_like(x1))).squeeze()

        A[0,1,:] = np.ones_like(x[0])
        A[1,0,:] = -np.ones_like(x[0]) -2*self.mu*x1*x2
        A[1,1,:] = self.mu*(1-x[0]**2)

        return np.moveaxis(A, -1, 0),np.moveaxis(B, -1, 0)

if __name__ == "__main__":
    vdp = TestClass(mu=0.1)
    vdp.solve()
