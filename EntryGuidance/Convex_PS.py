import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
from scipy.integrate import odeint
from scipy.interpolate import interp1d

sys.path.append('./')
from Utils.RK4 import RK4
from EntryGuidance.Mesh import Mesh
from Utils.DA import compute_jacobian

class OCP:
    """ Defines an abstract class for solving nonlinear optimal control
        problems via successive convexification
    """

    def __init__(self):
        pass

    def dynamics(self):
        raise NotImplementedError

    def jac(self, x, u):  # Could be optional
        raise NotImplementedError

    def lagrange(self, t, x, u, *args):
        raise NotImplementedError

    def mayer(self, xf, *args):
        raise NotImplementedError

    def constraints(self):
        raise NotImplementedError

    def integrate(self, x0, u, t):
        # u[-1] = u[-2]
        ut = interp1d(t, u,
                      kind='cubic',
                      assume_sorted=True,
                      fill_value=u[-1],
                      bounds_error=False)
        X = odeint(self.dynamics, x0, t, args=(ut,))
        return np.asarray(X)

    def solve(self, guess, scaling=None, max_size=500, max_iter=20):
        X = []
        X_cvx = []
        J_cvx = []
        U = []
        T = []

        mesh = Mesh(t0=guess['time'][0], tf=guess['time'][-1], orders=guess['mesh'])
        self.mesh = mesh 
        t = mesh.times

        # Initial "guess" used for linearization - this should be an input
        u = guess['control']
        x = guess['state']
        ti = guess['time']
                
        # Interpolate the guess onto the initial mesh 
        u = interp1d(ti, u, axis=0)(t)
        x = interp1d(ti, x, axis=0)(t).T
        F = self.dynamics(x, t, interp1d(t, u, axis=0)).T
        # print("Dimensions going into jacobian = {}, {}".format(x.shape, u.T.shape))
        A, B = self.jac(x, u.T)

        x_approx = x

        iters = int(max_iter)                 # Maximum number of iterations
        rel_diff = None

        # Main Loop
        for it in range(iters):
            print("Iteration {}".format(it))
            try:
                x_approx, u, J_approx = self.LTV(A, B, F, x_approx.T, u, mesh)
            except cvx.SolverError as msg:
                print(msg)
                print("Failed iteration, aborting sequence.")
                print(len(T))
                print(len(X_cvx))
                break

            if J_approx is None:  # Failed iteration
                trust_region *= 0.8
                print("New trust region = {}".format(trust_region))

                mesh.bisect()
                t_u = t
                t = mesh.times
                u = interp1d(t_u, u, kind='linear', axis=0, copy=True, bounds_error=None, fill_value=np.nan, assume_sorted=True)(t)
                x = self.integrate(self.x0, u, t).T
                x_approx = x

            else:
                # x = self.integrate(self.x0, u, t).T

                ufun = interp1d(t, u, kind='linear', axis=0, copy=True, bounds_error=None, fill_value=np.nan, assume_sorted=True)

                F = self.dynamics(x_approx, t, ufun).T
                A, B = self.jac(x_approx, u.T)

                X_cvx.append(x_approx)
                J_cvx.append(J_approx)
                # X.append(x)
                U.append(u)
                T.append(t)

                if len(J_cvx) > 1:
                    if np.abs(J_cvx[-1]) > 1e-3:
                        rel_diff = np.abs(J_cvx[-1]-J_cvx[-2])/np.abs(J_cvx[-1])
                        print("Relative change in cost function = {:.2f}%".format(rel_diff*100))
                    else: # near zero cost so we use the absolute difference instead
                        rel_diff = np.abs(J_cvx[-1]-J_cvx[-2])
                if rel_diff is None or rel_diff < 0.1:  # check state convergence instead?
                        # In contrast to NLP, we only refine after the solution converges on the current mesh
                        if it < iters-1:
                            current_size = mesh.times.size
                            if it%2 and False:
                                _ = mesh.refine(u, np.zeros_like(u), tol=1e-2, rho=0) # Control based refinement
                            else:
                                refined = mesh.refine(x_approx.T, F, tol=1e-7, rho=1.5, scaling=scaling, verbose=False) # Dynamics based refinement for convergence check
                            if mesh.times.size > max_size:
                                print("Terminating because maximum number of collocation points has been reached.")
                                break
                            if not refined:
                                print('Terminating with optimal solution.')
                                break
                            t_u = t
                            print("Mesh refinement resulted in {} segments with {} collocation points\n".format(len(mesh.orders),t.size))
                            t = mesh.times
                            ufun = interp1d(t_u, u, kind='linear', axis=0, copy=True, bounds_error=None, fill_value=np.nan, assume_sorted=True)
                            u = ufun(t)
                            x_approx = interp1d(t_u, x_approx, kind='linear', axis=-1, copy=True, bounds_error=None, fill_value=np.nan, assume_sorted=True)(t)
                            F = self.dynamics(x_approx, t, ufun).T
                            A, B = self.jac(x_approx, u.T)

        # x = self.integrate(self.x0, u, t)
        # X.append(x.T)
        # T.append(t)
        # U.append(u)

        self.plot(T, U, X_cvx, J_cvx, mesh._times)

    def LTV(self, A, B, f_ref, x_ref, u_ref, mesh):
        """ Solves a convex LTV subproblem

            A - state linearization around x_ref
            B - control linearization around x_ref
            f_ref - the original nonlinear dynamics evaluated along x_ref
            x_ref - the trajectory used to linearize the problem, the current iterate
            u_ref - the control trajectory used to linearize the problem

            mesh - Mesh.Mesh instance

        """
        t0 = time.time()

        n = A[0].shape[0]
        try:
            m = B[0].shape[1]
        except IndexError:
            m = 1
        N = mesh.n_points
        T = range(N)

        x = np.array([cvx.Variable(n) for _ in range(N)])  # This has to be done to "chunk" it later
        u = np.array([cvx.Variable(m) for _ in range(N)])
        v = np.array([cvx.Variable(n) for _ in range(N)])  # Virtual controls

        # Alternatively, we could create meshes of variables directly
        X = mesh.chunk(x)
        A = mesh.chunk(A)
        B = mesh.chunk(B)
        F = mesh.chunk(f_ref)
        Xr = mesh.chunk(x_ref)
        U = mesh.chunk(u)
        Ur = mesh.chunk(u_ref)
        V = mesh.chunk(v)

        # Compute the Lagrange cost and ode constraints
        states = []
        # Iteration over the segments of the mesh
        for d,xi,f,a,b,xr,ur,ui,w,vi in zip(mesh.diffs, X,F,A,B,Xr,Ur,U, mesh.weights, V):
            L = self.lagrange(0, xi, ui)
            cost = np.dot(w, L)                # Clenshaw-Curtis quadrature

            # Estimated derivatives:
            dx = d.dot(xi)

            # Differential equation constraints
            ode = [dxi == fi + ai*(xii-xri) + bi*(uii-uri) + vii for xii,fi,ai,bi,xri,uri,uii,dxi,vii in zip(xi,f,a,b,xr,ur,ui,dx,vi) ] # Iteration over individual collocation points

            states.append(cvx.Problem(cvx.Minimize(cost), ode))

        # Mayer Cost, including penalty for virtual control
        Phi = self.mayer(x[-1])
        Penalty = cvx.Problem(cvx.Minimize(1e3*cvx.norm(cvx.vstack(*v), 'inf')))

        # sums problem objectives and concatenates constraints.
        prob = sum(states) + Phi + Penalty
        prob.constraints += self.constraints(T, x, u, x_ref, u_ref)

        t1 = time.time()
        prob.solve(solver='ECOS')
        t2 = time.time()

        print("status:        {}".format(prob.status))
        print("optimal value: {}".format(np.around(prob.value, 3)))
        print("solution time:  {} s".format(np.around(t2-t1, 3)))
        print("setup time:     {} s".format(np.around(t1-t0, 3)))

        try:
            x_sol = np.array([xi.value.A for xi in x]).squeeze()
            u_sol = np.array([ui.value for ui in u]).squeeze()
            v_sol = np.array([xi.value.A for xi in v]).squeeze()
            print("penalty value:  {}\n".format(np.linalg.norm(v_sol.flatten(), np.inf)))

            return x_sol.T, u_sol, prob.value
        except Exception as e:
            print(e)
            return x_ref.T, u_ref, None


class TestClass(OCP):
    """ A very basic vanderpol oscillator for testing OCP solver """

    def __init__(self, mu, x0, xf, tf):
        self.mu = mu
        self.x0 = x0
        self.xf = xf
        self.tf = tf

    def dyn(self, x):
        # returns f,g evaluated at x (vectorized)
        return np.array([x[1],-x[0] + self.mu*(1-x[0]**2)*x[1]]), np.vstack((np.zeros_like(x[0]),np.ones_like(x[0]))).squeeze()

    def dynamics(self, x, t, u):  # integrable function
        f, g = self.dyn(x)
        return f + g*u(t)

    def jac(self, x, *args):
        x1, x2 = x
        shape = [x.shape[0]]
        shape.extend(x.shape)
        A = np.zeros(shape)

        B = np.vstack((np.zeros_like(x1),np.ones_like(x1))).squeeze()

        A[0,1,:] = np.ones_like(x[0])
        A[1,0,:] = -np.ones_like(x[0]) - 2*self.mu*x1*x2
        A[1,1,:] = self.mu*(1-x[0]**2)

        return np.moveaxis(A, -1, 0), np.moveaxis(B, -1, 0)

    def lagrange(self, t, x, u):
        return np.array([cvx.abs(ui) for ui in u])

    def mayer(self, *args, **kwargs):
        # return cvx.Problem(cvx.Minimize(0*cvx.norm(u,'inf')))
        return 0

    def constraints(self, t, x, u, x_ref, u_ref):
        """ Implements all constraints, including:
            boundary conditions
            control constraints
            trust regions
        """
        # Relaxed final conditions - we can make P a variable (or vector) and penalize it heavily in the cost function like v
        # bc = [x[0] == x0, x[-1] <= xf+P*1, x[-1] >= xf-P*1]
        bc = [x[-1] == self.xf, x[0] == x_ref[0]]
        trust_region = 4
        umax = 3
        # bc = [x[0] == x0]   # Initial condition only
        # bc = [x[-1] == xf]  # Final condition only   

        constr = []
        for ti, xr in zip(t, x_ref):
            # constr += [cvx.norm((x[t]-xr)) <= trust_region]   
            # constr += [cvx.quad_form(x[ti], np.eye(2)/trust_region**2) < 1]
            constr += [cvx.norm(x[ti]-xr) < trust_region]  # Documentation recommends norms over quadratic forms
            constr += [cvx.abs(u[ti]) <= umax]  # Control constraints
        return constr + bc

    def plot(self, T, U, X, J, ti):
        for i, xux in enumerate(zip(T, U, X)):

            t, u, xc = xux

            # plt.figure(1)
            # plt.plot(x[0], x[1], label=str(i))
            # plt.title('State Iterations (Integration)')
            plt.figure(5)
            plt.plot(xc[0], xc[1], label=str(i))
            plt.title('State Iterations (Discretization)')
            plt.figure(2)
            plt.plot(t, u, label=str(i))
            plt.title('Control Iterations')

            plt.figure(3)
            xcvx = interp1d(T[-1], X[-1].T, kind='linear', axis=0, assume_sorted=True)(ti).T
            plt.plot(X[-1][0], X[-1][1], '*-', label='Chebyshev Nodes')
            plt.plot(xcvx[0], xcvx[1], 'ko', label='Mesh Points')
            # plt.plot(X[-1][0], X[-1][1], label='Integration')
            plt.title('Optimal Trajectory')
            plt.legend()

            plt.figure(4)
            plt.plot(T[-1], U[-1])
            plt.title('Optimal control')

            plt.figure(7)
            plt.semilogy(J, 'o-')
            plt.ylabel('Objective Function')
            plt.xlabel('Iteration')

            mesh.plot(show=False)
            for fig in [2,3,5]:
                plt.figure(fig)
                plt.legend()
            plt.show()

if __name__ == "__main__":
    vdp = TestClass(mu=0.5, x0=[2, 2], xf=[0, 0], tf=8)

    guess = {}
    
    t = np.linspace(0,8,20)
    u = np.zeros_like(t)
    x = vdp.integrate(vdp.x0, u, t)
    guess['state'] = x
    guess['control'] = u 
    guess['time'] = t 
    guess['mesh'] = [4]*3
    vdp.solve(guess)
