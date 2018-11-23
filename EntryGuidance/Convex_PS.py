import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
from scipy.integrate import odeint
from scipy.interpolate import interp1d
sys.path.append('./')
# from Utils.RK4 import RK4
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
        # ut = interp1d(t, u,
        #               kind='cubic',
        #               assume_sorted=True, 
        #               fill_value=u[-1],
        #               bounds_error=False)
        X = odeint(self.dynamics, x0, t, args=(u,))
        return np.asarray(X)

    def solve(self, guess, scaling=None, max_size=500, max_iter=20, penalty=lambda it: 1e4, plot=False, solver='ECOS', linesearch=0.1, refine=True, verbose=False):
        """ 
        guess
        scaling - this is a vector scale factor used solely during mesh refinement to determine appropriate errors in each state 
        penalty - value used in virtual control penalty term. Set to None for no virtual control 
        linesearch - numeric value, or False, for a linesearch 
        refine - whether or not to perform mesh adaptation. Currently, linesearch and refine cannot be used together. 
        """
        X_cvx = []
        J_cvx = []
        U = []
        T = []
        Tsol = []

        mesh = Mesh(t0=guess['time'][0], tf=guess['time'][-1], orders=guess['mesh'])
        self.mesh = mesh
        t = mesh.times

        # Initial "guess" used for linearization - this should be an input
        u = guess['control']
        x = guess['state']
        ti = guess['time']
                
        # Interpolate the guess onto the initial mesh
        u = interp1d(ti, u, axis=0, bounds_error=False, fill_value='extrapolate', kind='cubic')(t)
        x = interp1d(ti, x, axis=0, bounds_error=False, fill_value='extrapolate', kind='cubic')(t).T
        F = self.dynamics(x, t, u).T
        # print("Dimensions going into jacobian = {}, {}".format(x.shape, u.T.shape))
        A, B = self.jac(x, u.T)

        x_approx = x

        X_cvx.append(x_approx)
        U.append(u)
        T.append(t)

        iters = int(max_iter)                 # Maximum number of iterations
        rel_diff = None

        # Main Loop
        for it in range(iters):
            print("Iteration {}".format(it))
            try:
                if solver.lower() == "ipopt":
                    x_sol, u_sol, J_approx, v_sol, tsolve = self.LTV_I(A, B, F, x_approx.T, u, mesh, penalty(it), verbose)
                else:
                    x_sol, u_sol, J_approx, v_sol, tsolve = self.LTV(A, B, F, x_approx.T, u, mesh, penalty(it), solver)

                if linesearch: # set to 1 to begin without a linesearch,  or 0 or False for no linesearch at all 
                    x_approx = linesearch * x_sol + (1-linesearch) * X_cvx[-1]
                    u = linesearch * u_sol + (1-linesearch) * U[-1]
                else:
                    x_approx = x_sol
                    u = u_sol

            except cvx.SolverError as msg:
                print(msg)

                if linesearch:
                    print("Failed Iteration, updating current solution using a smaller linesearch.")
                    x_approx = 0.5 * X_cvx[-2] + 0.5 * X_cvx[-1]
                    u = 0.5 * U[-2] + 0.5 * U[-1]

                    F = self.dynamics(x_approx, t, u).T
                    A, B = self.jac(x_approx, u.T)

                    X_cvx[-1] = x_approx
                    U[-1] = u

                    continue

                else:
                    print("Failed iteration and no linesearch enabled, aborting sequence.")
                    break

            if J_approx is None:  # Failed iteration, can this happen? Isnt it caught above?
                break 

            else:

                F = self.dynamics(x_approx, t, u).T
                A, B = self.jac(x_approx, u.T)

                X_cvx.append(x_approx)
                J_cvx.append(J_approx)
                U.append(u)
                T.append(t)
                Tsol.append(tsolve)

                if len(J_cvx) > 2:
                    if np.abs(J_cvx[-1]) > 1e-3:
                        rel_diff = np.abs(J_cvx[-1]-J_cvx[-2])/np.abs(J_cvx[-1])
                        print("Relative change in cost function = {:.2f}%\n".format(rel_diff*100))
                    else:  # near zero cost so we use the absolute difference instead
                        rel_diff = np.abs(J_cvx[-1]-J_cvx[-2])
                if refine and (rel_diff is None or rel_diff < 0.1):  # check state convergence instead?
                        # In contrast to NLP, we only refine after the solution converges on the current mesh
                        if it < iters-1:
                            current_size = mesh.times.size
                            if it%2 and False:
                                _ = mesh.refine(u, np.zeros_like(u), tol=1e-2, rho=0)  # Control based refinement
                            else:
                                refined = mesh.refine(x_approx.T, F, tol=1e-4, rho=1.5, scaling=scaling, verbose=False)  # Dynamics based refinement for convergence check
                            if mesh.times.size > max_size:
                                print("Terminating because maximum number of collocation points has been reached.")
                                break
                            if not refined and rel_diff is not None:  # or (v_sol <= 1e-6 and rel_diff < 0.02)
                                print('Terminating with optimal solution.')
                                break
                            t_u = t
                            print("Mesh refinement resulted in {} segments with {} collocation points\n".format(len(mesh.orders), t.size))
                            if 0:
                                mesh.plot(show=True)  # for debugging 
                            t = mesh.times
                            if t[0] < t[-1]:  # Increasing IV
                                fill = (u[0], u[-1])
                                fillx = (x[:,0], x[:,-1])
                            else:
                                fill = (u[-1], u[0])
                                fillx = (x[:,-1], x[:,0])

                            ufun = interp1d(t_u, u, kind='linear', axis=0, copy=True, bounds_error=False, fill_value=fill, assume_sorted=False)
                            u = ufun(t)
                            x_approx = interp1d(t_u, x_approx, kind='linear', axis=-1, copy=True, bounds_error=False, fill_value=fillx, assume_sorted=False)(t)
                            F = self.dynamics(x_approx, t, u).T
                            A, B = self.jac(x_approx, u.T)

        self.sol = {'state' : X_cvx[-1], 'control': U[-1], 'time':T[-1]}
        self.sol_history = {'state' : X_cvx, 'sol_time' : Tsol, 'cost':J_cvx, 'control': U, 'time':T}
        if plot:
            self.plot(T, U, X_cvx, J_cvx, mesh._times, Tsol)
        return self.sol.copy()

    def LTV(self, A, B, f_ref, x_ref, u_ref, mesh, penalty, solver):
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
        if penalty is None:
            v = np.zeros_like(x_ref)
        else:
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
            L = self.lagrange(0, xi, ui, xr, ur, vi)
            cost = np.dot(np.abs(w), L)                # Clenshaw-Curtis quadrature

            # Estimated derivatives:
            dx = d.dot(xi)

            # Differential equation constraints
            ode = [dxi == fi + ai*(xii-xri) + bi*(uii-uri) + vii for xii,fi,ai,bi,xri,uri,uii,dxi,vii in zip(xi,f,a,b,xr,ur,ui,dx,vi) ] # Iteration over individual collocation points

            states.append(cvx.Problem(cvx.Minimize(cost), ode))

        # Mayer Cost, including penalty for virtual control
        Phi = self.mayer(x[-1], u)
        if penalty is None:
            Penalty = 0
        else:
            Penalty = cvx.Problem(cvx.Minimize(penalty*cvx.norm(cvx.vstack(v), 'inf')))

        # sums problem objectives and concatenates constraints.
        C = self.constraints(mesh.times, x, u, x_ref, u_ref)
        prob = sum(states) + Phi + Penalty + cvx.Problem(cvx.Minimize(0), C)
        
        # prob.constraints.extend(self.constraints(mesh.times, x, u, x_ref, u_ref))

        t1 = time.time()
        prob.solve(solver=solver)
        t2 = time.time()

        print("status:        {}".format(prob.status))
        print("optimal value: {}".format(np.around(prob.value, 3)))
        print("solution time:  {} s".format(np.around(t2-t1, 3)))
        print("setup time:     {} s".format(np.around(t1-t0, 3)))

        try:
            x_sol = np.array([xi.value for xi in x]).squeeze()
            u_sol = np.array([ui.value for ui in u]).squeeze()
            v_sol = np.array([xi.value for xi in v]).squeeze()
            if penalty is not None:
                penalty = np.linalg.norm(v_sol.flatten(), np.inf)
                print("penalty value:  {}\n".format(penalty))
            else:
                penalty = 0

            return x_sol.T, u_sol, prob.value, penalty, t2-t1
        except Exception as e:
            print(e)
            return x_ref.T, u_ref, None, 0, 0

    def LTV_I(self, A, B, f_ref, x_ref, u_ref, mesh, penalty, verbose=False, solve_num=3, *args):
        """ Solves a convex subproblem with IPOPT """

        from Utils import ipopt 

        solver = ipopt.Solver()

        t0 = time.time()

        n = A[0].shape[0]
        try:
            m = B[0].shape[1]
        except IndexError:
            m = 1

        N = mesh.n_points
        T = range(N)

        guess = np.zeros((N,n))
        guess_u = np.zeros((N,m)).squeeze()

        X0 = solver.create_vars(guess)
        U0 = solver.create_vars(guess_u)

        X = mesh.chunk(X0)
        U = mesh.chunk(U0)
        A = mesh.chunk(A)
        B = mesh.chunk(B)
        XR = mesh.chunk(x_ref)
        UR = mesh.chunk(u_ref)

        # Constraints
        self.solver = solver 
        self.solver.model.options.SOLVER = solve_num
        self.constraints(mesh.times, X0, U0, x_ref, u_ref)

        L = []
        for a, x, b, u, d, w, xr, ur in zip(A,X,B,U, mesh.diffs, mesh.weights, XR, UR): # Iterate over mesh segments 
            solver.StateSpace(a, x, b, u, d) # linear dynamic constraints 
            
            # Running cost computation 
            vi = 0 # not virtual control implementation 
            lagrange = self.lagrange(0, x, u, xr, ur, vi)
            la_var = solver.model.Var(0.)
            solver.Equation(la_var == w.dot(lagrange)) # The use of these intermediate variables allows the obj to be written as a small sum. This avoids the 15k character limit. 
            L.append(la_var)


        Phi = self.mayer(X0[-1], x_ref[-1])
        solver.Obj(sum(L) + Phi)  # Overall objective is sum of endpoint and running costs 

        t_prep = time.time()

        solver.solve(verbose=verbose) 

        t_solve = time.time()
        if True:
            print("optimal value:       {}".format(np.around(solver.model.options.OBJFCNVAL, 3)))
            print("Pre-processing time: {:.3f} s".format(t_prep - t0))
            print("Solver time:         {:.3f} s".format(t_solve - t_prep))
            print("Total time:          {:.3f} s".format(t_solve - t0))

        x_sol = solver.get_values(X0) + x_ref
        u_sol = solver.get_values(U0) + u_ref

        return x_sol.T, u_sol, solver.model.options.OBJFCNVAL, 0, t_solve-t_prep


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
        return f + g*u

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

    def lagrange(self, t, x, u, *args):
        # return np.array([ui**2 for ui in u])
        return np.array([cvx.abs(ui) for ui in u])
        # return np.zeros_like(u)

    def mayer(self, xf, u, *args, **kwargs):
        # return cvx.Problem(cvx.Minimize(cvx.norm(cvx.vstack(u), 'inf')))
        return 0

    def constraints(self, t, x, u, x_ref, u_ref):
        """ Implements all constraints, including:
            boundary conditions
            control constraints
            trust regions
        """
        bc = [x[-1] == self.xf, x[0] == x_ref[0]]
        trust_region = 4
        umax = 3

        constr = bc
        for ti, xr in enumerate(x_ref):
            # constr += [cvx.norm((x[t]-xr)) <= trust_region]   
            # constr += [cvx.quad_form(x[ti], np.eye(2)/trust_region**2) < 1]
            constr += [cvx.norm(x[ti]-xr) <= trust_region]  # Documentation recommends norms over quadratic forms
            constr += [cvx.abs(u[ti]) <= umax]  # Control constraints
        return constr

    def plot(self, T, U, X, J, ti, tcvx):
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

        self.mesh.plot(show=False)
        for fig in [2,3,5]:
            plt.figure(fig)
            plt.legend()
        plt.show()

def test():
    tf = 3
    vdp = TestClass(mu=0.25, x0=np.array([2., 2]), xf=np.array([0., 0]), tf=tf)

    guess = {}
    
    t = np.linspace(0, tf, 20)
    u = np.zeros_like(t)
    # x = np.array([np.linspace(vdp.x0[i],0,20) for i in [0,1]]).T #
    x = vdp.integrate(vdp.x0, 0, t)
    guess['state'] = x
    guess['control'] = u 
    guess['time'] = t 
    guess['mesh'] = [3]*4
    vdp.solve(guess, max_iter=10, linesearch=False, plot=True)


if __name__ == "__main__":
    test()
