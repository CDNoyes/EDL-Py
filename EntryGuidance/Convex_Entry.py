import sys
sys.path.append('./')
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
import time
from Mesh import Mesh
from numpy import pi


def LTV(x0, A, B, f_ref, x_ref, u_ref, mesh, trust_region, xf, umax, solver='ECOS'):

    """ Solves a convex LTV subproblem

    x0 - fixed initial condition
    A - state linearization around x_ref
    B - control linearization around x_ref
    f_ref - the original nonlinear dynamics evaluated along x_ref
    x_ref - the trajectory used to linearize the problem, the current iterate

    mesh - Mesh.Mesh instance
    Trust region parameters:
        trust_region - the size of the generalized trust region


    """
    t0 = time.time()

    n = A[0].shape[0]
    m = 1
    N = mesh.n_points
    T = range(N)

    x = np.array([cvx.Variable(n) for _ in range(N)])       # This has to be done to "chunk" it later
    u = cvx.Variable(N,m) # Only works for m=1
    v = np.array([cvx.Variable(n) for _ in range(N)]) # Virtual controls
    # Alternatively, we could create meshes of variables directly
    X = mesh.chunk(x)
    A = mesh.chunk(A)
    B = mesh.chunk(B)
    F = mesh.chunk(f_ref)
    Xr = mesh.chunk(x_ref)
    U = mesh.chunk(u)
    Ur = mesh.chunk(u_ref)
    V = mesh.chunk(v)


    bc = [x[0] == x0]   # Initial condition only
    # bc = [x[-1] == xf, x[0] == x0] # Both ends fixed
    # bc = [x[0] == x0, x[-1] <= xf+P*1, x[-1] >= xf-P*1] # Relaxed final conditions
    # bc = [x[-1] == xf]  # Final condition only

    constr = bc
    for t,xr in zip(T,x_ref):
        constr += [cvx.abs(x[t]-xr) <= trust_region]
        constr += [cvx.abs(x[t][-1]) <= np.radians(80)]
        # constr += [v[t][0] == 0]

    # Control constraints
    constr.append(cvx.abs(u) <= umax)

    # Lagrange cost and ode constraints
    states = []
    for d,xi,f,a,b,xr,ur,ui,w,vi in zip(mesh.diffs,X,F,A,B,Xr,Ur,U,mesh.weights,V): # Iteration over the segments of the mesh
        # w = np.abs(w)
        L = 0*(ur-ui)**2                                              # Lagrange integrands for a single mesh
        cost = w*L                                            # Clenshaw-Curtis quadrature
        lqi = np.array([cvx.quad_form(xii-xri,np.diag([1,1,1,1.,1,1,1])) for xii,xri in zip(xi,xr)]) # LQ type cost
        lq = w.dot(lqi)/np.sum(w)
        cost += lq
        # cost += cvx.abs(lq)
        # Estimated derivatives:
        dx = d.dot(xi)

        # Differential equation constraints
        # ode =  [dxi == fi + ai*(xii-xri) + bi*(uii-uri) for xii,fi,ai,bi,xri,uri,uii,dxi,vii in zip(xi,f,a,b,xr,ur,ui,dx,vi) ] # No virtual control
        ode =  [dxi == fi + ai*(xii-xri) + bi*(uii-uri) + vii  for xii,fi,ai,bi,xri,uri,uii,dxi,vii in zip(xi,f,a,b,xr,ur,ui,dx,vi) ] # Iteration over individual collocation points
        states.append(cvx.Problem(cvx.Minimize(cost), ode))

    # Mayer Cost, including penalty for virtual control
    hf = x[-1][0]/1000 - 3397. # Not sure if correct syntax
    miss = 0*(x[-1][1]-xf[1])**2 + (x[-1][2]-xf[2])**2
    weight = 3397
    Phi = cvx.Problem(cvx.Minimize(weight*miss - 0*hf))
    Penalty = cvx.Problem(cvx.Minimize(1e5*cvx.norm(cvx.vstack(*v),'inf') ))

    # sums problem objectives and concatenates constraints.
    prob = sum(states) + Penalty #+ Phi
    prob.constraints += constr

    t1 = time.time()
    # prob.solve(solver='CVXOPT')
    # prob.solve(solver='SCS')
    prob.solve(solver=solver)
    t2 = time.time()

    print "status:        ", prob.status
    print "optimal value: ", np.around(prob.value,3)
    print "solution time:  {} s".format(np.around(t2-t1,3))
    print "setup time:     {} s".format(np.around(t1-t0,3))

    try:
        x_sol = np.array([xi.value.A for xi in x]).squeeze()
        u_sol = u.value.A.squeeze()
        try:
            v_sol = np.array([xi.value.A for xi in v]).squeeze()
            print "penalty value:  {}\n".format(np.linalg.norm(v_sol.flatten(), np.inf))
            for v in v_sol.T:
                plt.figure()
                plt.plot(x_sol.T[3], v)
            plt.show()
        except:
            print "Could not compute penalty value"

        return x_sol.T, u_sol, prob.value
    except:
        return x_ref.T, u_ref, None





if __name__ == "__main__":
    pass
