import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx

def main():
    # n = 2
    # m = 1
    #
    # T = 40
    # tf = 4 # assume known and fixed for now
    # dt = tf/T
    #
    # x = cvx.Variable(n, T+1)
    # u = cvx.Variable(m, T)
    #
    # A = np.eye(n)
    # A[0,1] = 1*dt
    # B = dt*np.array([0,1]).T
    # x_0 = np.array([-3,0]).T
    # constr = [x[:,t+1] == A*x[:,t] + B*u[:,t] for t in range(T)]    # dynamics
    # constr += [norm(u[:,t], 'inf') <= 1 for t in range(T)]          # control constraints
    np.random.seed(1)
    n = 2
    m = 1
    T = 200
    alpha = 0.02
    # alpha = cvx.Variable() # for free final time
    # constr_dt = alpha >= 0.001 # min step size allowed
    beta = 5
    A = np.eye(n) + alpha*np.array([[0,1],[0,0]])
    # B = np.random.randn(n,m)
    B = alpha*np.array([0,1]).T
    # x_0 = beta*np.random.randn(n,1)
    C = np.zeros((n+1,1))
    C[-1] = 1 # C is used for free final time problems
    x_0 = np.zeros((n,1))
    x = cvx.Variable(n, T+1)
    u = cvx.Variable(m, T)
    states = []
    for t in range(T):
        cost = 0*cvx.sum_squares(x[:,t+1]) + 0*cvx.sum_squares(u[:,t]) + cvx.sum_squares(x[0,t]-0.01*np.sin(400*np.pi*alpha*t/T))
        constr = [x[:,t+1] == A*x[:,t] + B*u[:,t],
                  cvx.norm(u[:,t], 'inf') <= 1]
                  # x[1,t] <= 8]
        states.append( cvx.Problem(cvx.Minimize(cost), constr) )
    # sums problem objectives and concatenates constraints.
    prob = sum(states) #+ cvx.Problem(cvx.Minimize(cvx.sum_squares(x[:2,-1])))
    prob.constraints += [x[:,0] == x_0]
    # prob.constraints += [x[:,T] == 0, x[:,0] == x_0]
    prob.solve()
    print "status:", prob.status
    print "optimal value", prob.value

    f = plt.figure()
    # print u[0,:].value

    # Plot (u_t)_1.
    ax = f.add_subplot(311)
    plt.plot(u[0,:].value.A.flatten())
    plt.ylabel(r"$(u_t)_1$", fontsize=16)
    plt.yticks(np.linspace(-1.0, 1.0, 3))
    # plt.xticks([])

    # Plot (u_t)_2.
    # plt.subplot(4,1,2)
    # plt.plot(u[1,:].value.A.flatten())
    # plt.ylabel(r"$(u_t)_2$", fontsize=16)
    # plt.yticks(np.linspace(-1, 1, 3))
    # plt.xticks([])

    # Plot (x_t)_1.
    plt.subplot(3,1,2)
    x1 = x[0,:].value.A.flatten()
    ref = [0.01*np.sin(400*np.pi*alpha*t/T) for t in range(T)]
    plt.plot(x1)
    plt.plot(ref,'k--')
    plt.ylabel(r"$(x_t)_1$", fontsize=16)
    # plt.yticks([-10, 0, 10])
    # plt.ylim([-10, 10])
    # plt.xticks([])

    # Plot (x_t)_2.
    plt.subplot(3,1,3)
    x2 = x[1,:].value.A.flatten()
    plt.plot(x2)
    # plt.yticks([-25, 0, 25])
    # plt.ylim([-25, 25])
    plt.ylabel(r"$(x_t)_2$", fontsize=16)
    plt.xlabel(r"$t$", fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
