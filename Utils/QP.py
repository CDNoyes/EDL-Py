""" Quadratic Programming """

import numpy as np
import cvxpy as cvx 


def TRQP(g, H, T, delta, A=None, b=None, Aeq=None, beq=None):
    """ Trust Region Quadratic Program
    min g'x + 0.5 x'Hx subject to ||Tx|| < delta

    Optionally Ax <= b and/or Aeq x ==  b

    """

    n = len(g) 
    x = cvx.Variable(n)
    J = g*x + cvx.quad_form(x, H)

    TR = cvx.norm(T*x) <= delta 
    C = [TR] 
    if A is not None:
        for Ai, bi in zip(A,b):
            Lin = Ai*x <= bi
            C.append(Lin)

    if Aeq is not None:
        for Ai, bi in zip(Aeq, beq):
            LinEq = Aeq*x == beq
            C.append(LinEq)

    P = cvx.Problem(cvx.Minimize(J), C)
    P.solve(solver="ECOS")

    return x.value



if __name__ == "__main__":
    from scipy.optimize import minimize 
    import time 

    n = 4
    H = np.eye(n)
    D = np.eye(n)
    delta = 1
    N = 5000 
    T = []
    for _ in range(N):
        g = np.random.random(n)*2
        A = np.random.random((n,n))*3
        b = np.random.random((n,))
        t0 = time.time()
        x = TRQP(g, H, D, delta, [A], [b], )
        t = time.time()
        T.append(t-t0)
    import seaborn as sns 
    import matplotlib.pyplot as plt 
    sns.distplot(np.array(T)*1000,
             hist_kws=dict(cumulative=True),
             kde_kws=dict(cumulative=True))
    plt.show()
    # print("Solved in {:.1f} ms".format((t-t0)*1000))
    # print(x)
    # print("||Tx|| - D = {:.2f} <= 0".format(np.linalg.norm(D.dot(x))-delta))
    # print("Ax - b = {} <= 0".format(A.dot(x)-b))