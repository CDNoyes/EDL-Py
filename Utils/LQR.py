import numpy as np
from RK4 import RK4 


def LQR(tf, A, B, Q, R, Qf):
    """Computes the optimal control to a continuous time
       finite-horizon linear quadratic regulator
    """

    n = np.shape(A)[0]
    m = np.shape(R)[0]

    if Qf is None:
        P0 = np.zeros((n,n))
    else:
        P0 = Qf
    
    Ri = np.linalg.inv(R)
    t = np.linspace(0, tf, 1000)
    P = RK4(_dyn, P0, t, args=(A, B, Q, Ri))[::-1]
    print(P[0])
    K = Ri @ B.T @ P

    return np.transpose(K, (0, 2, 1))


def _dyn(P, t, A, B, Q, Ri):

    Pdot = A.T @ P + P @ A - P @ B @ Ri @ B.T @ P + Q

    return Pdot


if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    import chaospy as cp 
    from scipy.interpolate import interp1d

    A = np.diag([1, 1])
    B = np.array([[0, 1]]).T
    Q = np.diag([1, 1])
    R = np.array([[1]])
    Qf = np.diag([10, 10])
    K = LQR(5, A, B, Q, R, Qf)
    t = np.linspace(0, 5, 1000)
    # plt.plot(t, K.squeeze())

    N = cp.Normal(0, 1)
    x0 = np.array( (N.sample(20), N.sample(20)))

    def LTI(x,t, K):
        k = K(t).T
        u = -k@x
        return A@x +B@u

    Ki = interp1d(t, K, axis=0)
    X = RK4(LTI, x0, t, args=(Ki, ))

    plt.plot(t, X[:, 0])
    plt.show()