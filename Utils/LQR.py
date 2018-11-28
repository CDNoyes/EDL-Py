import numpy as np
from scipy.interpolate import interp1d
from RK4 import RK4 


def LQRI(tf, A, B, Q, R, Qf):
    """Computes the optimal control to a continuous time
       finite-horizon linear quadratic regulator
       TIME INVARIANT VERSION 
    """

    n = np.shape(A)[0]
    m = np.shape(R)[0]

    if Qf is None:
        P0 = np.zeros((n,n))
    else:
        P0 = Qf
    
    Ri = np.linalg.inv(R)
    t = np.linspace(0, tf, 1000)
    P = RK4(_dyni, P0, t, args=(A, B, Q, Ri))[::-1]
    K = Ri @ B.T @ P

    return np.transpose(K, (0, 2, 1))


def LQR(t, A, B, Q, R, Qf):
    """Computes the optimal control to a continuous time
       finite-horizon linear quadratic regulator
    """

    try:
        n = np.shape(A[0])[0]
    except IndexError:
        n = 1 
    m = np.shape(R)[0]

    if Qf is None:
        P0 = np.zeros((n,n))
    else:
        P0 = Qf
    
    Ri = np.linalg.inv(R)
    Af = interp1d(t, A, axis=0)
    Bf = interp1d(t, B, axis=0)
    P = RK4(_dyn, P0, t, args=(Af, Bf, Q, Ri))[::-1]
    K = np.array([Ri.dot(Bi.T).dot(Pi) for Bi,Pi in zip(B,P)])
    if n > 1:
        K = np.transpose(K, (0, 2, 1))
    return K


def _dyn(P, t, Af, Bf, Q, Ri):
    A = Af(t)
    B = Bf(t)
    Pdot = A.T.dot(P) + P.dot(A) - P.dot(B).dot(Ri).dot(B.T).dot(P) + Q

    return Pdot


def _dyni(P, t, A, B, Q, Ri):

    Pdot = A.T @ P + P @ A - P @ B @ Ri @ B.T @ P + Q

    return Pdot


def test():
    import matplotlib.pyplot as plt 
    import chaospy as cp 
    from scipy.interpolate import interp1d

    A = np.array([[0, 1],[-0.5, 0]])
    B = np.array([[0, 1]]).T
    Q = np.diag([1, 1])
    R = np.array([[1]])
    Qf = np.diag([10, 10])
    K = LQRI(5, A, B, Q, R, Qf)
    t = np.linspace(0, 5, 1000)

    plt.figure()
    plt.plot(t, K.squeeze())

    N = cp.Normal(0, 1)
    x0 = np.array( (N.sample(20), N.sample(20)))

    def LTI(x,t, K):
        k = K(t).T
        u = -k@x
        return A@x +B@u

    Ki = interp1d(t, K, axis=0)
    X = RK4(LTI, x0, t, args=(Ki, ))
    
    plt.figure()
    plt.plot(t, X[:, 0])
    plt.show()


def test1d():
    import matplotlib.pyplot as plt 
    x = np.hstack( (np.linspace(1,1.5,50,), np.linspace(1.49,0,50)))
    t = np.linspace(0, 1, 100)
    A = -2*np.abs(x)
    B = np.ones_like(t)
    Q = [0]
    Qf = [1]
    R = [[0.1]] 
    K = LQR(t, A, B, Q, R, Qf)

    plt.plot(t, K)
    plt.show()

if __name__ == "__main__":
    # test()
    test1d()