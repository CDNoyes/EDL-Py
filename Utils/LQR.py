import numpy as np
from RK4 import RK4 


def LQR(x0, tf, A, B, Q, R, Qf):
    """Computes the optimal control to a continuous time
       finite-horizon linear quadratic regulator
    """

    n = np.shape(np.asarray(x0).squeeze())
    m = np.shape(R)[0]

    if Qf is None:
        P0 = np.zeros((n,n))
    else:
        P0 = Qf
    
    Ri = np.linalg.inv(R)
    t = np.linspace(0, tf, 100)
    P = RK4(_dyn, P0, t, args=(A, B, Q, Ri))[::-1]
    for Pi in P:
        K = Ri @ B.T @ P

    return K 

def _dyn(P, t, A, B, Q, Ri):

    Pdot = A.T @ P + P @ A + P @ B @ Ri @ B.T @ P + Q

    return Pdot

if __name__ == "__main__":
    A = np.diag([1,0])
    B = np.array([0,1])
    Q = np.diag([1,1])
    R = np.array([[1]])
    Qf = np.diag([1,10])
    x0 = np.random.random((2,))
    K = LQR(x0, 5, A, B, Q, R, Qf)