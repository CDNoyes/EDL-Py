import numpy as np
from scipy.integrate import odeint


def dynamics3D(X, t, thrust_vector, Isp, g=np.array([0,0,-3.71])):

    # x,y,z,u,v,w,m = X
    dR = X[3:6]
    dV = thrust_vector/X[-1] + g
    dm = -np.linalg.norm(thrust_vector)/(Isp*9.81)

    return np.concatenate((dR,dV,dm[None]))


def controlled(control_fun):
    def internal(x, t, *args):
        return dynamics3D(x, t, control_fun(x), *args)
    return internal

def adapt_lyap(x):
    x = np.abs(x)
    z = np.concatenate((x[3:6],x[0:3]))
    return lyap_control(x, np.diag(z))

def lyap_control(x, Q):
    from Utils.submatrix import submatrix

    Qr = submatrix(Q, range(0,3), cols=None)
    Qv = submatrix(Q, range(3,6), cols=None)
    # Qvi = np.linalg.inv(Qv)
    # Qvi = np.diag(1/np.diag(Qv)) # Assuming Qv is diagonal, this avoids the small matrix inverse


    r = x[0:3]
    v = x[3:6]
    m = x[6]
    g = np.array([0,0,-3.7])

    if r[2] <= 0:
        T = -m*g
    else:
        T = -m*(g + Qr.dot(r) + Qv.dot(v))
        Tmag = np.linalg.norm(T)
        T = np.clip(Tmag,40*m,70*m)*T/Tmag

        # T = m*(-g - Qvi.dot(Qr.dot(r) + K*v))
    return T

def test():
    from functools import partial
    import matplotlib.pyplot as plt

    Q = np.diag([10,10,10,150,150,150])*100
    u = partial(lyap_control, Q=Q)
    # u = adapt_lyap

    m0 = 8500.
    # ve = 290.*9.81
    Isp = 290
    x0 = np.array([-3200., 400, 3200, 625., -80, -270., m0])

    t = np.linspace(0,75,100)
    x = odeint(controlled(u), x0, t, args=(Isp,))
    idx = x[:,2]>10
    T = np.array([u(xi) for xi in x[idx]])
    print T.shape
    Tmag = np.linalg.norm(T, axis=1)

    x,y,z,u,v,w,m = x[idx,:].T
    t = t[idx]

    plt.figure()
    plt.plot(t,x)
    plt.plot(t,y)
    plt.plot(t,z)

    plt.figure()
    plt.plot(t,u)
    plt.plot(t,v)
    plt.plot(t,w)

    plt.figure()
    plt.plot(t,m)

    plt.figure()
    plt.plot(t, T)
    plt.plot(t, Tmag, 'k--')

    plt.show()
