""" Utilities for Chebyshev polynomial based pseudospectral methods """

import numpy as np


def ChebyshevQuad( N ):
    """ Computes the points and weights associated with Curtis-Clenshaw quadrature of Nth degree

        This is used to discretize Langrangian cost functions when using a
        Chebyshev pseudospectral method to solve OCP

    """
    assert(N > 1)

    K = np.arange(N+1)
    points = -np.cos(np.pi*K/N)
    w = np.zeros((N+1,))

    if np.mod(N,2): # odd N
        w[0] = 1./(N**2)
        w[N] = w[0]
        Nsum = (N-1)/2 + 1

    else: # even N
        w[0] = 1./(N**2 - 1)
        w[N] = w[0]
        Nsum = N/2 + 1

    for s in range(1,Nsum):
        j = np.arange(Nsum)
        tempval = np.cos(2*np.pi*j*s/N)/(1-4*j**2)
        tempval[0] = tempval[0]*0.5
        tempval[-1] = tempval[-1]*0.5
        w[s] = 4./N*sum(tempval)
        w[N-s] = w[s]

    return points, w

def ChebyshevDiff(n):
    """
    Returns the (Sorted) Chebyshev Differentiation Matrix, D, with n+1 nodes
    using the collocated points t.

    Unsorted collocation points are used to compute the standard
    differentiation matrix, then the matrix is converted to its sorted form.
    See Ross, Fahroo (2002) for details.
    """
    t = np.cos(np.pi*np.arange(0,n+1)/n) # Unsorted collocation points
    D = np.empty((n+1,n+1))

    for j in range(n+1):
        for k in range(n+1):
            if j == 0 and j == k:
                D[k,j] = (2*n**2+1)/6.
            elif j == n and j ==k:
                D[k,j] = -(2*n**2+1)/6.
            elif j == k:
                D[k,j] = -t[k]/2./(1-t[k]**2)
            else:
                if k == 0 or k == n:
                    ck = 2.
                else:
                    ck = 1.

                if j == 0 or j == n:
                    cj = 2.
                else:
                    cj = 1.
                D[k,j] = (ck/cj)*( ( (-1)**(j+k) ) / (t[k]-t[j]) )

    return -D


def test_quad():
    """ Seems ChebQuad works best with even values """
    from scipy.integrate import quad

    y = lambda x: (x**4 - 2 * x**3)*np.sin(x) + np.exp(0.1*x)*np.cos(x)*x

    bound = 1

    I = quad(y, -bound,bound)[0]

    # A single 4th order integration is sufficient
    xi,wi = ChebyshevQuad(4)
    I4 = sum(y(xi*bound)*wi)*bound

    # A single 2nd order integration will not integrate a 4th order polynomial exactly
    xi,wi = ChebyshevQuad(2)
    I2 = sum(y(xi*bound)*wi)*bound

    # Split the interval and use 2 order - works quite well even for non-polynomial functions
    xi,wi = ChebyshevQuad(2)
    intervals = 15
    ti = np.linspace(-bound,bound,intervals+1)
    Isplit = 0

    for ta,tb in zip(ti,ti[1:]):
        tau = 0.5*((tb-ta)*xi + ta+tb) #map the collocation points to the true time interval
        Isplit += sum(y(tau)*wi)*(0.5*(tb-ta))


    print I
    print I4
    print I2
    print Isplit

if __name__ == "__main__":
    test_quad()
