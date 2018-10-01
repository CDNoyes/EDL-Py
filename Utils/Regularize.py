import numpy as np


def Regularize(A, eps=0):
    """
    Computes the nearest symmetric, positive (semi)definite approximation of A in the sense of the Frobenius norm.
    Returns a regularized version of the matrix A. When
    called with only one input argument A, the result X will be positive
    semidefinite unless A was already positive definite to begin with. When
    called with a second, positive argument EPSILON, the resulting matrix X
    is guaranteed to be positive definite with minimum eigenvalue equal to
    EPSILON.
    """
    A = np.asarray(A)
    B = (A + A.T)/2.0

    eigvals, eigvecs = np.linalg.eig(B)
    eigvals[eigvals < eps] = eps
    X = np.dot(eigvecs, np.dot(np.diag(eigvals), eigvecs.T))
    return X


def AbsRegularize(A, shift=0):
    """
    Returns a regularized version of the matrix A in which negative eigenvalues
    are replaced by their absolute value. This has application in saddle-free
    Newton methods. See "Identifying and attacking the saddle point problem in
    high dimensional non-convex optimization"

    The shift parameter is used to handle zero eigenvalues,
    which can be shifted to ensure positive definiteness.

    """
    A = np.asarray(A)
    B = (A + A.T)/2.0

    eigvals, eigvecs = np.linalg.eig(B)
    eigvals = np.maximum(np.abs(eigvals), shift)
    X = np.dot(eigvecs, np.dot(np.diag(eigvals), eigvecs.T))
    return X


def Split(A):
    """ Returns the matrices P and N such that
            A = P + N
            P >= 0
            N <= 0 
    """
    
    A = np.asarray(A)
    B = (A + A.T)/2.0

    eigvals, eigvecs = np.linalg.eig(B)
    pos = eigvals[:]
    neg = eigvals[:]
    pos[eigvals < 0] = 0
    neg[eigvals > 0] = 0
    P = np.dot(eigvecs, np.dot(np.diag(pos), eigvecs.T))
    N = np.dot(eigvecs, np.dot(np.diag(neg), eigvecs.T))
    return P, N
