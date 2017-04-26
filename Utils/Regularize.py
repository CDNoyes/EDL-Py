import numpy as np 

def Regularize(A, eps=0):
    """
    REGULARIZE Computes the nearest symmetric, positive (semi)definite approximation of A in the sense of the Frobenius norm.
    REGULARIZE(A,EPSILON) finds a regularized version of the matrix A. When
    called with only one input argument A, the result X will be positive
    semidefinite unless A was already positive definite to begin with. When
    called with a second, positive argument EPSILON, the resulting matrix X
    is guaranteed to be positive definite with minimum eigenvalue equal to
    EPSILON. 
    """
    A = np.asarray(A)
    B = (A+A.T)/2.0
    
    eigvals,eigvecs = np.linalg.eig(B)
    eigvals[eigvals<eps] = eps 
    X = np.dot(eigvecs, np.dot(np.diag(eigvals), eigvecs.T))
    return X