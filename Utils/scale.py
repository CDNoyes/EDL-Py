import numpy as np 

def normalize(X):
    """ Transforms data into zero-mean, unit variance
        
        Commonly used in machine learning pre-processing.

        Inputs is an array of [n_variables, n_observations]
        Returns an array of the same shape as the input 
    """
    m = np.mean(X, axis=1)[:,None]
    T = eigenscale(X)

    return T.dot(X-m)

def eigenscale(X):
    """ Performs scaling of variables based on eigensystem of covariance

    X is an array of [n_variables, n_observations]
    returns the scaling array of size [n_variables, n_variables]

    """
    S = np.shape(X)
    if S[0] > S[1]:
        print("Warning in eigenscale: number of variables is greater than number of observations, input may need a transpose.")

    C = np.cov(X)
    vals, vecs = np.linalg.eig(C)

    T = np.array([vec/val for val, vec in zip(np.sqrt(vals), vecs)])
    return T

def test():
    import matplotlib.pyplot as plt 

    m = [10, -3]
    C = np.diag([0.01, 1000]) # Use variables with very different scales 
    X = np.random.multivariate_normal(mean=m, cov=C, size=1000).T

    ms = np.mean(X, axis=1)[:,None]
    
    Y = normalize(X)

    plt.figure(1)
    plt.scatter(*(X-ms))
    plt.title("Unscaled (but centered)")

    plt.figure(2)
    plt.scatter(*Y)
    plt.title("Scaled")

    plt.show()


if __name__ == "__main__":
    test()
        