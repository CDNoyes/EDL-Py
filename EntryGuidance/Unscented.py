import numpy as np

def Transform(mean, covariance, alpha=1, beta=2, k=0):
    """
        Unscented Transform

        Computes (2*n + 1) sigma points and associated weights.

        alpha, beta and k are tuning parameters:
            higher alpha reduces the covariance weight (wc) on the mean (first sigma point)
            beta=2 is optimal for Gaussian distributions
            higher k induces higher distances from the mean (5-8 is often good)
    """
    mean = np.array(mean, ndmin=2)
    n = mean.shape[1]
    lamb = alpha**2 * (n+k) - n

    # Compute sigma points
    x0 = np.tile(mean, (n,1))
    S = np.linalg.cholesky((n+lamb)*covariance) # S.dot(S.T) = covariance
    X = np.concatenate((mean,x0+S,x0-S))

    # Compute weights
    wm = [lamb/(n+lamb)] + [0.5/(n+lamb) for i in range(2*n)] # mean weights
    wc = wm[:]
    wc[0] += 1-alpha**2 + beta                                      # covariance weights

    return X, np.array(wm), np.array(wc)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    m = [0,1]
    P = np.diag((1,3))

    import chaospy as cp
    N = cp.MvNormal(m,P)
    S = N.sample(5000,'S')

    for k in [0,2,5,8,10]:
        X,Wm,Wc = Transform(m, P, alpha=1, k=k)
        plt.plot(X.T[0],X.T[1],'x',label="k={}".format(k))
    plt.scatter(S[0],S[1],10)
    plt.show()
