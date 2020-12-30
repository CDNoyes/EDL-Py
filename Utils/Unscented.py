import numpy as np

def Transform(mean, covariance, k=0.01):
    """
        Unscented Transform

        Computes (2*n + 1) sigma points and associated weights.


        alpha, beta and k are tuning parameters:
            higher alpha reduces the covariance weight (wc) on the mean (first sigma point)
            beta=2 is optimal for Gaussian distributions
            higher k induces higher distances from the mean (5-8 is often good)

        outputs:
            sigma points
            mean weights
            cov weights
    """
    mean = np.array(mean, ndmin=2)
    covariance = np.array(covariance, ndmin=2)
    n = mean.shape[1]

    # Compute sigma points
    x0 = np.tile(mean, (n,1))
    S = np.linalg.cholesky((n+k)*covariance).T # IT NEEDS THE TRANSPOSE
    X = np.concatenate((mean,x0+S,x0-S))

    # Compute weights
    wm = [k/(n+k)] + [0.5/(n+k) for i in range(2*n)] # mean weights
    # wm = [lamb/(n+lamb)] + [0.5/(n+lamb) for i in range(2*n)] # mean weights
    wc = wm[:]
    # wc[0] += 1-alpha**2 + beta                                      # covariance weights

    return X, np.array(wm), np.array(wc)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    m = [-3,1]
    P = np.array([[1,3],[3,10]])#np.diag((1,3))

    import chaospy as cp
    N = cp.MvNormal(m,P)
    S = N.sample(50000,'S')
    f = lambda x: np.vstack((x[0]**2,x[1]*x[0]))
    for k in [0,2,5,8,10]:
        X,Wm,Wc = Transform(m, P, k=k)
        print(X,Wm)
    #     X = f(X.T).T
    #     mean = Wm.dot(X)
    #     E = X.T-mean[:,None]
    #     cov = E.dot((Wc*E).T)
    #     print("----------------")
    #     print(mean)
    #     print(cov)
    #     print(" ")
    #     # plt.title('$\mu=${}, P={}'.format(mean,cov))
    #     plt.plot(X.T[0],X.T[1],'x',label="k={}".format(k))
    # S = f(S)
    # print("----------------")
    # print("Truth, estimated from {} samples".format(S[0].size))
    # print(S.mean(axis=1))
    # print( np.cov(S))
    # plt.scatter(S[0],S[1],10)
    # plt.show()
