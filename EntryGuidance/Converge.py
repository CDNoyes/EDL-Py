# Bootstrapping based convergence estimates 


import numpy as np 
import matplotlib.pyplot as plt 
from numpy.random import randint 

def Bootstrap(fun, data, sampleSizes, resamples=100):
    
    """
        Inputs:
            fun - A function operating on data which returns the value to be bootstrapped - could be a mean, percentile, variance, etc 
            data -  an array of length N, where N is greater than (or equal) to the largest sample size
            sampleSizes - a collection of sizes at which to bootstrap the results. More sizes will produce a smoother graph
            resamples - The number of times a sampleSize is drawn from the data. More resampling will reduce variability to a point.
    
    """
    N = len(data)
    Means = []
    Stds = []
    
    for sampleSize in sampleSizes:
        ind = randint(N,size=(resamples, sampleSize)) # all of the indices used to compute the values and their variability 
        data_point = []
        for subset in ind:
            data_subset = data[subset]
            fun_subset = fun(data_subset)
            data_point.append(fun_subset)
        data_mean = np.mean(data_point)
        data_std = np.std(data_point)
        Means.append(data_mean)
        Stds.append(data_std)
    return np.array(Means), np.array(Stds) 
        
def test():
    """
        Bootstraps the a gaussian with known mean and std deviation 
    """
    from numpy.random import randn 
    mean = 2
    std = 1
    
    x = mean + std*randn(5000) # The data 
    p99 = np.percentile(x,99.68) # Used as truth 
    
    # First let's recover the mean 
    sizes = [2,5,10,50,100,250,500,1000,2000,5000]
    mu, sig = Bootstrap(np.mean, x, sizes)
    
    Plot(sizes, mu, sig,label='Mean')
    plt.plot(sizes,mean*np.ones_like(sizes),label='True')
    plt.legend()
    
    
    # Then we can try to find the std deviation = 1 
    mu, sig = Bootstrap(np.std, x, sizes)
    
    Plot(sizes, mu, sig,label='StdDev',fignum=2)
    plt.plot(sizes,std*np.ones_like(sizes),label='True')
    plt.legend()
    
    # Finally lets try to find the tails via percentile
    mu, sig = Bootstrap(lambda y:np.percentile(y,99.68), x, sizes)
    
    Plot(sizes, mu, sig,label='99.68%-ile',fignum=3)
    plt.plot(sizes,p99*np.ones_like(sizes),label='True')
    plt.legend()  
    
    
    plt.show()
    
def Plot(sizes, mu, sig, fignum=None, label=None):
    if fignum is not None:
        plt.figure(fignum)
    plt.plot(sizes, mu,label=label)
    plt.plot(sizes, mu+3*sig,'k--',label=label+'$\pm 3\sigma$')
    plt.plot(sizes, mu-3*sig,'k--')
    plt.xlabel('Monte Carlo Sample Size')
    
if __name__ == "__main__":
    test()