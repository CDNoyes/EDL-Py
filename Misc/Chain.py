import numpy as np 
from numpy.random import uniform 
from numpy.random import multivariate_normal as normal 

import matplotlib.pyplot as plt 

import abc 

class Chain(object):
    
    __metaclass__ = abc.ABCMeta
    
    def __init__(self):
        pass 
        
    @abc.abstractmethod
    def generate(self, n_steps, n_population):       
        raise NotImplementedError('Child class must define generate method')
        
            
    def plot(self,x,mean=True,show=True):
        n_steps,n_population = x.shape
        t = range(n_steps)
        
        if n_population == 2:
            plt.figure()
            plt.plot(x.T[0],x.T[1])
            
        else:
            m = np.mean(x,axis=1)
            
            plt.figure()
            plt.plot(t,x)
            plt.plot(t,m,'k--',label='Mean')
            
        if show:
            plt.show()

class NormalChain(Chain):
    
    def generate(self, n_steps, n_population, step_size=1, rate_limit=1):
        x = [np.zeros((n_population))]
        # x = [2*np.random.random((n_population))-1]
        
        cov = np.diag(np.ones((n_population))*rate_limit*step_size/3)**2
        for _ in range(n_steps):
            x.append(normal(x[-1],cov))
            
        return np.array(x)     
            
            
class UniformChain(Chain):

    def generate(self, n_steps, n_population, step_size=1, rate_limit=1):
        x = [np.zeros((n_population))]

        for _ in range(n_steps):
            low = x[-1] - rate_limit*step_size
            high = x[-1] + rate_limit*step_size
            x.append(uniform(low,high))
            
        return np.array(x)     
               
            
if __name__ == "__main__":
    
    N = NormalChain()
    x = N.generate(100,150,rate_limit=0.1)
    N.plot(x,True,False)
    
    U = UniformChain()
    y = U.generate(100,150)
    U.plot(y,True,True)
    
    # x2 = N.generate(100,2)
    # N.plot(x2)