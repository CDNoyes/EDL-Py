import numpy as np
import chaospy as cp
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class VDP(object):
    
    
    
    def __init__(self):
        print "Initialized"
        
    def __dynamics__(self, x, t, mu):
        return np.array([x[1],-x[0] + mu*(1-x[0]**2)*x[1]])
        
    def __jacobian__(self, x, mu):    
        return np.array([[0, 1],[-1-2*x[0]*x[1]*mu, mu*(1-x[0]**2)]])
        
    # def dynamics(self):
        # return lambda x,t: self.__dynamics__(x)
        
    def simulate(self, sample, tf):
        x = odeint(self.__dynamics__, sample[0:2], np.linspace(0,tf,51), args=(sample[2]*0,))
        return x


    def monte_carlo(self, samples, tf, pdf):
        ''' Samples is an (N,3) array-like of deltas [x1,x2,mu] '''
        
        X = np.array([self.simulate(sample,tf) for sample in samples])
        self.outputs = X
        self.samples = samples
        self.pdf = pdf
        print X.shape
        # return X
        
    def plot(self):
        ''' Visualizations of the monte carlo results. '''
        for traj in self.outputs:
            plt.figure(1)
            plt.plot(traj[:,0],traj[:,1],'k',alpha=0.1)

            plt.figure(2)
            plt.plot(traj[-1,0],traj[-1,1],'o')
        
        plt.figure(1)
        for i in range(0,self.outputs.shape[1],10):
            plt.scatter(self.outputs[:,i,0],self.outputs[:,i,1],20,self.pdf)
        plt.show()    
        
    def test(self):
        if 1:
            N1 = cp.Normal(3,0.1)
            N2 = cp.Normal(0,0.3)
            MU = cp.Uniform(0.,.0005)
        elif 0:
            N1 = - 0.5+cp.Beta(2,5)
            N2 = cp.Beta(1,1)-0.5
            MU = cp.Uniform(0.,.0005)

        else:
            N1 = cp.Uniform(2.8,3.2)
            N2 = cp.Uniform(0,0.3)
            MU = cp.Uniform(0.,.05)

        delta = cp.J(N1,N2,MU)
        samples = delta.sample(500,'S').T
        pdf = delta.pdf(samples.T)
        self.monte_carlo(samples,5,pdf)
        self.plot()

if __name__ == '__main__':    
    vdp = VDP()
    vdp.test()