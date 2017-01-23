import numpy as np
import chaospy as cp
import matplotlib.pyplot as plt
from scipy.integrate import odeint



class VDP(object):
    ''' A van der pol oscillator class '''
    
    def __init__(self):
        print "Initialized"
        
    def __dynamics__(self, x, t, mu):
        return np.array([x[1],-x[0] + mu*(1-x[0]**2)*x[1], self.__pf__(x,mu)])
        
    # def __jacobian__(self, x, mu):    
        # return np.array([[0, 1],[-1-2*x[0]*x[1]*mu, mu*(1-x[0]**2)]])
        
    def __pf__(self, x, mu):
        return -mu*(1-x[0]**2)*x[2]
        
    def simulate(self, sample, tf, p):
        x = odeint(self.__dynamics__, [sample[0],sample[1],p], np.linspace(0,tf,51), args=(sample[2],))
        return x


    def monte_carlo(self, samples, tf, pdf):
        ''' Samples is an (N,3) array-like of deltas [x1,x2,mu] '''
        
        X = np.array([self.simulate(sample,tf,p0) for sample,p0 in zip(samples,pdf)])
        self.outputs = X
        self.samples = samples
        self.pdf = pdf
        
    def plot(self):
        ''' Visualizations of the monte carlo results. '''
        if 0:
            for traj in self.outputs:
                plt.figure(1)
                plt.plot(traj[:,0],traj[:,1],'k',alpha=0.1)

            plt.figure(1)
            for i in range(0,self.outputs.shape[1],10):
                # plt.scatter(self.outputs[:,i,0],self.outputs[:,i,1],20,self.pdf)
                plt.scatter(self.outputs[:,i,0],self.outputs[:,i,1],20,self.outputs[:,i,2])
            
            
        plt.figure(3)
        counts,xe,ye,im = plt.hist2d(self.outputs[:,-1,0], self.outputs[:,-1,1], normed=True, bins=(250,250))
        plt.title('QMC')    
        plt.colorbar()    
            
        # print type(counts)    
        # print counts.shape    
            
        vmax = np.max((counts.max(),self.outputs[:,-1,2].max()))    
            
        plt.figure(2)
        plt.scatter(self.outputs[:,-1,0],self.outputs[:,-1,1],20,self.outputs[:,-1,2],vmin=0) #PF operator
        plt.scatter(self.outputs[:,-1,0],self.outputs[:,-1,1],20,self.outputs[:,-1,2],vmin=0,vmax=vmax) #force same color range as the histogram
        plt.title('PF Operator')
        plt.colorbar()

            
        
    def test(self):
        ''' 
            Runs the VDP tests:
                Monte Carlo 
                Perron-Frobenius Operator
            
        '''
        if 0:
            N1 = cp.Normal(3,0.1)
            N2 = cp.Normal(0,0.3)
            MU = cp.Normal(0.1,.1)
        elif 1:
            N1 = - 0.5+cp.Beta(2,5)
            # N2 = cp.Beta(1.5,2)-0.5
            N2 = cp.Normal(0,0.3)

            MU = cp.Uniform(0.,.0005)

        else:
            N1 = cp.Uniform(2.7,3.3)
            N2 = cp.Uniform(-0.9,0.9)
            # MU = cp.Uniform(0.,.05)
            MU = cp.Normal(0.1,.0001)

        # delta = cp.J(N1,N2,MU)
        delta = cp.J(N1,N2)
        
        tf = 10
        Mu = 0.5
        
        samples = delta.sample(200000,'L').T
        pdf = delta.pdf(samples.T)
        samples=np.append(samples,Mu*np.ones((samples.shape[0],1)),1)
                  
        self.monte_carlo(samples,tf,pdf)
        self.plot()
                
        # xy = np.mgrid[2.7:3.3:50j, -0.9:0.9:50j].reshape(2,-1).T
        # mu = Mu*np.ones((xy.shape[0],1))

        # grid = np.append(xy, mu, axis=1)
        # pdf_grid = delta.pdf(xy.T)

        # self.monte_carlo(grid,tf,pdf_grid)
        # self.plot()

        plt.show()    

if __name__ == '__main__':    
    vdp = VDP()
    vdp.test()