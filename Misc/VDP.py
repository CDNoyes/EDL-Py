import numpy as np
import chaospy as cp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from EntryGuidance.PDF import grid, marginal


class VDP(object):
    ''' A van der pol oscillator class '''
    
    def __init__(self):
        print "Initialized"
        
    def __dynamics__(self, x, t, mu):
        return np.array([x[1],-x[0] + mu*(1-x[0]**2)*x[1], self.__pf__(x,mu)])
        
    def __jacobian__(self, x, mu):    
        # Experimentally, this jacobian includes the elements of the PF operator to estimate its sensitivity
        return np.array([[0, 1, 0],[-1-2*x[0]*x[1]*mu, mu*(1-x[0]**2), 0], [2*mu*x[0], 0, -mu*(1-x[0]**2)]])
        
    def __pf__(self, x, mu):
        """ Perron-Frobenius operator dynamics """
        return -mu*(1-x[0]**2)*x[2]
            
    def __stm__(self, stm, t, x, mu):   
        """ State-transition matrix dynamics """
        stm.shape = (3,3)
        A = self.__jacobian__(x(t), mu)
        
        return np.dot(A,stm).flatten()
        
    def simulate(self, sample, tf, p, return_stm=False):
        ''' Integrate a single sample. '''
        t = np.linspace(0,tf,61)
        x = odeint(self.__dynamics__, [sample[0],sample[1],p], t, args=(sample[2],))
        stm = self.integrate_stm(t, x, sample[2])
        if return_stm:
            return x, stm
        else:
            return x

    def integrate_stm(self, t, x, mu):
        ''' Integrate the STM along a trajectory to determine linear sensitivity. '''
        stm0 = np.eye(3).flatten()
        X = interp1d(t, x, kind='cubic', axis=0, bounds_error=False, assume_sorted=True,fill_value=(x[0],x[-1]))
        stm_vec = odeint(self.__stm__, stm0, t, args=(X,mu))
        
        return np.array([np.reshape(stm,(3,3)) for stm in stm_vec])
        
    def monte_carlo(self, samples, tf, pdf):
        ''' Performs a Monte Carlo. Samples is an (N,3) array-like of deltas [x1,x2,mu] '''
        
        X = [self.simulate(sample,tf,p0,True) for sample,p0 in zip(samples,pdf)]
        
        self.outputs = np.array([x[0] for x in X])  # State trajectories
        self.stms = np.array([x[1] for x in X])     # Sensitivities along the state trajectories
        self.samples = samples
        self.pdf = pdf
        
    def sample_stm(self, radius):
        """ Radius is a scalar or a tuple with the length of the of n-dimensional ball around each point """
        delta = np.array([[x[0]-3,x[1]-3,p-pdf0] for x,p in zip(samples,pdf)])    
        xnew = np.array([xf + np.dot(phi,d) for d in delta])
        
        
    def plot(self,fignum=0):
        ''' Visualizations of the monte carlo results. '''
        
        cm = 'YlOrRd'
        
        if 1:
            for traj in self.outputs[::10]:
                plt.figure(3+fignum)
                plt.plot(traj[:,0],traj[:,1],'k',alpha=0.01)

            plt.figure(3+fignum)
            for i in [0,-1]:
                # plt.scatter(self.outputs[:,i,0],self.outputs[:,i,1],20,self.pdf)
                plt.scatter(self.outputs[:,i,0],self.outputs[:,i,1],20,self.outputs[:,i,2])
            
            
        plt.figure(1+fignum)
        counts,xe,ye,im = plt.hist2d(self.outputs[:,-1,0], self.outputs[:,-1,1], normed=True, bins=(250,250),cmap=cm)
        plt.title('MC')    
        plt.colorbar()    
            
        vmax = np.max((counts.max(),self.outputs[:,-1,2].max()))    
        
        # Nb = int(self.outputs.shape[0]**(1./2.5))
        # Nb = 45
        # centers,p = grid(self.outputs[:,-1,0:2], self.outputs[:,-1,2], bins=(Nb,Nb))
        # X,Y = np.meshgrid(centers[0],centers[1])

        # plt.figure(2+fignum)
        # plt.contourf(X,Y,p.T,cmap=cm)
        # plt.hlines(centers[1],centers[0].min(),centers[0].max())
        # plt.vlines(centers[0],centers[1].min(),centers[1].max())
        # plt.title('Grid-based estimate of PF results ({} partitions per dimension)'.format(Nb))
        # plt.colorbar()

            
        
    def test(self):
        ''' 
            Runs the VDP tests:
                Monte Carlo 
                Perron-Frobenius Operator
            
        '''
        if 1:
            N1 = cp.Normal(3,.1)
            N2 = cp.Normal(3,.2)
        elif 0:
            N1 = - 0.5+cp.Beta(2,5)
            # N2 = cp.Beta(1.5,2)-0.5
            N2 = cp.Normal(3,0.1)

        else:
            N1 = cp.Uniform(2.7,3.3)
            N2 = cp.Uniform(-0.9,0.9)

        delta = cp.J(N1,N2)
        
        tf = 2
        Mu = .2
        
        samples = delta.sample(500,'S').T
        pdf = delta.pdf(samples.T)
        samples=np.append(samples,Mu*np.ones((samples.shape[0],1)),1)
                  
        self.monte_carlo(samples,tf,pdf)
        
                        # samples = delta.sample(1000,'L').T
                        # pdf = delta.pdf(samples.T)
                        # samples=np.append(samples,Mu*np.ones((samples.shape[0],1)),1)
                                  
                        # self.monte_carlo(samples,tf,pdf)
                        # self.plot(3)
                        
                        # samples = delta.sample(50000,'L').T
                        # pdf = delta.pdf(samples.T)
                        # samples=np.append(samples,Mu*np.ones((samples.shape[0],1)),1)
                                  
                        # self.monte_carlo(samples,tf,pdf)
                        # self.plot(6)
                        
                        # xy = np.mgrid[2.7:3.3:50j, -0.9:0.9:50j].reshape(2,-1).T
                        # mu = Mu*np.ones((xy.shape[0],1))

                        # grid = np.append(xy, mu, axis=1)
                        # pdf_grid = delta.pdf(xy.T)

                        # self.monte_carlo(grid,tf,pdf_grid)
                        # self.plot()
        
        # ##### Compare linear predictions using STM against true points
        pdf0 = delta.pdf(np.array([[3],[3]]))[0]
        x, stm = vdp.simulate([3,3,Mu], tf, pdf0,True)
        # print stm
        plt.figure()
        plt.plot(x[:,0],x[:,1])
        xf = x[-1]
        phi = stm[-1]
        
        delta = np.array([[x[0]-3,x[1]-3,p-pdf0] for x,p in zip(samples,pdf)])    
        xnew = np.array([xf + np.dot(phi,d) for d in delta])
        plt.scatter(xnew[:,0],xnew[:,1],20,xnew[:,2])
        plt.colorbar()    
        self.plot(1)

        plt.show()    

if __name__ == '__main__':    
    vdp = VDP()
    vdp.test()
    # t = np.linspace(0,5,61)
    # x,stm = vdp.simulate([3,3,.2], 5, 0.01,True)
    # plt.figure()
    # plt.plot(x[:,0],x[:,1])
    # plt.figure()
    # plt.plot(t, stm[:,0,0],label='x1')
    # plt.plot(t, stm[:,1,1],label='x2')
    # plt.plot(t, stm[:,2,0],label='dp/dx1(0)')
    # plt.plot(t, stm[:,2,1],label='dp/dx2(0)')
    # plt.show()