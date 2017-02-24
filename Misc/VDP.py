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
import time

from itertools import product 

def box_grid(bounds, N, interior=False):
    """ n-dimensional box grid, either just the exterior, or with interior points as well
    bounds is an n-length list/tuple with each element being the (min,max) along that dimension
    N is the number of samples per dimension, either a scalar or N-length list of integers. """
    try:
        N[0]
    except:
        N = [N for _ in bounds]
    n = len(bounds)    
    vectors = [np.linspace(b[0],b[1],ni) for b,ni in zip(bounds,N)] # Each one of these will be used 2**(n-1) times
    # nTotal = np.product(N)
    grid_points = []

    if interior:
        bounds = vectors

    for dim in range(n):
        reduced_bounds = list(bounds[:])
        reduced_bounds.pop(dim)
        new_points = np.zeros((N[dim],n)) # Preallocate
        dim_iter = range(n)
            
        for corner in product(*reduced_bounds):
            for dim_ in dim_iter:
                if dim_ < dim:
                    new_points[:,dim_] = np.tile(corner[dim_],(N[dim]))
                elif dim_ > dim:    
                    new_points[:,dim_] = np.tile(corner[dim_-1],(N[dim]))
                else:
                    new_points[:,dim_] = vectors[dim]

            grid_points.append(np.copy(new_points))
    return np.vstack(grid_points)

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
        
    def sample_stm(self, radius, N, dist):
        """ Radius is a scalar or a tuple with the length of the of n-dimensional ball around each point
            N is the number of additional samples to add around each existing sample point. Thus the total number of points MC points will be E*(N+1) where E is the original MC size.
            For accuracy using linear approximations, the delta in initial conditions needs to be small, not the delta around the final states.
            This is an issue -> Just check the IC after you generate them.
        """
        n = self.outputs[0].shape[1]-1 # state dimension without PF operator
        nMC = self.outputs.shape[0]
        try:
            radius[0]
        except:
            radius = [radius for _ in range(n)] # Turn scalar into iterable
        
        normals = [cp.Normal(0,r/3.) for r in radius]
        # normals = [cp.Uniform(-r,r) for r in radius]
        local_dist = cp.J(*normals) # Joint distribution for sampling the deltas around each sample point
        # local_dist = cp.J(normals[0],cp.Uniform(-radius[1],radius[1])) # Joint distribution for sampling the deltas around each sample point
        
        x0_new = []
        p0_new = []
        xf_new = []
        pf_new = []
        delta_x0 = local_dist.sample(N,'S') 

        # pf_min = np.min(self.outputs[:,-1,-1])
        pf_min = 0
        for traj,stm_traj in zip(self.outputs, self.stms):
            x0 = traj[0,:-1]                                                   # All initial states except the probability density state
            xf = traj[-1,:-1]                                                  # All final states except the probability density state
            p0 = traj[0,-1]                                                    # The initial probability density state
            pf = traj[-1,-1]                                                   # The final probability density state
            stmf = stm_traj[-1]

            # Sample x0 and propagate them instead of the reverse. Seems to work much better
            x0.shape = (x0.shape[0],1)
            xf.shape = (xf.shape[0],1)
            x0_new.append(x0 + delta_x0)
            p0_new.append(dist.pdf(x0_new[-1]))
            delta_p0 = p0_new[-1]-p0
            xf_new.append(xf + np.dot(stmf[0:n,0:n],delta_x0))
            pf_new.append(pf + np.dot(stmf[-1,:],np.vstack((delta_x0,delta_p0))))


        x0_new = np.hstack(x0_new).T
        xf_new = np.hstack(xf_new).T
        pf_new = np.reshape(np.array(pf_new),(nMC*N,1))
        keep = (pf_new>pf_min)[:,0]
        print "{}% of STM samples removed. ".format((1-np.sum(keep)/float(keep.shape[0]))*100)
        self.stm_inputs = x0_new
        self.stm_outputs = np.hstack((xf_new[keep,:],pf_new[keep])) # Now we can use these as if they were actual MC data    
        self.extended_outputs = np.vstack((self.outputs[:,-1,:],self.stm_outputs)) # Now we can use these as if they were actual MC data    
        
        
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
            
            plt.colorbar()
            
        plt.figure(10)    
        plt.scatter(self.stm_outputs[:,0],self.stm_outputs[:,1],10,self.stm_outputs[:,2])
        plt.title('New final states')
        plt.colorbar()
        
        plt.figure(11)
        plt.scatter(self.stm_inputs[:,0],self.stm_inputs[:,1],10)
        plt.title('New initial states')
            
        plt.figure(1+fignum)
        counts,xe,ye,im = plt.hist2d(self.outputs[:,-1,0], self.outputs[:,-1,1], normed=True, bins=(250,250), cmap=cm)
        plt.title('MC')    
        plt.colorbar()    
        
        # plt.figure(3+fignum)
        # plt.scatter(self.extended_outputs[:,0], self.extended_outputs[:,1], 10, 'k',alpha=0.1)
        # plt.title('MC + STM approximations')    
        # plt.colorbar()   
        
        # vmax = np.max((counts.max(),self.outputs[:,-1,2].max()))    
        
        # Nb = int(self.outputs.shape[0]**(1./2.5))
        Nb = 30
        centers,p = grid(self.outputs[:,-1,0:2], self.outputs[:,-1,2], bins=(Nb,Nb)) # Just actual samples
        # centers,p = grid(self.extended_outputs[:,0:2], self.extended_outputs[:,2], bins=(Nb,Nb)) # Augmented samples
        X,Y = np.meshgrid(centers[0],centers[1])

        plt.figure(2+fignum)
        plt.contourf(X,Y,p.T,cmap=cm)
        plt.hlines(centers[1],centers[0].min(),centers[0].max())
        plt.vlines(centers[0],centers[1].min(),centers[1].max())
        plt.title('Grid-based estimate of PF results ({} partitions per dimension)'.format(Nb))
        plt.colorbar()
        
        Nb=40
        centers,p = grid(self.extended_outputs[:,0:2], self.extended_outputs[:,2], bins=(Nb,Nb)) # Augmented samples
        X,Y = np.meshgrid(centers[0],centers[1])

        plt.figure(5+fignum)
        plt.contourf(X,Y,p.T,cmap=cm)
        plt.hlines(centers[1],centers[0].min(),centers[0].max())
        plt.vlines(centers[0],centers[1].min(),centers[1].max())
        plt.title('Grid-based estimate of PF results with STM augment ({} partitions per dimension)'.format(Nb))
        plt.colorbar()

            
        
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
        
        tf = 0.5
        Mu = 1
        
        samples = delta.sample(20000,'L').T
        # samples = box_grid(((2.7,3.3),(2.4,3.6)), N=50, interior=True)
        pdf = delta.pdf(samples.T)
        samples=np.append(samples,Mu*np.ones((samples.shape[0],1)),1)
        
        t0 = time.time()
        self.monte_carlo(samples,tf,pdf)
        t1 = time.time()
        self.sample_stm((0.02,0.04),5,delta)
        t2 = time.time()
        print "MC time: {} s".format(t1-t0)
        print "STM time: {} s".format(t2-t1)
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
        # pdf0 = delta.pdf(np.array([[3],[3]]))[0]
        # x, stm = vdp.simulate([3,3,Mu], tf, pdf0,True)

        # plt.figure()
        # plt.plot(x[:,0],x[:,1])
        # xf = x[-1]
        # phi = stm[-1]
        
        # delta = np.array([[x[0]-3,x[1]-3,p-pdf0] for x,p in zip(samples,pdf)])    
        # xnew = np.array([xf + np.dot(phi,d) for d in delta])
        # plt.scatter(xnew[:,0],xnew[:,1],20,xnew[:,2])
        # plt.colorbar()    
        self.plot(1)

        plt.show()    
        
def test_box_grid():
    
    # 2-d example
    
    pts = box_grid(((-3,3),(-1,1)), (20,10),interior=True)
    plt.figure()
    plt.plot(pts[:,0],pts[:,1],'o')
    plt.show()


if __name__ == '__main__':    
    vdp = VDP()
    vdp.test()
    # test_box_grid()