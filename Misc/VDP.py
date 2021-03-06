import numpy as np
import chaospy as cp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# from scipy.integrate import odeint
from scipy.interpolate import interp1d
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from EntryGuidance.PDF import grid, marginal
from Utils.RK4 import RK4 as odeint # About twice as slow as odeint but capable of vectorization
from Utils.boxgrid import boxgrid

import time


class VDP(object):
    ''' A van der pol oscillator class.
            This serves as a simple 2D testbed for research in dynamic systems.
            The system is linear when the parameter mu is 0.
            
            The system dynamics have been augmented with the Perron-Frobenius operator for uncertainty propagation.
            An analytical expression for the system jacobian is available. 
            First order sensitivities are available in both forward (i.e. STM) and reverse (i.e. adjoint) methodologies.
            Simulations are vectorized and thus very fast, but sensitivities are computed one by one (for now). 

            TODO: Use adjoint method on arbitrary function of final states and try to recover the same info using STM methods
                  Vectorize STM/Adjoint integrations
    '''
    def __init__(self):
        self.stm_inputs = None
        self.stm_outputs = None
        self.extended_outputs = None
        self.outputs = None
        self.stms = None
        self.samples = None
        self.pdf = None
        
    def __dynamics__(self, x, t, mu):
        """ System dynamics, including PF operator """
        return np.array([x[1],-x[0] + mu*(1-x[0]**2)*x[1], self.__pf__(x,mu)])
        
        
    def __jacobian__(self, x, mu):    
        """ Jacobian of the system dynamics. Experimentally, this includes the elements of the PF operator to estimate its sensitivity as well. """
        # print "In __jacobian__"
        # print x[0].shape
        # Attempt to vectorize:
        return np.array([[np.zeros_like(x[0]), np.ones_like(x[0]), np.zeros_like(x[0])],[-1-2*x[0]*x[1]*mu, mu*(1-x[0]**2), np.zeros_like(x[0])], [2*mu*x[0], np.zeros_like(x[0]), -mu*(1-x[0]**2)]])
        
        
    def __pf__(self, x, mu):
        """ Perron-Frobenius operator dynamics """
        return -mu*(1-x[0]**2)*x[2]
            
            
    def __stm__(self, stm, t, x, mu):   
        """ State-transition matrix dynamics for forward sensitivity analysis """
        N = stm.shape[1]
        stm.shape = (3,3,N)
        A = self.__jacobian__(x(t), mu)

        dstm = np.array([np.dot(a, s) for a,s in zip(np.rollaxis(A,2), np.rollaxis(stm,2))])
        dstm = np.transpose(dstm, axes=(1,2,0))
        return np.reshape(dstm, (9,N) )
        
        
    def __adjoint__(self, costate, t, x, mu):
        """ Adjoint dynamics for reverse sensitivity analysis """
        A = self.__jacobian__(x(t), mu)
        A = np.transpose(A, axes=(2,1,0))
        print "In __adjoint__"

        print A.shape
        print costate.shape
        
        # return np.dot(-A.T,costate)
        return np.dot(-A, costate)
        
        
    def simulate(self, sample, tf, p, return_stm=False):
        ''' Integrate a single sample. '''
        t = np.linspace(0,tf,201)
        x = odeint(self.__dynamics__, [sample[0],sample[1],p], t, args=(sample[2],))
        X = interp1d(t, x, kind='cubic', axis=0, bounds_error=False, assume_sorted=True,fill_value=(x[0],x[-1]))
        print "In simulate"
       
        if return_stm:
            # t0 = time.time()
            # l = self.integrate_adjoint(t, x, sample[2])    # Reverse sensitivity, dx(tf)/dx(t)
            # t_adj = time.time()-t0
            
            # plt.figure()
            # plt.plot(t, l[:,0,0], label='adjoint')
            # plt.plot(t, l[:,0,1])
            # plt.plot(t, l[:,1,0])
            # plt.plot(t, l[:,1,1])
            
            t0 = time.time()
            stm = self.integrate_stm(t, x, sample[2])         # Forward sensitivity, dx(t)/dx(t0), at t=tf matches (adjoint evaluated at t=t0)
            t_stm = time.time()-t0
            
            # t0 = time.time()
            # lfor = self.compose_stm(stm)   # Adjoint sensitivity from stm matrices
            # t_com = time.time()-t0
            
            # plt.plot(t, lfor[:,0,0], '--', label='recovered from STM')
            # plt.plot(t, lfor[:,0,1], '--')
            # plt.plot(t, lfor[:,1,0], '--')
            # plt.plot(t, lfor[:,1,1], '--')
            # plt.title('dx(tf)/dx(t)')
            # plt.xlabel('Time (s)')
            # plt.ylabel('Sensitivity')
            # plt.legend(loc='best')
            
            # print "Adjoint integration time:  {} s".format(t_adj)
            print "STM integration time:      {} s".format(t_stm)
            # print "STM composition time:      {} s".format(t_com)
            # print "Sum:                       {} s".format(t_stm+t_com)
            
            # plt.figure()
            # plt.plot(t, stm[:,0,0], '-o')
            # plt.plot(t, stm[:,0,1], '-o')
            # plt.plot(t, stm[:,1,0], '-o')
            # plt.plot(t, stm[:,1,1], '-o')
            
            # plt.title('dx(t)/dx(t0)')
            # plt.xlabel('Time (s)')
            # plt.ylabel('Sensitivity')
            # plt.show()
            # print l[0] 
            # print stm[-1]
            return x, stm
        else:
            return x

            
    def integrate_stm(self, t, x, mu):
        ''' Integrate the STM along a trajectory to determine linear sensitivity. '''
        stm0 = np.eye(3).flatten()
        stm0 = np.tile(stm0, (x.shape[2],1)).T
        X = interp1d(t, x, kind='cubic', axis=0, bounds_error=False, assume_sorted=True,fill_value=(x[0],x[-1]))

        stm_vec = odeint(self.__stm__, stm0, t, args=(X,mu))
        
        # return np.array([np.reshape(stm,(3,3)) for stm in stm_vec])
        return np.reshape(stm_vec,(stm_vec.shape[0], 3,3,x.shape[2]))

        
    def compose_stm(self, stms):
        """ From dx(t)/dx(t0) get all dx(tf)/dx(t) """
        
        stmi = [np.dot(stms[-1], np.linalg.inv(stm)) for stm in stms] # These are equivalent, not sure which is faster
        # stmf = stms[-1].T
        # stmi = [np.linalg.solve(stm, stmf).T for stm in np.transpose(stms,axes=(0,2,1))]
        return np.array(stmi)
    
    
    def integrate_adjoint(self,t,x,mu):
        """ Integrate the adjoint backward along a trajectory to determine sensitivity """
        
        nsteps,nstate,nsamples = x.shape
        I = np.eye(3) # Partial of each final state to the final state vector
        # print I.shape
        I.shape = (3,3,1)
        I = np.tile(I, (1, 1, nsamples))
        # print I.shape
        # I = np.transpose(I, axes=(2,0,1))
        # print I.shape
        # print I[0].shape
        
        X = interp1d(t, x, kind='cubic', axis=0, bounds_error=False, assume_sorted=True,fill_value=(x[0],x[-1]))
        
        costates = [odeint(self.__adjoint__, x0, t[::-1], args=(X,mu))[::-1] for x0 in I]
        return np.transpose(np.array(costates), axes=(1,0,2))
    
    
    def forward_adjoint(self, t, x, mu, costates):
    
        X = interp1d(t, x, kind='cubic', axis=0, bounds_error=False, assume_sorted=True,fill_value=(x[0],x[-1]))
        cx = [odeint(self.__fadjoint__, x0, t, args=(X,mu)) for x0 in costates]
        
        return np.transpose(np.array(cx), axes=(1,0,2))
        
    def __fadjoint__(self, costate, t, x, mu):
        """ Forward adjoint dynamics for reverse sensitivity analysis starting from STM """
        A = self.__jacobian__(x(t), mu)
        return np.dot(A.T,costate)
    
    
    def monte_carlo(self, samples, tf, pdf, return_stm=False):
        ''' Performs a Monte Carlo. Samples is an (N,3) array-like of deltas [x1,x2,mu] '''
        
        # X = [self.simulate(sample,tf,p0,False) for sample,p0 in zip(samples,pdf)]
        if return_stm:
            X, STM = self.simulate(samples.T, tf, pdf, return_stm) # Vectorized version, seems to be ~30x faster
            STM = np.transpose(STM, axes=(3,0,1,2))
            print STM.shape
        else:
            X = self.simulate(samples.T, tf, pdf, return_stm) # Vectorized version, seems to be ~30x faster
            STM = None
        X = np.transpose(X,axes=(2,0,1))
        print X.shape
        self.outputs = np.array(X) #np.array([x[0] for x in X])  # State trajectories
        self.stms = STM #np.array([x[1] for x in X])     # Sensitivities along the state trajectories
        self.samples = samples
        self.pdf = pdf
    
    def plot_ctrb(self, mu, fignum=0):
        from scipy.linalg import expm, inv
        
        A = self.__jacobian__([0,0], mu)[0:2,0:2]
        b = [0,1]
        
        dR = np.array([ np.dot((-2*expm(-A*t) + np.eye(2)), np.dot(inv(A),b)) for t in np.linspace(0,10)])
        
        plt.figure(fignum)
        plt.plot(dR[:,0],dR[:,1],'k--')
        plt.plot(-dR[:,0],-dR[:,1],'k--')
        # plt.show()
        
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
        N = self.outputs.shape[0]
        Nmax = 5000.0
        skip = int(np.ceil(N/Nmax))
        if 1:
            print "In plot"
            print self.outputs.shape
            for traj in self.outputs[::skip]:
                plt.figure(3+fignum)
                plt.plot(traj[:,0],traj[:,1],'k',alpha=0.1)
            # plt.axis([-10,10,-10,10])
            # plt.figure(3+fignum)
            # for i in [0,-1]:
                # plt.scatter(self.outputs[:,i,0],self.outputs[:,i,1],20,self.pdf)
                # plt.scatter(self.outputs[:,i,0],self.outputs[:,i,1],20,self.outputs[:,i,2])
            
            # plt.colorbar()
            
        if self.stm_outputs is not None:    
            
            plt.figure(4)    
            plt.scatter(self.stm_outputs[:,0],self.stm_outputs[:,1],10,self.stm_outputs[:,2])
            plt.title('New final states')
            plt.colorbar()
            
            plt.figure(11)
            plt.scatter(self.stm_inputs[:,0],self.stm_inputs[:,1],10)
            plt.title('New initial states')
            
        # plt.figure(1+fignum)
        # counts,xe,ye,im = plt.hist2d(self.outputs[:,-1,0], self.outputs[:,-1,1], normed=True, bins=(250,250), cmap=cm)
        # plt.title('MC')    
        # plt.colorbar()    
        
        # plt.figure(3+fignum)
        # plt.scatter(self.extended_outputs[:,0], self.extended_outputs[:,1], 10, 'k',alpha=0.1)
        # plt.title('MC + STM approximations')    
        # plt.colorbar()   
        
        # vmax = np.max((counts.max(),self.outputs[:,-1,2].max()))    
        
        # Nb = int(self.outputs.shape[0]**(1./2.5))
        # Nb = 20
        # centers,p = grid(self.outputs[:,-1,0:2], self.outputs[:,-1,2], bins=Nb) # Just actual samples
        # centers,p = grid(self.extended_outputs[:,0:2], self.extended_outputs[:,2], bins=(Nb,Nb)) # Augmented samples
        # X,Y = np.meshgrid(centers[0],centers[1])

        # plt.figure(2+fignum)
        # plt.contourf(X,Y,p.T,cmap=cm)
        # plt.hlines(centers[1],centers[0].min(),centers[0].max())
        # plt.vlines(centers[0],centers[1].min(),centers[1].max())
        # plt.title('Grid-based estimate of PF results ({} partitions per dimension)'.format(Nb))
        # plt.colorbar()
        
        # Nb=40
        # centers,p = grid(self.extended_outputs[:,0:2], self.extended_outputs[:,2], bins=(Nb,Nb)) # Augmented samples
        # X,Y = np.meshgrid(centers[0],centers[1])

        # plt.figure(5+fignum)
        # plt.contourf(X,Y,p.T,cmap=cm)
        # plt.hlines(centers[1],centers[0].min(),centers[0].max())
        # plt.vlines(centers[0],centers[1].min(),centers[1].max())
        # plt.title('Grid-based estimate of PF results with STM augment ({} partitions per dimension)'.format(Nb))
        # plt.colorbar()

            
        
    def test(self):
        ''' 
            Runs the VDP tests:
                Monte Carlo 
                Perron-Frobenius Operator
            
        '''
        if 1:
            N1 = cp.Normal(3,.1)
            N2 = cp.Normal(3,.1)
            # MU = cp.Normal(0.35,.09)
            # MU = cp.Uniform(0,-1)
            # MU = cp.Uniform(0,1)
        elif 0:
            N1 = - 0.5+cp.Beta(2,5)
            # N2 = cp.Beta(1.5,2)-0.5
            N2 = cp.Normal(3,0.1)

        else:
            N1 = cp.Uniform(2.7,3.3)
            N2 = cp.Uniform(-0.9,0.9)

        delta = cp.J(N1,N2)
        
        tf = 12
        Mu = 2
        
        samples = delta.sample(2000,'S').T
        # samples = boxgrid(((2.7,3.3),(2.4,3.6)), N=225, interior=True)
        pdf = delta.pdf(samples.T)/1e6
        samples=np.append(samples,Mu*np.ones((samples.shape[0],1)),1)
        
        t0 = time.time()
        self.monte_carlo(samples,tf,pdf,False)
        t1 = time.time()
        
        # self.sample_stm((0.1,0.1),50,delta)
        t2 = time.time()
        print "MC time:           {} s".format(t1-t0)
        print "STM sampling time: {} s".format(t2-t1)
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
        

if __name__ == '__main__':    
    vdp = VDP()
    # vdp.plot_ctrb(2, 4)
    vdp.test()
    # sample = [3, 3, .1]
    # states, stms = vdp.simulate(sample, 10, 0, return_stm=True)
    # print stms[-1]
    