""" 
    Utilities for working with joint probability density functions over n-dimensional domains.
    
    Typical flow is 
    1. Convert monte carlo style data (i.e. scattered points in n-D space) and corresponding PDF values into a grid using "grid"
    2. Estimate marginals, expectations, etc. using the remaining methods. 


    Note: When using "grid" be careful not to use too many bins relative to the number of samples.
    
    TODO: Allow for marginal computation (by permuting the order of integration). Eventually extend to an order of marginal, or at least bivariate.
          Increase test example to 3D, then compare 2-d bivariate distributions in addition to marginals
"""

from numba import jit
import numpy as np

# @jit
def grid(data, density, bins=50):
    ''' 
    This method partitions the domain defined by data into equal intervals defined by bins.
    Then, the probability density in each partition is estimated as the average of the density of
    each point in the partition, normalized by the total probability mass.
        
        Inputs:
            data    - An (N,n) numpy array of points where each column is a different variable (x1,x2,...,xn). 
            density - An array-like with N elements
            bins    - An integer or list of integers with len == n (columns of data).
            
        Outputs:
            centers - A list of 1-d arrays representing the center of each hypercube in each dimension
            Pest    - The estimated density matrix
    '''
    
    if isinstance(bins,int):
        bins = [bins for _ in range(data.shape[1])]
    elif len(bins) != data.shape[1]:
        raise "Invalid input in grid. bins must be an integer or an array with length equal to n variables in data."
        
    edges = [np.linspace(row.min(),row.max(),nbins+1) for row,nbins in zip(data.T,bins)] # Iterate over the columns of the original matrix
    centers_pdf = [np.linspace(row.min(),row.max(),nbins) for row,nbins in zip(data.T,bins)] # This is better for estimate of joint pdf, worse for marginals
    centers = [xe[:-1] + np.diff(xe)/2. for xe in edges] # True center of each bin
    
    
    # Two matrices, one with the total density and the other with nPoints
    Ngrid = np.zeros(bins)
    Pgrid = np.zeros(bins)
    
    for ndpoint, p in zip(data,density):                                         # Iterate over samples
    
        grid_index = []
    
        for index, scalar in enumerate(ndpoint):                                 # Iterate over each element of an n-dimensional point 
            for i in range(bins[index]):                                         # Iterate over each bin
                if scalar >= edges[index][i] and scalar <= edges[index][i+1]:    # Find the bin in which it is located
                    grid_index.append(i)
                    break

        # Once we know the index of the point in each direction, add a point to the number in that grid space, and add the probability.    
        gi = tuple(grid_index)
        Ngrid[gi] += 1
        Pgrid[gi] += p
    
    Ngrid[Ngrid==0] = 1                  # This just removes division by zero. Does not change the results since probability in those points is also 0
    Pave = Pgrid/Ngrid                   # Average density in each grid
    
    Ptotal = integrate_pdf(centers_pdf,Pave) # Total density
    Pest = Pave/Ptotal

    return centers, Pest
   
# @jit
def integrate_pdf(grid_points, pdf, return_all=False):  
    ''' Given a grid of points and the probability density in each grid point, compute the integral over the entire domain. 
    
        grid should be a array-like with each element corresponding to the center of each element of the grid in that direction.
        The number of elements in grid should match the dimensionality of the pdf input.
        
        The matrix M is filled from highest dimension down to the scalar total probability.
        Thus, the second to last element will be a univariate marginal, the third to last element will be a bivariate marginal, etc.
    
    '''
    # from scipy.integrate import trapz
    from scipy.integrate import simps as trapz # just so I don't have to rewrite it
    from itertools import product
       
    n = len(grid_points)                           # Number of variables
    N = [len(a) for a in grid_points]              # Number of points in each dimension  
    M = [pdf] + allocate(n,N)                      # Storage of all intermediate arrays
    
    for i in range(1,n+1):
        dir = [range(d) for d in M[i].shape]
        
        # These can be converted to a comprehension at some point. No need to preallocate then.
        for ndpoint in product(*dir): # Loop over all the possible combinations of 1-D arrays
            M[i][ndpoint] = trapz(M[i-1][ndpoint],grid_points[-i])    
            
    if return_all:
        return M 
    else:
        return M[-1]

# @jit        
def marginal(grid_points, pdf, index=None):
    ''' Given n-dimensional data, compute the univariate marginal distributions corresponding the
        directions given by index. Set index to None to compute all n marginals.
    
    '''
    
    if index is not None:
        if isinstance(index, int):                                                           # Compute a single marginal
            grid_points_new, pdf_new = permute_data(grid_points[:], np.copy(pdf), index)
            M = integrate_pdf(grid_points_new, pdf_new, return_all=True)[-2]  
        else:                                                                                # Compute a list of marginals
            M = []
            for ind in index:
                grid_points_new, pdf_new = permute_data(grid_points[:], np.copy(pdf), ind)
                M.append(integrate_pdf(grid_points_new, pdf_new, return_all=True)[-2])  
    else:                                                                                    # Compute all the marginals
        M = []
        for index in range(len(grid_points)):
            grid_points_new, pdf_new = permute_data(grid_points[:], np.copy(pdf), index)
            M.append(integrate_pdf(grid_points_new, pdf_new, return_all=True)[-2])  
    return M

@jit    
def permute_data(grid_points, pdf, index):
    ''' Permutes the grid_points list and pdf ndarray such that the dimension specified by index becomes the first dimension. '''
    n = len(grid_points)
    
    if not isinstance(index,int):
        raise('permute_data can only move one index. Please specify an integer.')
    elif index < 0 or index > (len(grid_points)-1):
        raise('Invalid input for index in permute_data')
        
    if index == 0:
        return grid_points, pdf
    
    indices = range(n)
    indices.insert(0, indices.pop(index))
    
    grid_points.insert(0, grid_points.pop(index))
    pdf = np.transpose(pdf, indices)
    return grid_points, pdf
    
# @jit    
def allocate(n, N):
    ''' 
        Pre-allocates a list of tensors in decreasing dimension. 
        For use as storage of intermediate results in integrating
        a probability density function over n-dimensional domains.
        
        n is the dimensionality of the full state
        N is a list of length n with the number of points along that dimension
    '''
    storage_sizes = [tuple(N[:i]) for i in range(n-1,0,-1)]   # Assumes we integrate along the last dimension first. Stores each intermediate tensor.
    # loop_sizes = [tuple(N[i:]) for i in range(1,n)]          # Assumes we integrate along the first dimension first.
    tensor = [np.zeros(size) for size in storage_sizes] + [np.array(0)]   # Can easily be combined with the above line
    return tensor
    
    
def estimate_sparsity(pdf):
    ''' Divide zero elements by total number of elements '''
    return 1-float(np.count_nonzero(pdf))/pdf.size
    
def test_3d():
    import chaospy as cp
    import matplotlib.pyplot as plt
    
    N1 = - 0.5+cp.Beta(2,5)
    N2 = cp.Normal(0,0.3)
    MU = cp.Uniform(0.,.5)
    
    delta = cp.J(N1,N2,MU)
    
    N = 500000
    Nb = int(N**(1./3.3))
    samples = delta.sample(N,'S').T
    pdf = delta.pdf(samples.T)
        

    centers,p = grid(samples, pdf, bins=(Nb+1,Nb+3,Nb-1))
    X,Y = np.meshgrid(centers[0],centers[1])

    p_np, edges = np.histogramdd(samples,bins=(Nb+1,Nb+3,Nb-1),normed=True)
    
    print p.shape
    print p_np.shape
    print "Sparsity: {}".format(estimate_sparsity(p))
    e = np.abs(p-p_np)
    print e.max()
    
    M = integrate_pdf(centers, p, return_all=True)
    M2 = marginal(centers,p,1)
    Mmu = marginal(centers,p,2)
    
    # Truth for comparison
    x1_samples = sorted(N1.sample(1000,'S'))
    x1_marginal = N1.pdf(x1_samples)
    x2_samples = sorted(N2.sample(1000,'S'))
    x2_marginal = N2.pdf(x2_samples)
    mu_samples = sorted(MU.sample(1000,'S'))
    mu_marginal = MU.pdf(mu_samples)   
    
    plt.figure()
    plt.plot(centers[0],M[-2],'k',label='Estimated')
    plt.plot(x1_samples,x1_marginal,'--',label='Truth')
    plt.hist(samples[:,0],bins=200,normed=True,histtype='step',label='QMC' )
    plt.legend(loc='best')
    
    plt.figure()
    plt.plot(centers[1],M2,'k',label='Estimated')
    plt.plot(x2_samples,x2_marginal,'--',label='Truth')
    plt.hist(samples[:,1],bins=200,normed=True,histtype='step',label='QMC' )
    plt.legend(loc='best')
    
    plt.figure()
    plt.plot(centers[2],Mmu,'k',label='Estimated')
    plt.plot(mu_samples,mu_marginal,'--',label='Truth')
    plt.hist(samples[:,2],bins=200,normed=True,histtype='step',label='QMC' )
    plt.legend(loc='best')    
    
    
    plt.show()
    
def test_grid_resolution():
    ''' Uses a simple 2-d example with various grid sizes for a large data set '''
    import chaospy as cp
    import matplotlib.pyplot as plt
    
    N1 = - 0.5+cp.Beta(2,5)
    N2 = cp.Normal(0,0.3)
    
    delta = cp.J(N1,N2)
    
    N = 100000
    # Nb = int(N**(1./3.5))
    samples = delta.sample(N,'S').T
    pdf = delta.pdf(samples.T)
        
        
    for Nb in [4,14,24]:
        print Nb
        centers,p = grid(samples, pdf, bins=(Nb,Nb))
        X,Y = np.meshgrid(centers[0], centers[1])
        print "Sparsity: {}".format(estimate_sparsity(p))

        plt.figure()
        plt.contourf(X,Y,p.T)
        plt.hlines(centers[1],centers[0].min(),centers[0].max())
        plt.vlines(centers[0],centers[1].min(),centers[1].max())
        plt.title('Grid-based estimate of PF results ({} partitions per dimension)'.format(Nb))
        plt.colorbar()
        
    plt.figure()
    plt.scatter(samples[:,0],samples[:,1],20,pdf)
    plt.title('Truth')
    plt.colorbar()
    
    plt.show()
    
    
if __name__ == '__main__':    
    test_3d()
    # test_grid_resolution()