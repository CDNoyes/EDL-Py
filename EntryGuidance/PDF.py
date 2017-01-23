""" 
    Utilities for working with joint probability density functions over n-dimensional domains.
    
    Typical flow is 
    1. Convert monte carlo style data (i.e. scattered points in n-D space) and corresponding PDF values into a grid using "grid"
    2. Estimate marginals, expectations, etc. using the remaining methods. 


    Note: When using "grid" be careful not to use too many bins relative to the number of samples.
"""

import numpy as np

def grid(data, density, bins=50):
    ''' 
    This method partitions the domain defined by data into equal intervals defined by bins.
    Then, the probability density in each partition is estimated as the average of the density of
    each point in the partition.
        
        Inputs:
            data - An (N,n) numpy array of points where each column is a different variable (x1,x2,...,xn). 
            density - An array-like with N elements
            bins - an integer or list of integers with len == n (columns of data).
            
        Outputs:
            centers - A list of 1-d arrays representing the center of each hypercube in each dimension
            Pest - The estimated density matrix
    '''
    
    if isinstance(bins,int):
        bins = [bins for _ in range(data.shape[1])]
        
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
   

def integrate_pdf(grid_points, pdf, return_all=False):  
    ''' Given a grid of points and the probability density in each grid point, compute the integral over the entire domain. 
    
        grid should be a array-like with each element corresponding to the center of each element of the grid in that direction.
        The number of elements in grid should match the dimensionality of the pdf input.
        
        The matrix M is filled from highest dimension down to the scalar total probability.
        Thus, the second to last element will be a univariate marginal, the third to last element will be a bivariate marginal, etc.
    
    '''
    from scipy.integrate import trapz
    from itertools import product
       
    n = len(grid_points)                           # Number of variables
    N = [len(a) for a in grid_points]              # Number of points in each dimension  
    M = [pdf] + allocate(n,N)               # Storage of all intermediate arrays
    
    for i in range(1,n+1):
        # print "Dimension: {}".format(i)
        dir = [range(d) for d in M[i].shape]
        
        # These can be converted to a comprehension at some point. No need to preallocate then.
        for ndpoint in product(*dir): # Loop over all the possible combinations of 1-D arrays
            # print "Point: {}".format(ndpoint)
            M[i][ndpoint] = trapz(M[i-1][ndpoint],grid_points[-i])    
            
    if return_all:
        return M 
    else:
        return M[-1]
    
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
    
    
def test():
    import chaospy as cp
    import matplotlib.pyplot as plt
    
    N1 = - 0.5+cp.Beta(2,5)
    N2 = cp.Normal(0,0.3)
    MU = cp.Uniform(0.,.0005)
    
    delta = cp.J(N1,N2)
    
    samples = delta.sample(200,'S').T
    pdf = delta.pdf(samples.T)
        

    centers,p = grid(samples, pdf, bins=(10,10))
    X,Y = np.meshgrid(centers[0],centers[1])

    plt.figure()
    plt.contourf(X,Y,p.T)
    plt.title('Grid-based estimate of PF results')
    plt.colorbar()
    
    plt.figure()
    plt.scatter(samples[:,0],samples[:,1],20,pdf)
    plt.title('Truth')
    plt.colorbar()
    
    M = integrate_pdf(centers, p, return_all=True)
    x1_samples = N1.sample(100,'S')
    x1_marginal = N1.pdf(x1_samples)
    plt.figure()
    plt.plot(centers[0],M[1],'k',label='Estimated')
    plt.plot(x1_samples,x1_marginal,'o',label='Truth')
    plt.legend(loc='best')
    plt.show()
    
if __name__ == '__main__':    
    test()