from itertools import product 
import numpy as np

def boxgrid(bounds, N, interior=False):
    """ n-dimensional box grid, either just the exterior, or with interior points as well
    bounds is an n-length list/tuple with each element being the (min,max) along that dimension
    N is the number of samples per dimension, either a scalar or N-length list of integers. """
    try:
        N[0]
    except:
        N = [N for _ in bounds]
        
    n = len(bounds)    
    vectors = [np.linspace(b[0],b[1],ni) for b,ni in zip(bounds,N)]

    grid_points = np.array(list(product(*vectors)))
    
    if not interior:                # Remove the interior points
        bmin = np.array(bounds)[:,0]
        bmax = np.array(bounds)[:,1]
        ikeep = []
        for i,pt in enumerate(grid_points):
            if np.any( np.asarray(pt)-bmin == 0) or np.any( np.asarray(pt)-bmax == 0):
                ikeep.append(i)
        grid_points = grid_points[ikeep]        
    return grid_points