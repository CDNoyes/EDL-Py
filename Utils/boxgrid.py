from itertools import product
import numpy as np


def boxgrid(bounds, N, interior=False):
    """ n-dimensional box grid, either just the exterior, or with interior points as well.
    Inputs:
        bounds is an n-length list/tuple with each element being the (min,max) along that dimension
        N is the number of samples per dimension, either a scalar or n-length list of integers

    Setting N=2 and interior=False will return just the vertices of the hyper-rectangle.

    """
    if isinstance(N, int):
        N = [N for _ in bounds]


    n = len(bounds)
    vectors = [np.linspace(b[0],b[1],ni) for b,ni in zip(bounds,N)]

    grid_points = np.array(list(product(*vectors)))
    if not interior:                # Remove the interior points
        bmin = np.array(bounds)[:,0]
        bmax = np.array(bounds)[:,1]
        ikeep = []
        for i,pt in enumerate(grid_points):
            # In n-dimensions, >=(n-1) points must be extremal to constitute an edge
            if np.count_nonzero(np.logical_or(pt-bmin == 0, pt-bmax == 0)) >= (len(bounds)-1): # Count is faster than sum for some reason
                ikeep.append(i)
        grid_points = grid_points[ikeep]
    return grid_points


def test_boxgrid():

    # 2-d example
    import matplotlib.pyplot as plt
    import time

    pts = boxgrid(((-3,3),(-1,1)), (7,4),interior=True)
    plt.figure()
    plt.plot(pts[:,0],pts[:,1],'o')

    pts = boxgrid(((-3,3),(-1,1)), 5, interior=False)
    plt.figure()
    plt.plot(pts[:,0],pts[:,1],'o')

    # Demonstrate that the correct number of vertices is returned for higher dimensions
    n = 10
    bounds = [(0,1)]*n
    N = 2
    t0 = time.time()
    pts = boxgrid(bounds, N, interior=False)
    assert(len(pts)==N**10)

    print("Time elapsed: {} s".format(time.time()-t0))

    plt.show()

if __name__ == "__main__":
    test_boxgrid()
