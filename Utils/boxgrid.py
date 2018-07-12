from itertools import product
import numpy as np


def boxgrid(bounds, N, interior=False, chebyshev=False, surface=False):
    """ n-dimensional box grid, either just the exterior, or with interior points as well.
    Inputs:
        bounds is an n-length list/tuple with each element being the (min,max) along that dimension
        N is the number of samples per dimension, either a scalar or n-length list of integers
        interior indicates whether or not only surface points are kept
        chebyshev uses Chebyshev nodes instead of equal spacing 
        surface keeps points along the surface, otherwise only edge points are kept.

    surface=True is only meaningful for n>2 and interior=False
    Setting N=2 and interior=False, surface=False will return just the vertices of the hyper-rectangle.

    """
    if isinstance(N, int):
        N = [N for _ in bounds]
    if surface:
        subfactor = 1
    else:
        subfactor = len(bounds) - 1

    n = len(bounds)
    if chebyshev:
        vectors = [0.5*(1-np.cos(np.pi*np.arange(0, ni)/(ni-1)))*(b[1]-b[0])+b[0] for b, ni in zip(bounds, N)]
    else:
        vectors = [np.linspace(b[0], b[1], ni) for b, ni in zip(bounds, N)]

    grid_points = np.array(list(product(*vectors)))
    if not interior:                # Remove the interior points
        bmin = np.array(bounds)[:,0]
        bmax = np.array(bounds)[:,1]
        ikeep = []
        for i, pt in enumerate(grid_points):
            # In n-dimensions, >=(n-1) points must be extremal to constitute an edge
            # Only 1 pt must be extremal required to be a surface point ?
            if np.count_nonzero(np.logical_or(pt-bmin == 0, pt-bmax == 0)) >= subfactor:  # Count is faster than sum for some reason
                ikeep.append(i)
        grid_points = grid_points[ikeep]
    return grid_points


def test_boxgrid():

    # 2-d example
    import matplotlib.pyplot as plt
    import time

    pts = boxgrid(((-3, 3),(-1, 1)), (10, 6), interior=True, chebyshev=True)
    plt.figure()
    plt.plot(pts[:,0], pts[:,1], 'o')

    pts = boxgrid(((-3, 3), (-1, 1)), 5, interior=False, chebyshev=True)
    plt.figure()
    plt.plot(pts[:,0], pts[:,1], 'o')

    # Demonstrate that the correct number of vertices is returned for higher dimensions
    # n = 10
    # bounds = [(0,1)]*n
    # N = 2
    # t0 = time.time()
    # pts = boxgrid(bounds, N, interior=False)
    # assert(len(pts)==N**10)

    # print("Time elapsed: {} s".format(time.time()-t0))

    plt.show()

if __name__ == "__main__":
    test_boxgrid()
