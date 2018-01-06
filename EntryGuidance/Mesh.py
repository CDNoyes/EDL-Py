""" A class for working with multiple intervals of collocation points """

import numpy as np
from Chebyshev import ChebyshevDiff, ChebyshevQuad

class Mesh(object):

    def __init__(self, min_order=2, max_order=10):
        self.min = min_order
        self.max = max_order

        Ni = range(self.min,self.max+1)

        tw = [ChebyshevQuad(N) for N in Ni]

        self._D = [ChebyshevDiff(N) for N in Ni]        # All possible differentiation matrices
        self._tau = [twi[0] for twi in tw]              # All [-1,1] points
        self._w   = [twi[1] for twi in tw]              # All Clenshaw-Curtis weights

        self.orders = [2,4,2]                 # initial mesh orders
        self.points = [self.tau(N) for N in self.orders] # list over array because they may be different lengths
        self.diffs = [self.D(N) for N in self.orders]
        self.n_points = sum(self.orders)+1 # number of actual collocation points, accounting for meshes overlap in the interior (and e.g. N=2 yields 3 points)


    def D(self,N):
        return self._D[N-self.min]

    def tau(self,N):
        return self._tau[N-self.min]

    def w(self,N):
        return self._w[N-self.min]

    def tau2time(self, times):
        """ For a list of times such (len(times) = len(mesh)+1), returns the true times
            corresponding to the normalized times self.points
        """
        assert(len(times) == len(self.orders)+1)
        return [0.5*((tb-ta)*tau + ta+tb) for ta,tb,tau in zip(times,times[1:],self.points)] # map the collocation points to the true time interval

    def chunk(self, x):
        """
        Takes an array x of size (self.n_points, arbitary dimensions) and returns a list of
        decompositions of x matching the length of the collocation points in each mesh segment.

        Useful to avoid linkage constraints between two mesh segments by using
        the same point for both segments.

        """
        X = []
        tally = 0
        for order in self.orders:
            X.append(x[tally:tally+order+1])
            tally += order
        return X





    # def split(self, i):



    # def __getitem__(self,i):



if __name__ == "__main__":
    mesh = Mesh(min_order=2, max_order=10)
    # print mesh.tau2time([0,3,10])
    # print mesh.tau2time([1,9,10])
    # x = np.random.random((mesh.n_points,3))
    x = np.linspace(0,mesh.n_points-1,mesh.n_points)
    print x
    X = mesh.chunk(x)
    print X
