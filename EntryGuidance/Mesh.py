import numpy as np
from scipy.interpolate import interp1d
from Chebyshev import ChebyshevDiff, ChebyshevQuad

class Mesh(object):
    """ Defines a mesh class for working with multiple intervals of collocation points """

    def __init__(self, tf, orders=None, min_order=2, max_order=12, t0=0, default_order=6, increment=2):
        """ Initializes an instance of the Mesh class """
        self.min = min_order
        self.max = max_order
        self.default = default_order                 # Default order when splitting a mesh into two
        self.inc = increment                         # Default increase when raising the order of a segment

        Ni = range(self.min,self.max+1)
        tw = [ChebyshevQuad(N) for N in Ni]

        # Stores of data
        self._D = [ChebyshevDiff(N) for N in Ni]        # All possible differentiation matrices
        self._tau = [twi[0] for twi in tw]              # All [-1,1] points
        self._w   = [twi[1] for twi in tw]              # All Clenshaw-Curtis weights

        if orders is None:
            self.orders = [self.default]*10                 # default initial mesh orders
        else:
            self.orders = orders

        self._times = np.linspace(t0,tf,len(self.orders)+1).tolist() # The times representing the mesh end points [t0, t1, t2, ..., tf]
        self.points = [self.tau(N) for N in self.orders] # list over array because they may be different lengths
        self.diffs = [self.D(N)*2./interval for N,interval in zip(self.orders,np.diff(self._times))] # Differentiation matrices scaled for their appropriate interval
        self.n_points = sum(self.orders)+1 # number of actual collocation points, accounting for meshes overlap in the interior (and e.g. N=2 yields 3 points)
        self.times = self.tau2time(self._times)
        self.weights = [self.w(N)*interval/2. for N,interval in zip(self.orders, np.diff(self._times))]

        self.history = [self._times[:]]

    def D(self,N):
        return self._D[N-self.min]

    def tau(self,N):
        return self._tau[N-self.min]

    def w(self,N):
        return self._w[N-self.min]

    def tau2time(self, times, mesh=False):
        """
        For a list of times such that (len(times) = len(mesh)+1), returns the
        true times corresponding to the normalized times in self.points

        """
        assert(len(times) == len(self.orders)+1)
        mesh_times = [0.5*((tb-ta)*tau + ta+tb) for ta,tb,tau in zip(times,times[1:],self.points)] # map the collocation points to the true time interval
        if mesh: # return a mesh of times, i.e. a list of lists
            return mesh_times
        else:
            # remove the first element of every array after the first
            mesh_times = [mesh_times[0]] + [t[1:] for t in mesh_times[1:]]
            all_times = np.concatenate(mesh_times)
            return all_times

    def chunk(self, x):
        """
        Takes an array x of size (self.n_points, arbitary dimensions) and returns a list of
        decompositions of x matching the length of the collocation points in each mesh segment.

        Useful to avoid linkage constraints between two mesh segments by using
        the same point for both segments.

        """
        if np.shape(x):
            assert(np.shape(x)[0]==len(self.times))
        X = []
        tally = 0
        for order in self.orders:
            X.append(x[tally:tally+order+1])
            tally += order
        return X

    def update(self, i, new_order):
        """ Change the order of the ith mesh """
        intervals = np.diff(self._times)
        self.orders[i] = new_order
        self.points[i] = self.tau(new_order)
        self.diffs[i]  = self.D(new_order)*2./intervals[i]
        self.weights[i]  = self.w(new_order)*intervals[i]/2.
        self.n_points = sum(self.orders)+1
        self.times = self.tau2time(self._times)

    def split(self, i, t=None):
        """ Splits the ith mesh at time t into 2 meshes
            If t is not given, the mesh is bisected.
        """
        # Validate the inputs
        assert(i<len(self.orders))
        if t is None: #Split the mesh in half by default
            t = (self._times[i]+self._times[i+1])/2.
        else:
            if self._times[i]<self._times[i+1]:
                assert(t>self._times[i])
                assert(t<self._times[i+1])
            else:
                assert(t<self._times[i])
                assert(t>self._times[i+1])

        self._times.insert(i+1, t)
        intervals = np.diff(self._times)

        self.orders.insert(i, self.default)
        self.orders[i+1] = self.default

        self.points.insert(i, self.tau(self.default))
        self.points[i+1] = self.tau(self.default)

        self.diffs.insert(i, self.D(self.default)*2./intervals[i])
        self.diffs[i+1]  = self.D(self.default)*2./intervals[i+1]

        self.weights.insert(i, self.w(self.default)*intervals[i]/2.)
        self.weights[i+1]  = self.w(self.default)*intervals[i+1]/2.

        self.n_points = sum(self.orders)+1
        self.times = self.tau2time(self._times)

    def refine(self, X, F, tol=1e-3, rho=1, scaling=None, verbose=True):
        """ Refines the mesh using hp-adaptation

            tol is the tolerance on the residual matrix above which the mesh is refined
            rho is a fraction (>0) relative to the mean scaled residual above which the mesh is split
                e.g. rho=0.5 corresponds to errors 50% greater than the mean error

        """
        # X,F are the current solution and its derivative on the current mesh

        if scaling is None:
            scaling = np.ones_like(X)
        if len(np.shape(scaling))==1:
            scaling = np.asarray(scaling)[None,:]

        refined = False
        rho += 1.

        intervals = np.diff(self._times)/2.
        Tc = self.points
        Xc = self.chunk(X)
        Fc = self.chunk(F)

        segment = 0
        for t,x,f,interval in zip(Tc,Xc,Fc,intervals):
            ti = t[:-1] + np.diff(t)  # Midpoints in (-1,1)
            Di = colloc(ti)          # Collocation matrix on arbitrary points

            xi = interp1d(t, x, kind='cubic', assume_sorted=True, axis=0)(ti)
            fi = interp1d(t, f, kind='cubic', assume_sorted=True, axis=0)(ti)

            R = np.abs(Di.dot(xi) - interval*fi)/scaling    # Residual matrix
            ij = np.unravel_index(R.argmax(), R.shape)
            if len(ij)==2:
                col = ij[1]
                r = R[:,col]                    # Residual column with the largest error
            else:
                col=ij
                r = R
            beta = r/r.mean()               # scaled midpoint residual vector

            if R[ij] > tol:
                if verbose: print "Refining segment {}".format(segment)
                refined = True

                if beta.max() <= rho and (self.orders[segment]+self.inc < self.max): # Uniform type error
                    if verbose: print "Raising polynomial order..."
                    self.update(segment, self.orders[segment]+self.inc)

                else: # Isolated errors or max order reached
                    if verbose: print "Splitting the segment..."
                    # Find the highest points (skipping adjacent errors )
                    if self.orders[segment]+self.inc<self.max:
                        # splits = np.where(beta>=rho)[0]
                        # ds = np.diff(splits)
                        splits = [np.argmax(beta)] # for now, just split at the highest error point
                    else: # Resulting mesh order would be too high, so we split at the highest error point instead
                        splits = [np.argmax(beta)]

                    for isplit in splits:
                        tsplit = (ti[isplit]*interval + self._times[segment] +self._times[segment+1])/2. # convert to real time
                        self.split(segment, t=tsplit) # have to add another increment each time a split is done
                        segment +=1
            segment+= 1
        self.history.append(self._times[:])
        return refined

    def plot(self,show=True):
        """ Plots the mesh history """
        import matplotlib.pyplot as plt

        plt.figure()
        for i,t in enumerate(self.history):
            plt.plot(t,np.ones_like(t)*i,'x')
        plt.xlabel('Grid points')
        plt.ylabel('Refinements')

        plt.figure()
        plt.hist(self.orders)
        plt.xlabel('Mesh orders')

        if show:
            plt.show()

def colloc(x):
    """
        Computes the pseudospectral/collocation differentiation matrix for the
        arbitrary nodes stored in the vector x. Uses the lagrange polynomial
        formulation.

        Reference:
        Jean-Paul Berrut & Lloyd N. Trefethen, "Barycentric Lagrange Interpolation"
        http://web.comlab.ox.ac.uk/oucl/work/nick.trefethen/berrut.ps.gz
    """
    x = np.asarray(x)
    x.sort()
    N = x.size
    N1 = N+1
    N2 = N*N
    X = np.tile(x,(N,1)).T
    Xd = X-X.T+np.eye(N)
    W = np.tile(1./np.prod(Xd, axis=1),(N,1)).T
    D = W/(W.T * Xd)
    Ds = np.sum(D,axis=0)
    D = D - np.diag(np.diag(D)) + np.diag(1-Ds)
    D = -D.T
    return D

def test_quad():
    """ A test of numerical quadrature using a mesh with Chebyshev polynomials"""
    from scipy.integrate import quad

    y = lambda x: (x**4 - 2 * x**3)*np.sin(x) + np.exp(0.1*x)*np.cos(x)*x

    bound = -2 # Integration bounds

    I = quad(y, -bound, bound)[0]
    M = Mesh(t0=-bound, tf=bound, orders=[12], max_order=30)
    integral = 0
    for x,w in zip(M.tau2time(M._times, True), M.weights):
        integral += w.dot(y(x))



    print("Integrating test function: \nf(x) = (x**4 - 2 * x**3)*sin(x) + exp(0.1*x)*cos(x)*x")
    print("True value using scipy.integrate.quad = {}".format(I))
    print("Value using mesh quad = {}".format(integral))


if __name__ == "__main__":
    test_quad()
