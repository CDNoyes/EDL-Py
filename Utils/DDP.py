""" A base class for solution of OCP by differential dynamic programming """
import matplotlib.pyplot as plt 
import numpy as np 
import pyaudi as pa 

import pdb 

import DA as da 
from submatrix import submatrix
from ProjectedNewton import ProjectedNewton as solveQP
from Regularize import AbsRegularize

class DDP:
    """ 
        A base class for DDP solutions 
    """
    
    def __init__(self):
        pass 
    
    def solve(self, x0, N, bounds, u=None, maxIter=10, tol=1e-6, verbose=True):  # TODO: This should be implemented in specific versions of DDP, this algorithm is specific to control-limited 
        reg = 1   # initial regularization parameter 
        dreg = 1  # initial regularization update parameter 
        reg_factor = 1.6
        gradient_tol = 1e-5
        objective_tol = 1e-7

        reduction_ratio = []
        self.bounds = np.asarray(bounds)
        self.N = N 
        self.n = len(x0)
        if self.bounds.ndim == 1:
            self.m = 1
        else:
            self.m = self.bounds.shape[1]

        self.xnames = ["x{}".format(i) for i in range(self.n)] 
        self.unames = ["u{}".format(i) for i in range(self.m)]
        p = self.n+self.m
        xrows = list(range(self.n))
        urows = list(range(self.n, p))


        if u is None:
            u = np.zeros((self.N, self.m)) # Initial guess
        else:
            if np.ndim(u) == 1:
                u = u[:, None]
        u = np.clip(u, *bounds)
        u[-1] = 0 # Always 
        
        J = []
        for iter in range(maxIter):
            # Forward propagation
            if not iter:
                x = self.forward(x0, u)
            
            # Derivatives along trajectory
            Ld, Fd = self.Lagrange(x, u)
            Fg, Fh = Fd
            L, Lg, Lh = Ld
            LN, Mx, Mxx = self.Mayer(x[-1])

            J.append(np.sum(L) + LN)
            # if len(J) > 1:
            #     if np.abs(J[-1]-J[-2]) < tol:
            #         break



            # Final conditions on Value function derivs
            dV = [0, 0]
            Vx = Mx + Lg[-1][0:self.n]
            Vxx = Mxx + submatrix(Lh[-1], xrows, xrows)
            # Backward propagation
            k = [np.zeros((self.m,))]*self.N
            K = [np.zeros((self.m, self.n))]*self.N
            for i in range(N-1)[::-1]:
                Qg = Lg[i] + Fg[i].T @ Vx
                Qx = Qg[0:self.n]
                Qu = Qg[self.n:p]

                Qh = Lh[i] + Fg[i].T @ Vxx @ Fg[i] + vector_tensor(Vx, Fh[i])
                Qxx = submatrix(Qh, xrows, xrows)
                Qux = submatrix(Qh, urows, xrows)
                Quu = submatrix(Qh, urows, urows)

                
                # No control limits 
                if False:
                    k[i] = (-Qu/Quu)  # Shouldnt be any divide by zero error here 
                    K[i] = (-Qux/Quu)
                # Bounded controls:
                else:
                    QuuP = AbsRegularize(Quu, shift=reg)
                    k[i], Quuf = solveQP(k[i], QuuP, Qu, ([bounds[0]-u[i]], [bounds[1]-u[i]]), verbose=False)
                    K[i] = -np.linalg.pinv(Quuf) @ Qux  # Pinv is the same as inv when Quuf > 0, but also correctly handles clamped directions Quuf >= 0
                
                dV = [dV[0] + k[i].T @ Qu, dV[1] + 0.5*(k[i].T @ Quu @ k[i]).squeeze()]
                Vx = (Qx + K[i].T @ Quu @ k[i] + K[i].T @ Qu + Qux.T @ k[i]).squeeze()
                Vxx = Qxx + K[i].T @ Quu @ K[i] + K[i].T @ Qux + Qux.T @ K[i]
                Vxx = 0.5*(Vxx + Vxx.T)

            # Forward correction
            xnew, unew, Jnew, step, success = self.correct(x, u, k, K)  # Backtracking line search, simply looks to reduce cost
            if success:
                dreg = np.min((dreg/reg_factor, 1/reg_factor))
                reg *= dreg * float(reg > 1e-8)
                x = xnew 
                u = unew 
                reg = np.clip(reg, 1e-8, 1e6)
            else:
                dreg = np.maximum(dreg*reg_factor, reg_factor)
                reg = np.maximum(reg*reg_factor, 1e-4) # minimum regulation 
                if reg > 1e8:
                    if verbose:
                        print("Maximum regulation achieved and step failed. Aborting.")
                    break
                # if verbose:
                #     print("No cost reduction achieved. Increasing regulation.")
                continue 

            expected = -step*(dV[0] + step*dV[1])
            g_norm = np.mean(np.max(np.abs(k)/(np.abs(u)+1), axis=1))
            if g_norm < gradient_tol and reg < 1e-3:
                if verbose:
                    print("Success: gradient norm less than tolerance with minimal regulation")
                break
            reduction_ratio.append((J[-1]-Jnew)/expected) # this is "z" in matlab implement 
            if J[-1]-Jnew < objective_tol:
                if verbose:
                    print("Success: change in objective function smaller than tolerance")
                break
            if expected < objective_tol/10:
                if verbose:
                    print("Success: expected change in objective function smaller than tolerance")
                break

            if verbose:
                if not iter or not iter % 5:
                    header_print()
                iter_print(iter, J[-1], J[-1]-Jnew, expected, g_norm, reg)
            
        plt.figure(1)    
        plt.plot(list(range(self.N)), x, label='Final')
        plt.title('State')
        # plt.legend()
        plt.figure(2)
        plt.plot(list(range(self.N)), u, label='Controls')
        plt.plot(list(range(self.N)), np.reshape(K, (N, self.n*self.m)), 'k', label='Gains', alpha=0.1)
        plt.title('Control')
        # plt.legend()
        plt.figure(3)
        plt.plot(J, 'o')
        plt.title('Cost vs Iteration')
        plt.figure()
        plt.plot(reduction_ratio)
        plt.title("Actual/Expected Cost Reduction")
        # plt.show()
            
        return x, u, K     
            
    def correct(self, x, u, k, K):
        min_step = 1e-2
        step = 1
        J = self.evalCost(x, u)
        Jnew = J+1
        success = False 
        while step > min_step: 
            unew = []
            xnew = [x[0]]
            for i in range(self.N-1):
                unew.append(np.clip(u[i] + step*k[i] + K[i]@(xnew[i]-x[i]), *self.bounds)) # This line search is actually essential to convergence 
                xnew.append(self.transition(xnew[i], unew[i], i))    
            unew.append(unew[-1]) # Just so it has the correct number of elements   
            Jnew = self.evalCost(np.array(xnew), np.array(unew).squeeze())
            if Jnew < J:
                success = True 
                break 
            step *= 0.8

        u = np.array(unew).squeeze()
        if np.ndim(u) == 1:
            u = u[:, None]
        return np.array(xnew), u, Jnew, step, success 
        
    # def da_correct(self):  #Make the stepsize a DA, and solve an optimization problem to find the next step 

    def evalCost(self, x, u):
        L = [self.lagrange(xi, ui) for xi, ui in zip(x, u)]
        LN = self.mayer(x[-1])
        return np.sum(L) + LN
        
    def transition(self, x, u, N):
        raise NotImplementedError
      
    def mayer(self, xf):
        raise NotImplementedError

    def lagrange(self, x, u):
        raise NotImplementedError
    
    def Q(self, l, V):  # Pseudo-Hamiltonian, i.e. its discrete analogue 
        return l + V
        
    def forward(self, x0, u):
        """ DA-based forward rollout to compute derivatives of transition and lagrange terms """

        x = [x0]

        for i in range(self.N-1):
            x.append(self.transition(x[i], u[i], i))

        return np.array(x)

    def Mayer(self, xf):

        x = da.make(xf, self.xnames, 2, True, False)
        M = self.mayer(x)

        return da.const(M), da.gradient(M, self.xnames), da.hessian(M, self.xnames)

    def Lagrange(self, x, u):
        
        p = self.n+self.m
        xrows = list(range(self.n))
        urows = list(range(self.n, p))

        Fg = []
        Fh = []  # List of hessian tensors

        L = []
        Lg = []
        Lh = []        

        for i in range(self.N):
            y0 = da.make(x[i], self.xnames, 2, array=True)
            v0 = da.make(u[i], self.unames, 2, array=True).squeeze()

            y = self.transition(y0, v0, i)

            F = da.jacobian(y, self.xnames+self.unames)
            Fg.append(F)

            HF = da.vhessian(y, self.xnames+self.unames)  # There's one hessian per state [n, p, p]
            Fh.append(HF)

            Ly = self.lagrange(y0, v0)
            L.append(da.const(Ly))

            g = da.gradient(Ly, self.xnames+self.unames)
            Lg.append(g)

            H = da.hessian(Ly, self.xnames+self.unames)
            Lh.append(H)

        return (L, Lg, Lh), (Fg, Fh)


def vector_tensor(v, T):
    # assumes first two dimensions match
    return np.sum([vi*Ti for vi, Ti in zip(v, T)])


def header_print():
    print("iteration    cost        reduction    expected     gradient     log10(lambda)")


def iter_print(iter, cost, reduction, expected, gradient, regularization):
    print("   {:<2}        {:<13.4g}{:<13.3g}{:<13.3g}{:<13.3g}{:<5.3g}".format(iter, cost, reduction, expected, gradient, np.log10(regularization)))


class Test(DDP):
    """ Inverted pendulum example from A Generalized Iterative LQG Method..."""

    def __init__(self):
        self.dt = 0.01

    def transition(self, x, u, N):
        dx = np.array([x[1], u - 4*pa.sin(x[0])])
        return x + dx*self.dt

    def lagrange(self, x, u):
        return 0.01*u**2

    def mayer(self, x):
        # return ( (1+pa.cos(x[0]))**2 + 0.1*x[1]**2 )
        return 10*( (x[0] - np.pi)**2 + x[1]**2 )

    def x0(self):
        return np.array([1.5, 0.])


class Linear(DDP):

    def __init__(self, n, m, h):
        from scipy.linalg import expm 
        self.dt = h 
        A = np.random.standard_normal((n, n))
        A = A - A.T 
        # A = np.diag([0, -1, 0.1, -0.5])
        A = expm(A*h)
        B = h*np.random.standard_normal((n,m))
        # B = h*np.ones((n,m))
        self.A = A 
        self.B = B 

    def transition(self, x, u, k):
        return self.A.dot(x) + self.B.dot(u)

    def lagrange(self, x, u):
        Q = self.dt  
        R = 0.01 * self.dt 
        return 0.5*Q*x.dot(x) + 0.5*R*u.dot(u)

    def mayer(self, xf):
        return 0 

if __name__ == "__main__":


    # test = Test()
    # bounds = np.ones((2, 1))*2
    # bounds[0] *= -1
    # x,u,K = test.solve(test.x0(), N=400, bounds=(-2, 2), maxIter=100)

    # plt.figure()
    # plt.plot(x.T[0], x.T[1])
    # plt.plot(np.pi, 0, 'kx')
    # plt.plot(-np.pi, 0, 'kx')
    # plt.title("Optimal Swing Up Maneuver")
    # plt.show()


    n = 10
    m = 5
    N = 1000
    test = Linear(n, m, 0.01)
    x0 = np.random.standard_normal((n,))
    # x0 = np.ones((n,))
    u0 = 0.1*np.random.standard_normal((N, m))
    # u0 = np.ones((N,m))*0.01
    bounds = np.ones((2, m))*0.6
    bounds[0] *= -1
    x,u,K = test.solve(x0, N, bounds=bounds, u=u0, maxIter=150) 

    plt.show()