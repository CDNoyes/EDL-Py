""" A base class for solution of OCP by differential dynamic programming """
import matplotlib.pyplot as plt 
import numpy as np 
import pyaudi as pa 
import pdb 

import DA as da 
from submatrix import submatrix

class DDP:
    """ 
        A base class for DDP solutions 
    """
    
    def __init__(self):
        pass 
    
    def solve(self, x0, N, bounds, u=None, maxIter=10):  # TODO: This should be implemented in specific versions of DDP, this algorithm is specific to control-limited 

        self.bounds = np.asarray(bounds)
        self.N = N 
        self.n = len(x0)
        if self.bounds.ndim == 1:
            self.m = 1
        else:
            self.m = self.bounds.shape[1]

        self.xnames = ["x{}".format(i) for i in range(self.n)] 
        self.unames = ["u{}".format(i) for i in range(self.m)]

        if u is None:
            u = np.zeros((self.N, self.m)) # Initial guess

        u = np.clip(u, *bounds)
        
        J = []
        for iter in range(maxIter):
            print("Iteration {}".format(iter))
            # Forward propagation
            
            x = self.forward(x0, u)
            Ld, Fd = self.Lagrange(x, u)
            Fg, Fh = Fd
            L, Lg, Lh = Ld
            LN, Mx, Mxx = self.Mayer(x[-1])

            J.append(np.sum(L) + LN)
            if len(J) > 1:
                if np.abs(J[-1]-J[-2]) < self.convergenceTolerance:
                    break
            if iter < 10 or not (iter+1)%10:            
                plt.figure(1)
                plt.plot(list(range(self.N)),x,'--',label="{}".format(iter))
                plt.figure(2)
                plt.plot(list(range(self.N)),u, '--',label="{}".format(iter))
                # plt.figure(4)
                # plt.plot(L, 'o', label="{}".format(iter))


            # Final conditions on Value function and its derivs
            V = LN  
            Vx = Mx
            Vxx = Mxx
            
            # Backward propagation
            k = [np.zeros((m,))]*N
            K = [np.zeros((m, n))]*N
            for i in range(N)[::-1]:
                Qx = Lx[i] + Fx[i]*Vx
                Qu = Lu[i] + Fu[i]*Vx
                Qxx = Lxx[i] + Fx[i]*Vxx*Fx[i]  + Vx*Fxx(x[i], u[i])
                Qux = Lux[i] + Fu[i]*Vxx*Fx[i]  + Vx*Fux(x[i], u[i])
                Quu = Luu[i] + Fu[i]*Vxx*Fu[i]  + Vx*Fuu(x[i], u[i])
                
                # No control limits 
                if False:
                    k[i] = (-Qu/Quu) # Shouldnt be any divide by zero error here 
                    K[i] = (-Qux/Quu)
                # Bounded controls:
                else:
                    k[i], Quuf = solveQP(k[i], Quu, Qu, ([bounds[0]-u[i]], [bounds[1]-u[i]]), verbose=False)
                    K[i] = -np.linalg.pinv(Quuf).dot(Qux)  # Pinv is the same as inv when Quuf > 0, but also correctly handles clamped directions Quuf >= 0

                
                V += -0.5*k[i]*Quu*k[i]
                Vx = Qx-K[i]*Quu*k[i]
                Vxx = Qxx-K[i]*Quu*K[i] 
            
            # Forward correction
            x, u = self.correct(x, u, k, K)  # Backtracking line search, simply looks to reduce cost

            
        plt.figure(1)    
        plt.plot(list(range(self.N)), x,label='Final')
        plt.title('State')
        plt.legend()
        plt.figure(2)
        plt.plot(list(range(self.N)), u, label='Final')
        plt.title('Control')
        plt.legend()
        plt.figure(3)
        plt.semilogy(J,'o')
        plt.title('Cost vs Iteration')
        plt.show()
            
        return u     
            
    def correct(self, x, u, k, K):
    
        step = 1
        J = self.evalCost(x, u)
        Jnew = J+1
        while Jnew > J: # should put a max iteration limit as well 
            xnew = [self.x0]
            unew = []
            for i in range(self.N-1):
                unew.append(np.clip(u[i] + step*k[i] + K[i]*(xnew[i]-x[i]), *self.bounds)) # This line search is actually essential to convergence 
                xnew.append(self.transition(xnew[i], unew[i], i))    
            unew.append(unew[-1]) # Just so it has the correct number of elements   
            Jnew = self.evalCost(np.array(xnew), np.array(unew).squeeze())
            step *= 0.8
        return np.array(xnew), np.array(unew).squeeze()
        
        
    def evalCost(self, x, u):
        L = self.lagrange(x, u)
        LN = self.mayer(x[-1])
        return sum(L) + LN 
        
    def transition(self, x, u, N):
        return NotImplementedError
      
    def mayer(self, xf):
        return NotImplementedError

    def lagrange(self, x, u):
        raise NotImplementedError
    
    def Q(self, l, V):  # Pseudo-Hamiltonian, i.e. its discrete analogue 
        return l + V
        
    def forward(self, x0, u):
        """ DA-based forward rollout to compute derivatives of transition and lagrange terms """

        x = [x0]

        for i in range(self.N):
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
        Fh = [] # List of hessian tensors 

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


class Test(DDP):
    """ Inverted pendulum example from A Generalized Iterative LQG Method..."""

    def __init__(self):
        pass

    def transition(self, x, u, N):
        dt = 0.01
        dx = np.array([x[1], u - 4*pa.sin(x[0])])
        return x + dx*dt

    def lagrange(self, x, u):
        return 0.01*u**2

    def mayer(self, x):
        return (1+pa.cos(x[0]))**2 + 0.1*x[1]**2

    def x0(self):
        return np.array([1., 2])

    

if __name__ == "__main__":


    test = Test()
    test.solve(test.x0(), N=400, bounds=(-2, 2))