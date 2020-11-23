""" A base class for solution of OCP by differential dynamic programming """
import matplotlib.pyplot as plt 
import numpy as np 
import pyaudi as pa 

import pdb 

import DA as da 
from submatrix import submatrix
from ProjectedNewton import ProjectedNewton as solveQP
from QP import TRQP 
from Regularize import Regularize, AbsRegularize

class DDP:
    """ 
        A base class for DDP solutions 
    """
    
    def __init__(self):
        pass 
    
    def solve(self, x0, N, u, bounds=None, maxIter=10, tol=1e-6, verbose=True, lambda0=1, order=2):  # TODO: This should be implemented in specific versions of DDP, this algorithm is specific to control-limited 
        history = {'lambda': [], 'cost': [], 'state': [], 'control': [], "reduction_ratio": [], "gains": []}

        reg = lambda0   # initial regularization parameter 
        dreg = 1  # initial regularization update parameter 
        reg_factor = 2.6
        reg_min = 1e-6  # below this reg = 0 
        gradient_tol = 1e-4
        objective_tol = 1e-7

        no_bounds = bounds is None 
        reduction_ratio = []
        if bounds is None:
            self.bounds = [-np.inf, np.inf]
        else:
            self.bounds = np.asarray(bounds)
        self.N = N 
        self.n = len(x0)
        if np.ndim(u) == 1:
            self.m = 1
        else:
            self.m = len(u[0])

        self.xnames = ["x{}".format(i) for i in range(self.n)] 
        self.unames = ["u{}".format(i) for i in range(self.m)]
        p = self.n+self.m
        xrows = list(range(self.n))
        urows = list(range(self.n, p))

        if np.ndim(u) == 1:
            u = u[:, None]

        if bounds is not None:    
            u = np.clip(u, *bounds)
        u[-1] = 0  # Always 
        
        J = []
        success_iter = 0
        total_iter = 0
        trials = 1  # Number of attempted steps before success 
        while True:
            # Forward propagation
            if not total_iter:
                x = self.forward(x0, u)
            
            # Derivatives along trajectory
            Ld, Fd = self.Lagrange(x, u, order=order)
            Fg, Fh = Fd
            L, Lg, Lh = Ld
            LN, Mx, Mxx = self.Mayer(x[-1])
            if not J:
                J.append(np.sum(L) + LN)
           
            # Final conditions on Value function derivs
            dV = [0, 0]
            Vx = Mx + Lg[-1][0:self.n]
            Vxx = Mxx + submatrix(Lh[-1], xrows, xrows)

            # Backward propagation
            k = [np.zeros((self.m,))]*self.N
            K = [np.zeros((self.m, self.n))]*self.N
            for i in range(N)[::-1]:
                Qg = Lg[i] + Fg[i].T @ Vx
                Qx = Qg[0:self.n]
                Qu = Qg[self.n:p]

                Qh = Lh[i] + Fg[i].T @ Vxx @ Fg[i] + vector_tensor(Vx, Fh[i])
                Qxx = submatrix(Qh, xrows, xrows)
                Qux = submatrix(Qh, urows, xrows)
                Quu = submatrix(Qh, urows, urows)

                # Different regularization methods 
                # QuuP = AbsRegularize(Quu, shift=reg)
                QuuP = Quu + np.eye(self.m)*reg  # this is simple but avoids eigendecomp
                # QuuP = Regularize(QuuP, 1e-3)

                # No control limits 
                if bounds is None:
                    k[i] = -np.linalg.lstsq(QuuP, Qu, rcond=None)[0] 
                    K[i] = -np.linalg.lstsq(QuuP, Qux, rcond=None)[0]
                # Bounded controls:
                else:
                    k[i], Quuf, free = solveQP(k[i], QuuP, Qu, ([bounds[0]-u[i]], [bounds[1]-u[i]]), verbose=False, tol=1e-3, iter_max=500)
                    K[i] = np.linalg.lstsq(-Quuf, Qux, rcond=None)[0] 
                    # K[i] = -np.linalg.pinv(Quuf) @ Qux  # Pinv is the same as inv when Quuf > 0, but also correctly handles clamped directions Quuf >= 0
                
                dV = [dV[0] + k[i].T @ Qu, dV[1] + 0.5*(k[i].T @ Quu @ k[i]).squeeze()]
                Vx = (Qx + K[i].T @ Quu @ k[i] + K[i].T @ Qu + Qux.T @ k[i]).squeeze()
                Vxx = Qxx + K[i].T @ Quu @ K[i] + K[i].T @ Qux + Qux.T @ K[i]
                Vxx = 0.5*(Vxx + Vxx.T)

            # Forward correction
            xnew, unew, Jnew, step, success = self.correct(x, u, k, K)  # Backtracking line search, simply looks to reduce cost
            # xnew, unew, Jnew, step, success = self.full_correct(x, u, k, K)  # Backtracking line search, looks for best cost 
            total_iter += 1
            if success:
                expected = -step*(dV[0] + step*dV[1])
                if expected < 0:
                    print("Warning: expected reduction < 0")
                    expected = J[-1]-Jnew 
                x = xnew 
                u = unew 
                J.append(Jnew)
                dJ = J[-2]-J[-1]
                success_iter += 1 

                g_norm = np.mean(np.max(np.abs(k)/(np.abs(u)+1), axis=1))
                reduction_ratio.append(dJ/expected)  # this is "z" in matlab implement 
                self.update_history(history, Jnew, x, u, reg, reduction_ratio, K)

                if verbose:
                    if success_iter == 1 or not success_iter % 5:
                        header_print()
                    iter_print(success_iter, J[-2], dJ, expected, g_norm, reg, trials)

                trials = 1
                dreg = np.min((dreg/reg_factor, 1/reg_factor))
                reg *= dreg * float(reg > reg_min)
                reg = np.clip(reg, 0, 1e8)

                # Convergence checks 
                if g_norm < gradient_tol and reg < 1e-3:
                    if verbose:
                        print("Success: gradient norm less than tolerance with minimal regularization")
                    break
                if dJ < objective_tol:
                    if verbose:
                        print("Success: relative change in objective function smaller than tolerance")
                    break
                if expected < objective_tol/10:
                    if verbose:
                        print("Success: expected change in objective function smaller than tolerance")
                    break

                if success_iter >= maxIter:
                    if verbose:
                        print("Maximum number of iterations reached")
                    break

            else:
                trials += 1
                dreg = np.maximum(dreg*reg_factor, reg_factor)
                reg = np.maximum(reg*reg_factor, reg_min)  # minimum regulation 
                if reg > 1e8:
                    if verbose:
                        print("Maximum regulation achieved and step failed. Aborting.")
                    break
                continue 
            

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
        plt.semilogy(J, 'o')
        plt.title('Cost vs Iteration')
        plt.figure(4)
        plt.plot(reduction_ratio)
        plt.ylim(0, np.max((2, np.max(reduction_ratio))))
        plt.title("Actual/Expected Cost Reduction")
        # plt.show()
            
        return x, u, K     
            
    def correctQP(self, x, u, k, K):
        """ Solves a QP at each stage of the forward pass to compute the optimal update
            while respecting linearized constraints
        See DDP w/ nonlinear constraints for details
        """
        pass 


    def correct(self, x, u, k, K):
        min_step = 1e-4
        step = 1
        J = self.evalCost(x, u)
        success = False 
        while step > min_step: 
            unew = []
            xnew = [x[0]]
            for i in range(self.N-1):
                unew.append(np.clip(u[i] + step*k[i] + K[i]@(xnew[i]-x[i]), *self.bounds))  # This line search is actually essential to convergence 
                xnew.append(self.transition(xnew[i], unew[i], i))    
            unew.append(unew[-1]*0)  # Just so it has the correct number of elements   
            Jnew = self.evalCost(np.array(xnew), np.array(unew).squeeze())
            if Jnew < J:
                success = True 
                break 
            step *= 0.8
        step = np.max((step, min_step))
        u = np.array(unew).squeeze()
        if np.ndim(u) == 1:
            u = u[:, None]
        return np.array(xnew), u, Jnew, step, success 

    def full_correct(self, x, u, k, K):
        N = 21
        min_step = 1e-3
        steps = np.logspace(-3, 0, N)
        J = self.evalCost(x, u)
        step_opt = min_step
        X = [np.array([x[0]]*N)] 
        uopt = [u[:]]*N 
        for step in steps: 
            unew = []
            xnew = [x[0]]
            for i in range(self.N-1):
                unew.append(np.clip(u[i] + step*k[i] + K[i]@(xnew[i]-x[i]), *self.bounds))  # This line search is actually essential to convergence 
                xnew.append(self.transition(xnew[i], unew[i], i))    
            unew.append(unew[-1])  # Just so it has the correct number of elements   
            Jnew = self.evalCost(np.array(xnew), np.array(unew).squeeze())
            if Jnew < J:
                J = Jnew 
                step_opt = step 
                success = True 
                xopt = xnew[:]
                uopt = unew[:]

        u = np.array(uopt).squeeze()
        if np.ndim(u) == 1:
            u = u[:, None]
        return np.asarray(xopt), u, J, step, success     

    def update_history(self, history, cost, x, u, reg, ratio, gains):
        history['state'].append(x) 
        history['control'].append(u) 
        history['cost'].append(cost) 
        history['lambda'].append(reg) 
        history['reduction_ratio'].append(ratio) 
        history['gains'].append(gains) 

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

    def Lagrange(self, x, u, order):
        
        p = self.n+self.m
        xrows = list(range(self.n))
        urows = list(range(self.n, p))

        Fg = []
        Fh = []  # List of hessian tensors

        L = []
        Lg = []
        Lh = []        

        for i in range(self.N):
            y0 = da.make(x[i], self.xnames, order, array=True)
            v0 = da.make(u[i], self.unames, order, array=True).squeeze()

            y = self.transition(y0, v0, i)

            F = da.jacobian(y, self.xnames+self.unames)
            Fg.append(F)
            if order > 1:
                HF = da.vhessian(y, self.xnames+self.unames)  # There's one hessian per state [n, p, p]
            else:
                HF = np.zeros((self.n, p, p))
            Fh.append(HF)

            Ly = self.lagrange(y0, v0)
            L.append(da.const(Ly))

            g = da.gradient(Ly, self.xnames+self.unames)
            Lg.append(g)
            if order > 1:
                H = da.hessian(Ly, self.xnames+self.unames)
            else:
                H = np.zeros((p,p))
            Lh.append(H)

        return (L, Lg, Lh), (Fg, Fh)


def vector_tensor(v, T):
    # assumes first dimensions match
    return np.sum([vi*Ti for vi, Ti in zip(v, T)], axis=0)


def header_print():
    print("iteration    cost        reduction    expected     gradient     log10(lambda)   trial steps")


def iter_print(iter, cost, reduction, expected, gradient, regularization, trials):
    if regularization == 0:
        L = -np.inf 
    else:
        L = np.log10(regularization)
    print("   {:<2}        {:<13.4g}{:<13.3g}{:<13.3g}{:<13.3g}{:<18.3g}{:<5.0f}".format(iter, cost, reduction, expected, gradient, L, trials))

def history_plot(history):

class Test(DDP):
    """ Inverted pendulum example from DDP with Terminal Constraints..."""

    def __init__(self, dt):
        self.dt = dt

    def transition(self, x, u, N):
        m = 1
        l = 0.5 
        b = 0.1 
        I = m*l**2
        g = 9.81 
        dx = np.array([x[1], u/I + m*g*l*pa.sin(x[0])/I - b*x[1]/I])
        return x + dx*self.dt

    def lagrange(self, x, u):
        return self.dt * 0.1*u**2

    def mayer(self, x):
        return 100*x[0]**2 + 10*x[1]**2 

    def x0(self):
        return np.array([np.pi, 0.])


class Linear(DDP):

    def __init__(self, n, m, h):
        from scipy.linalg import expm 
        from ctrb import ctrb 
        self.dt = h 
        A = np.random.standard_normal((n, n))
        A = A - A.T 
        # A = np.diag([0, -1, 0.1, -0.5])
        B = h*np.random.standard_normal((n,m))
        assert ctrb(A, B), "Linear system is not controllable"
        A = expm(A*h)
        # B = h*np.vstack([np.eye(2)]*2)
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


    # tf = 0.5 
    # N = 500 
    # test = Test(tf/(N-1))
    # bounds = np.ones((2, 1))*2
    # bounds[0] *= -1
    # x,u,K = test.solve(test.x0(), u=np.linspace(-20,20,N), N=N, bounds=None, maxIter=14, order=1)
    # x,u,K = test.solve(test.x0(), u=np.linspace(-20,20,N), N=N, bounds=None, maxIter=14, order=2)

    # plt.figure()
    # plt.plot(x.T[0], x.T[1])
    # plt.plot(np.pi, 0, 'kx')
    # plt.plot(-np.pi, 0, 'kx')
    # plt.title("Optimal Swing Up Maneuver")
    # plt.show()

    tf = 10 
    n = 12
    m = 3
    N = 500
    test = Linear(n, m, tf/(N-1))
    x0 = np.random.standard_normal((n,))
    # x0 = np.ones((n,))
    u0 = 0.1*np.random.standard_normal((N, m))
    # u0 = np.ones((N,m))*0.1
    bounds = np.ones((2, m))*0.6
    bounds[0] *= -1
    # x,u,K = test.solve(x0, N, bounds=None, u=u0, maxIter=5, lambda0=0, order=2)  # Unconstrained version
    # x,u,K = test.solve(x0, N, bounds=None, u=u0, maxIter=15, lambda0=1, order=1)  # Unconstrained version
    x,u,K = test.solve(x0, N, bounds=bounds, u=u0, maxIter=10, order=1) 
    x,u,K = test.solve(x0, N, bounds=bounds, u=u0, maxIter=10, order=2) 

    plt.show()