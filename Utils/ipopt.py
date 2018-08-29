"""Defines a class for transcribing optimization problems using GEKKO's interface to IPOPT """

from gekko import GEKKO
import numpy as np 
import time 

class Solver:

    def __init__(self):
        self.model = GEKKO(remote=False)    # solve locally instead of using a server, WINDOWS/LINUX only 
        self.model.IMODE = 3                # steady state optimization 

    def solve(self, verbose=False):
        self.model.solve(disp=verbose)

    def Obj(self, obj):
        self.model.Obj(obj)

    def Equation(self, eq):
        self.model.Equation(eq)

    def StateSpace(self, A, X, B, U, D):
        """ Adds the equations for linear dynamics at an array of time steps 
            
            The inputs should have the following dimensions:
                A - N x n x n
                B - N x n x m
                X - N x n
                U - N x m 
                D - N x N
        
        
        """
        DX = D.dot(X)

        for a, x, b, u, dx in zip(A, X, B, U, DX):
            Eqs = a.dot(x) + b.dot(u) - dx
            for Eq in Eqs:
                self.model.Equation(Eq == 0)

    # def Quadrature(self, V, W):
    #     return W.dot(V)

    def create_vars(self, x, lb=None, ub=None):
        if np.ndim(x) == 1:
            return np.array([self.model.Var(value) for value in x])
            
        elif np.ndim(x) == 2:
            return np.array([[self.model.Var(value) for value in row] for row in x])

    def get_values(self, X):
        if np.ndim(X) == 1:
            return np.array([value.value[0] for value in X])
            
        elif np.ndim(X) == 2:
            return np.array([[value.value[0] for value in row] for row in X])
        
       

def test():
    solver = Solver()
    x = solver.create_vars([0.1, 0.9])
    print(x)


    obj = lambda x: (x[0]-1)**4 + (x[1]-1)**4 + solver.model.sqrt(x[0])
    f = [lambda x: 1-x[0]**2-x[1]**2, lambda x: x[0]**2+x[1]**2-2]

    solver.Obj(obj(x))

    solver.Equation(f[0](x)<0)
    solver.Equation(f[1](x)<0)

    solver.solve()
    print('')
    print('Results')
    v = solver.get_values(x)

    print('x1: ' + str(v[0]))
    print('x2: ' + str(v[1]))


    
if __name__ == "__main__":
    test()   
