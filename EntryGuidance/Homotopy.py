# Defines Homotopy method for solution to OCP
# Reference: A continuation approach to state and adjoint calculation in optimal control

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import root
from scipy.integrate import odeint, trapz
import matplotlib.pyplot as plt


# Outline
    # provide initial guess to form auxiliary problem
    # define shooting function
    # use zero lambda as guess for solution of next zero-finding problem
    # increment continuation parameter

    # each user passed function must accept standard arguments


class Homotopy(object):

    def __init__(self,  fixed_states, fixed_time=False):

        self.fixed = fixed_time
        self.fixed_states = fixed_states

        self.c = np.array([0.0,0.0])          # continuation parameters
        self.c_history = [np.zeros((2))]

        self.sol = None
        self.sol_history = []


    def guess(self, time, state, control):
        """ Provides an initial trajectory from which the homotopy begins.

            The values provided are used to construct the vacuous OCP

        """

        time = np.asarray(time)
        state = np.asarray(state)
        control = np.asarray(control)

        print state.shape
        l = time.shape[0]
        self.n = state.shape[1]
        if len(control.shape)==2:
            self.m = control.shape[1]
        else:
            self.m = 1
        assert state.shape[0] == l
        # assert control.shape[0] == l
        print('Defining an auxiliary problem with {} states, {} controls'.format(self.n,self.m))

        # Optimal final time and state for the OCP(c=0)
        self.T0 = time[-1]
        self.Xf0 = state[-1]

        # First solution is all zero costates and the initial final time
        self.sol = [0]*self.n
        if not self.fixed:
            self.sol.append(time[-1])
        self.sol_history.append(self.sol)

        u = interp1d(time, control, kind='cubic')
        # control interpolated onto a new time horizon
        def u0(t,T):
            return u(t * self.T0/T)

        self.u0 = u0

        # newtime = np.linspace(0,self.T0*1.5,500)
        # plt.plot(time,control)
        # plt.plot(newtime,u0(newtime,newtime[-1]))
        # plt.show()

    def phi0(self, T):
        """ The auxiliary Mayer cost term, whose optimum is T=T0 """
        return 0.5*(T-self.T0)**2

    def L0(self, t, u, T):
        """ The auxiliary Lagrange cost term, whose optimum is the guess control """
        return 0.5 * (u - self.u0(t,T))**2


    def define(self, shooting, fixed_states=[]):
        """ Defines the shooting function of the OCP

        which simply takes inputs (typically initial costates and final time)
        and returns the vector of constraint violations by evaluating the forward
        dynamics, evaluating running and terminal costs, and using the necessary conditions.

        All Mayer functions should be M(xf,tf)
        All Lagrange functions should be L(x,u,t)

        fixed_states is a list of indices of the state variable that are fixed.

        Current assumption is that all initial states are fixed.

        The shooting function must return the following, in the order given,

        x(tf) - x_targeted    # Only for the fixed states
        lambda(tf) - l_nec    # Necessary condition on costates corresponding to the free states
        H(tf) - H_nec         # Necessary condition on Hamiltonian (only for free final time problems)


        """
        assert len(fixed_states) <= self.n
        if len(fixed_states): # Not empty
            assert np.max(fixed_states) < self.n

        fixed_costates = list( set(range(self.n*2)) - set(fixed_states) )


    def solve(self, fun):
        """ Seeks a solution based on Newton iteration to the current subproblem """

        nf = len(self.fixed_states)
        assert nf <= self.n
        if nf: # Not empty
            assert np.max(self.fixed_states) < self.n

        fixed_costates = list( set(range(self.n*2)) - set(self.fixed_states) )
        nl = self.n-nf

        free = not self.fixed

        # fun returns a vector N
        # define C,D such N_c = N*C + D

        for _ in range(9):

            C = np.array([self.c[1]]*nf + [self.c[0]]*(nl+free))
            D = np.concatenate( ( (1-self.c[1])*self.Xf0[self.fixed_states],
                                      np.zeros((nl)),
                                      np.array((1-self.c[0])*(self.sol[-1]), ndmin=1) )
                                      )
            if self.fixed:
                D = D[:-1]

            def subproblem(lambda_tf):
                return fun(lambda_tf)*C + D # TODO this is wrong


            print C
            print D
            sol = root(subproblem, self.sol, tol=1e-6)
            if sol.success:
                print "c1,c2 = {}".format(self.c)
                self.sol_history.append(sol.x)
                self.sol = sol.x
                self.c_history.append(self.c)
                self.c[0] += 0.05

            else:
                break
        print('Final solution: {}'.format(self.sol))
        return self.sol


    def step(self):
        """ Updates the continuation parameter.
        """
        self.c_history.append(self.c)

def test_problem_2():

    g = 9.81

    def dyn(X,t):
        x,y,v,l1,l2,l3 = X

        # u = np.arctan2(-l1*v, l2*v-g*l3)
        # ux = np.sin(u)
        # uy = np.cos(u)
        l = np.sqrt(l1**2+l2**2)
        ux = -l1/l
        uy = -l2/l

        dx = v*ux
        dy = -v*uy
        dv = g*uy

        dl1 = 0
        dl2 = 0
        dl3 = (l1*ux - l2*uy)

        return np.array((dx,dy,dv,dl1,dl2,dl3))

    def problem(l0tf,plot=False):
        l0 = l0tf[:-1]
        tf = l0tf[-1]

        x0 = np.zeros((3))
        xf_tar = np.array([10,-3])

        X0 = np.concatenate((x0,l0), axis=0)
        t = np.linspace(0,tf)
        X = odeint(dyn,X0,t)

        Xf = X[-1]
        xf = Xf[:3]
        lf = Xf[3:]
        xdot = dyn(Xf, 0)[:3]
        Hf = np.dot(lf, xdot)


        if plot:
            H = [np.dot(x[3:], dyn(x,0)[:3]) for x in X]
            v = X.T[2]
            l1 = X.T[3]
            l2 = X.T[4]
            l3 = X.T[5]

            u = np.arctan2(-l1*v, l2*v-g*l3)
            plt.figure()
            plt.plot(t,u)
            # plt.plot(X.T[0],X.T[1])
            plt.figure()
            plt.plot(t,X[:,:3])
            plt.figure()
            plt.plot(t,X[:,3:])
            plt.title('Costates')
            plt.figure()
            plt.plot(t,H)
            plt.title('Hamiltonian')
            plt.show()
        return np.concatenate( ( np.squeeze(xf[:2]-xf_tar), (lf[-1], (Hf+1)) ) )
        # return xf,lf,Hf

    tf = 1

    x0 = np.zeros((6))
    x0[3]=0.01
    x0[5]=0.01
    t = np.linspace(0,tf)
    X = odeint(dyn,x0,t)
    print X.shape
    # x,y,v,l1,l2,l3 = X.T
    # u = np.arctan2(-l1*v, l2*v-10*l3)

    # H = Homotopy(fixed_states=[0,1])
    # H.guess(t,X[:,:3],u)
    guess = [-0.11,0.066,-0.1,1.88]
    sol = root(problem, guess, tol=1e-6)
    print sol
    # sol = H.solve(problem)
    # problem(guess,True)
    problem(sol.x,True)

if __name__ == "__main__":
    t = np.linspace(0,10,200)
    x = np.random.random(size=(t.size,4))
    u = np.sin(2*t)

    # H = Homotopy().guess(t,x,u)
    test_problem_2()
    # test_problem_solution()
