""" Differential algebra based optimization """
import numpy as np
from pyaudi import gdual_double as gd
from pyaudi import invert_map
from pyaudi import log

import DA as da


def constraint_satisfaction(funs, guess, order=3, xtol=1e-4, max_iter=50, linesearch=False, verbose=True):

    history = [guess.copy()]
    gradient_history = []
    vars = ['x{}'.format(i) for i in range(len(guess))]
    pvars = ['p{}'.format(i) for i in range(len(guess))]

    if verbose: print("(guess)  constraint satisfaction value: {}".format(np.max([fun(guess) for fun in funs])))

    for it in range(max_iter):
        x = [gd(val, var, order) for val,var in zip(guess, vars)]

        con = np.array([fun(x) for fun in funs])
        active = da.const(con, True) > 0
        if not np.any(active):
            print("No active constraints, feasible point found.")
            break
        # if verbose: print("{} active constraints ".format(np.sum(active)))
        g = np.sum(con[active]**2)          # sum of squares of active constraints

        gp = da.differentiate(g, vars)   # constraint^2 gradient
        dgp = gp-da.const(gp, True)         # change in gradient

        try:
            dgp = [xx if isinstance(xx, gd) else gd(xx,name,order) for xx, name in zip(dgp, vars)]
            h = invert_map(dgp)
        except ValueError:
            if verbose: print("Terminating: gradient is zero to working precision. Either:\n  1. Solution is feasible, or\n  2. The gradient is degenerate at the guess.")
            break

        dxy = da.evaluate(h, pvars, [-da.const(gp, True)])[0]
        gradient_history.append(0.2*dxy/np.linalg.norm(dxy))
        if linesearch:
            if False:
                step = 10.
                while True:
                    guess_new = guess + step*dxy
                    con = np.array([fun(guess_new) for fun in funs])
                    if np.all(con <= 0):  # Should also potentially break if step gets too small
                        break
                    if step < 1e-5:
                        print("Stepsize has shrunk below tolerance during line search")
                        break
                    step *= 0.9
            else:
                nsteps = 50  # should always be even so that zero is not an option
                steps = np.linspace(-10, 10, nsteps)
                violation = []
                found = False
                for step in steps:
                    guess_new = guess + step*dxy
                    con = np.array([fun(guess_new) for fun in funs])
                    violation.append(np.linalg.norm(con[con>=0]))
                    if np.all(con <= 0):  # Should also potentially break if step gets too small
                        found = True
                        break
                # should do a check if no feasible point, take that step with the lowest violation to continue improving
                if not found:
                    if verbose: print("...no feasible point found during line search. Attempting to reduce constraint violation")
                    idx = np.argmin(violation)
                    guess_new = guess + steps[idx]*dxy
            guess = guess_new
        else:
            a = 0.01
            b = 0.9
            step = a/np.linalg.norm(dxy) + b
            guess += step*dxy

        history.append(guess.copy())
        if np.linalg.norm(dxy) < xtol:
            if verbose: print("Terminating: change in solution smaller than requested tolerance.")
            break
        if verbose: print("(iter {}) constraint satisfaction value: {:3g}".format(it+1,np.max([fun(guess) for fun in funs])))
    if it == max_iter-1:
        if verbose: print("Max iterations reached")
        return None, history, gradient_history

    return guess, history, gradient_history

def gradient_descent(obj, cons, guess, xtol=1e-4, max_iter=50, verbose=True):
    history = []
    gradient_history = []

    cons = np.asarray(cons)
    vars = ['x{}'.format(i) for i in range(len(guess))]
    pvars = ['p{}'.format(i) for i in range(len(guess))]

    if verbose:
        f0 = obj(guess)
        print("\n(guess) objective value: {}".format(f0))

    alpha = 1.
    k = 1.e2
    gamma = 1
    for it in range(max_iter):
        x = [gd(val, var, 1) for val,var in zip(guess, vars)]

        f = obj(x)

        # con = np.array([fun(x) for fun in cons])
        # cmax = np.argmax(da.const(con, True))
        # active = da.const(con, True) > -1.
        # scale = np.abs(da.const(con, True)).max() * 1.1
        # con = np.array([log(-c/scale) for c in con])
        # con = np.array([log(-fun(x)) for fun in cons]) # paper version
        con = np.array([log(-fun(x)) for fun in cons if x<-1])
        # con = np.array([-log((1-fun(x))**2) for fun in cons]) # my version
        # con = np.array([-1/fun(x) for fun in cons[active]])
        # con = log(-con[cmax])
        alpha *= k
        g = f + np.sum(con)/alpha          # sum of logarithm of constraints scaled and added to original objective

        # g = f + con/alpha          # sum of logarithm of constraints scaled and added to original objective
        dxy = -da.gradient(g, vars)      # gradient

        if np.any(np.isnan(dxy)):
            print("Something went wrong")
            break

        # linesearch if constraints are violated
        step = 1.
        while True:
            guess_new = guess + step*dxy
            con = np.array([fun(guess_new) for fun in cons])
            if np.all(con < 0):
                break
            step *= 0.8
        guess = guess_new

        history.append(guess.copy())

        if np.linalg.norm(step*dxy) < xtol:
            if verbose: print("Terminating: change in solution smaller than requested tolerance.")
            break
        if verbose: print("(iter {}) objective value: {}".format(it+1, obj(guess)))
    return guess, history


def optimize(obj, cons, guess, order=3, xtol=1e-4, ftol=1e-3, max_iter=50, verbose=True):
    """Map inversion based optimization. 

    Reduces to Newton's method if order=2.  
    Assumes a feasible guess.
    """

    history = []
    obj_history = []

    cons = np.asarray(cons)
    assert np.all(np.array([fun(guess) for fun in cons])), 'Initial point is not feasible.'

    vars = ['x{}'.format(i) for i in range(len(guess))]
    pvars = ['p{}'.format(i) for i in range(len(guess))]

    if verbose:
        f0 = obj(guess)
        print("\n(guess)  objective value: {:.3g}".format(f0))

    alpha = 1.
    k = 1.e2
    for it in range(max_iter):
        x = [gd(val, var, order) for val,var in zip(guess, vars)]

        f = obj(x)

        # con = np.array([fun(x) for fun in cons])
        # cmax = np.argmax(da.const(con, True))
        # scale = np.abs(da.const(con, True)).max() * 1.1
        # con = np.array([log(-c/scale) for c in con])
        # active = da.const(con, True) > -1.
        # fcon = np.array([fun(x) for fun in cons])
        # con = np.array([log(-fc)  np.abs(da.const(fc)) for fc in fcon]) # paper version
        # con = np.array([-log((1-fun(x))**2) for fun in cons]) # my version
        # con = np.array([-1/fun(x) for fun in cons[active]])
        # con = log(-con[cmax])
        alpha *= k
        g = f #+ np.sum(con)/alpha          # sum of logarithm of constraints scaled and added to original objective


        gp = da.differentiate(g, vars)      # gradient
        dgp = gp-da.const(gp, True)         # change in gradient

        try:
            h = invert_map(dgp)
        except ValueError:
            if it:
                guess = guess + 0.9*(guess_new-guess)
                continue
            else:
                if verbose: print("Terminating: map inversion failed.")
                break


        dxy = da.evaluate(h, pvars, [-da.const(gp,True)])[0]
        if np.any(np.isnan(dxy)):
            print("Something went wrong")
            break

        # linesearch to optimize subject to constraint satisfaction 

        nsteps = 100  # this many points will be checked, so a balance is needed. 
        steps = np.linspace(-1, 1, nsteps)
        feasible = []
        feval = []
        for step in steps:
            guess_new = guess + step*dxy
            con = np.array([fun(guess_new) for fun in cons])
            fnew = obj(guess_new)
            feasible.append(np.all(con<=0))
            feval.append(fnew)

        feasible = np.array(feasible, dtype=bool)
        feval = np.array(feval)
        try:
            idx = np.argmin(feval[feasible])
        except ValueError: # no feasible point found, bad situation 
            order = int(np.random.choice(range(2,5))) # randomly change the order and try again 
            print("Loss of feasibility - changing order to {}.".format(order))
            continue
        guess_new = guess + steps[feasible][idx]*dxy
        guess = guess_new

        history.append(guess.copy())
        obj_history.append(obj(guess))
        if np.linalg.norm(step*dxy) < xtol or (len(obj_history) > 1 and np.abs(obj_history[-1]-obj_history[-2])<ftol):
            if verbose: print("Terminating: change in solution smaller than requested tolerance.")
            break
        if verbose: print("(iter {}) objective value: {:.3g}".format(it+1, obj_history[-1]))
    return guess, history


def example_2d():
    import matplotlib.pyplot as plt
    from pyaudi import sqrt 

    obj = lambda x: (x[0]-1)**4 + (x[1]-1)**4 + sqrt(x[0])
    f = [lambda x: 1-x[0]**2-x[1]**2, lambda x: x[0]**2+x[1]**2-2, lambda x: -x[0], lambda x: -x[1]]

    # x0 = np.array([0.7,0.5])
    x0 = np.array([0.1, 0.9])

    x_f, h_f, g_f = constraint_satisfaction(f, x0, order=2, xtol=1e-2, max_iter=10, linesearch=True, verbose=True)
    if x_f is not None:
        # x_o, h_gd = gradient_descent(obj, f, x_f,  xtol=1e-4, max_iter=30, verbose=True)
        x_o, h_o = optimize(obj, f, x_f, order=5, xtol=1e-9, ftol=1e-12, max_iter=200, verbose=True)
        print("Opt: {}".format(x_o))
    else:
        h_o = []

    x_f = np.array(h_f)
    x_o = np.array(h_o)
    g_f = np.array(g_f)

    x1 = np.linspace(0, 1.414, 1000)
    x2_1 = np.sqrt(1-x1[x1<=1]**2)
    x2_2 = np.sqrt(2-x1**2)

    # plt.plot(x_f[:,0],x_f[:,1],'bx', label='Feasibility Iterates')

    # if h_o:
    #     plt.plot(x_o[:,0],x_o[:,1],'mo', label='Optimality Iterates')
    # plt.plot(x1[x1<=1], x2_1,'k--', label='Constraints')
    # plt.plot(x1, x2_2, 'k--')
    # plt.plot(1, 1, 'r*', label='True optimum')
    # plt.legend()
    # plt.show()


def example_rosenbrock(a=1, b=100):
    import matplotlib.pyplot as plt

    def fun(x):
        return (a-x[0])**2 + b*(x[1]-x[0]**2)**2

    guess = np.array([-3.,-4.])
    x_o, h_o, = optimize(fun, [], guess, order=5, xtol=1e-6, ftol=1e-6, max_iter=50, verbose=True)
    h_o = np.array(h_o)
    plt.plot(h_o[:,0],h_o[:,1],'mo',label='Optimality Iterates')
    plt.plot(a,a*a,'r*',label='Optimum')
    plt.plot(guess[0], guess[1], 'ko', label='Guess')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # example_rosenbrock()
    example_2d()
