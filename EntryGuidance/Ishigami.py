from numpy import sin, pi
import numpy as np
import chaospy as cp
import matplotlib.pyplot as plt


def fun(x):
    return (1+0.1*x[2,:]**4)*sin(x[0,:])+7*sin(x[1,:])**2
    
def run():
    U1 = cp.Uniform(-pi, pi)
    U2 = cp.Uniform(-pi, pi)
    U3 = cp.Uniform(-pi, pi)
    U4 = cp.Uniform(-pi, pi)
    J = cp.J(U1,U2,U3,U4)
    
    N = 10000
    samples = J.sample(N,'S')
    print samples.shape
    Y = fun(samples)
    
    # Plot the marginals as well
    n = 6
    val = np.linspace(-pi,pi,n)
    for i in range(samples.shape[0]):
        plt.figure()
        plt.hist(Y,50, normed=True,histtype='step')

        for v in val:
            inputs = samples
            inputs[i,:] = v
            try:
                plt.hist(fun(inputs), 500, normed=True,histtype='step')
            except:
                pass

        plt.axis([-15,20,0,0.25])

    plt.show()
    
if __name__ == "__main__":
    run()