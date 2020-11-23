import numpy as np

def PseudoHuber(p, k=0.05):
    """ Linear-like away from 0, quadratic near zero, positive second derivative 

    Can be used as a differentiable approximation to absolute value for small k (<=0.05)

    Wikipedia:
     It combines the best properties of L2 squared loss and L1 absolute loss by 
     being strongly convex when close to the target/minimum and less steep for extreme values.
    
    """
    return np.sqrt(p**2 + k**2) - k

huber = PseudoHuber
smooth_abs = lambda x: PseudoHuber(x, 0.05)

if __name__ == "__main__":      
    import matplotlib.pyplot as plt

    x = np.linspace(-3,3, 1000)
    y = np.abs(x)
    k = [1, 0.1, 0.01]
    yk = np.array([PseudoHuber(x, ki) for ki in k])

    plt.figure()
    plt.plot(x, y, label = 'abs')
    # plt.plot(x, x**2, label = 'x**2')
    for yi,ki in zip(yk,k):
        plt.plot(x, yi, '--', label='Pseudo Huber, k= {}'.format(ki))
    plt.legend()
    plt.grid()
    plt.show()