import numpy as np

def trapz(y,x):
    dx = np.diff(x)
    res = np.sum(dx*(y[:-1] + y[1:]) / 2.0, axis=0)
    return res 

def cumtrapz(y,x,initial=0):
    dx = np.diff(x)
    res = np.cumsum(dx*(y[:-1] + y[1:]) / 2.0, axis=0)
    
    return np.concatenate([[initial],res],axis=0)
    
    
def test_trapz():
    import matplotlib.pyplot as plt 
    
    t = np.linspace(0,2*np.pi,500)
    x = np.sin(t)
    y = np.cos(t) 
    
    xi = [trapz(y[:i],t[:i]) for i in range(500)]
    plt.plot(t,x)
    plt.plot(t,xi,'--')
    plt.show()
    
if __name__ == "__main__":
    test_trapz() 