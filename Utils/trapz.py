import numpy as np

def cumtrapz(y,x,initial=0):
    dx = np.diff(x)
    res = np.cumsum(dx*(y[:-1] + y[1:]) / 2.0, axis=0)
    
    return np.concatenate([[initial],res],axis=0)