

import numpy as np 
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt 


def compare(x1,y1,x2,y2):
    """ Compares two quantities that each have the same independent variable but not necessarily at the same values or the same number of points 
        e.g. compare two drag profiles as a function of energy
    """
    
    # Union of IVs
    xmin = np.min((x1.min(),x2.min()))
    xmax = np.max((x1.max(),x2.max()))
    
    # Intersection of IVs
    xl = np.max((x1.min(),x2.min()))
    xu = np.min((x1.max(),x2.max()))
    
    N = np.max((y1.size,y2.size))
    
    X = np.linspace(xl,xu,N)
    Y1 = interp1d(x1,y1)
    Y2 = interp1d(x2,y2)
    
    E = Y2(X)-Y1(X)
    
    fig,axes = plt.subplots(2,sharex=True)
    axes[0].plot(x1,y1,'b')
    axes[0].plot(x2,y2,'r')
    axes[0].set_title("Original curves")
    
    axes[1].plot(X,E)
    axes[1].set_title("Error on the common domain (red-blue)")
    plt.show()
    
if __name__ == "__main__":
    x1 = np.linspace(-1,0.9)    
    x2 = np.linspace(0.1,1.2)    
    y1 = np.sin(3*x1)
    y2 = np.sin(2.*x2)
    compare(x1,y1,x2,y2)
    