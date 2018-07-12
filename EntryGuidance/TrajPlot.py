import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 

def TrajPlot(x,y,z, V=None, T=None, ground=True, show=False, figNum=None, lineSpec=None, label=None, axesEqual=True):
    if figNum is None:
        fig = plt.figure()
    else:
        fig = plt.figure(figNum)
        
    ax = fig.gca(projection='3d')

    if lineSpec is not None:
        ax.plot(x, y, z, lineSpec, label=label)
    else:
        ax.plot(x, y, z, 'o-', label=label)
        
    if ground:
        ax.plot(x, y, np.zeros_like(z), 'k--')
        
        
    if V is not None:
        for xi,yi,zi,vi in zip(x,y,z,V):
            ax.plot([xi,xi+vi[0]],[yi,yi+vi[1]],[zi,zi+vi[2]],'k')

    if T is not None:
        for xi,yi,zi,vi in zip(x,y,z,T):
            ax.plot([xi,xi+vi[0]],[yi,yi+vi[1]],[zi,zi+vi[2]],'r')            
        
    if label is not None:
        plt.legend()
    
    if axesEqual:
        set_axes_equal(ax)
        
    if show:
        plt.show()
        
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
      
    Taken from 
    http://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to    
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])