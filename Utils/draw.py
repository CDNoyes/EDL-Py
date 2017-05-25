import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import chi2        # Used to compute confidence intervals 

def cov(mean, cov, ci=[0.99], fignum=None, show=False, legend=True, xlabel='',ylabel=''):
    """
    Given a 2-D covariance matrix (possibly a submatrix of a larger covariance matrix), 
    draws the ellipse under some assumptions.
    """   
    eigvals,eigvecs = np.linalg.eig(cov)
    
    major = np.max(eigvals)
    imax = np.argmax(eigvals)
    minor = np.min(eigvals)
    angle = np.arctan2(eigvecs[1,imax],eigvecs[0,imax])
    ellipse(mean, np.sqrt(major), np.sqrt(minor), angle, ci=ci,fignum=fignum,show=show,legend=legend,xlabel=xlabel,ylabel=ylabel)
    

    
def ellipse(center, major, minor, rotation, ci = [0.95], fignum=None, show=False, legend=True, xlabel='',ylabel=''):
    """ 
        Draws an ellipse centered at center, with major-axis equal to major,
        minor-axis equal to minor, and rotated counter-clockwise from the x-axis 
        by an angle specified by rotation (in radians)
    """
    t = np.linspace(-np.pi,np.pi,100)
        
    if fignum is None:
        plt.figure()
        
    for interval in ci:
        s = chi2.isf(1-interval,2)
        
        xy = np.array([major*np.sqrt(s)*np.cos(t), np.sqrt(s)*minor*np.sin(t)])
        R = np.array([[np.cos(rotation), -np.sin(rotation)],[np.sin(rotation),np.cos(rotation)]])
        XY = np.dot(R,xy)
        XY[:,0] += center[0]
        XY[:,1] += center[1]
        
        plt.plot(XY[0,:],XY[1,:],'--',label="CI={}".format(interval))
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend:
        plt.legend()
    
    if show:
        plt.show()
    
    
    
def test_cov():
    """ Draws the confidence intervals for a given covariance matrix and samples a large number of points and shows that they fall inside the ellipse """
    # from scipy.stats import multivariate_normal as normal 
    from numpy.random import multivariate_normal as normal 
    
    P = np.array([[4,-2],[-2,4]]) 
    
    data = normal(mean=[0,0],cov=P,size=1000)

    cov([0,0],P,ci=[0.9,0.95,0.99])
    plt.plot(data[:,0],data[:,1],'o')
    plt.show()
    
    
if __name__ == "__main__":
    # ellipse([0,0],1,0.5,0.0,ci=[0.9,0.99],show=True)
    # cov([0,0],[[16,1],[1,4]],ci=[0.99],show=True)
    test_cov()