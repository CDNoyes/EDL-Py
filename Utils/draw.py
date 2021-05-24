import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2        # Used to compute confidence intervals

# TODO: This module needs a better name, as do the methods 


def trajectory1d(x, y, V, nstd=3, plot_kws={'label': 'mean'}, fill_kws={'alpha': 0.25, 'color': 'k'}):
    """ Plots the mean trajectory with nstd deviations """

    std = np.sqrt(V)*nstd 

    plt.plot(x, y, **plot_kws)
    plt.fill_between(x, y-std, y+std, **fill_kws)
    plt.legend()


def trajectory2d(X, Y, C, confidence_intervals=[0.95], plot_kws={'label': 'mean'}, downsample=1, fignum=None):
    """ Plots the 2d state trajectory with covariance ellipses """
    if fignum is None:
        plt.gcf().number
    for x, y, P in zip(X[::downsample], Y[::downsample], C[::downsample]):
        cov([x, y], P, ci=confidence_intervals, fignum=fignum, show=False, legend=False)

    plt.plot(X, Y, **plot_kws)


def cov(mean, cov, ci=[0.99], fignum=None, show=False, legend=True, xlabel='', ylabel='', legtext=None, linespec=None):
    """
    Given a 2-D covariance matrix (possibly a submatrix of a larger covariance matrix),
    draws the ellipse under some assumptions.
    """
    eigvals, eigvecs = np.linalg.eig(cov)

    major = np.max(eigvals)
    imax = np.argmax(eigvals)
    minor = np.min(eigvals)
    angle = np.arctan2(eigvecs[1,imax],eigvecs[0,imax])
    ellipse(mean, np.sqrt(major), np.sqrt(minor), angle, ci=ci,fignum=fignum,show=show,legend=legend,xlabel=xlabel,ylabel=ylabel,legtext=legtext,linespec=linespec)


def ellipse(center, major, minor, rotation, ci=[0.95], fignum=None, show=False, legend=True, xlabel='', ylabel='', legtext=None, linespec=None):
    """
        Draws an ellipse centered at center, with major-axis equal to major,
        minor-axis equal to minor, and rotated counter-clockwise from the x-axis
        by an angle specified by rotation (in radians)
    """
    t = np.linspace(-np.pi, np.pi, 100)
    if linespec is None:
        linespec = 'k--'

    if fignum is None:
        plt.figure()
    else:
        plt.figure(fignum)

    for interval in ci:
        s = chi2.isf(1-interval, 2)

        xy = np.array([major*np.sqrt(s)*np.cos(t), np.sqrt(s)*minor*np.sin(t)])
        R = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])
        XY = np.dot(R, xy)
        XY[0,:] += center[0]
        XY[1,:] += center[1]
        if legtext is None:
            label = "CI={}".format(interval)
        else:
            label = "{}, CI={}".format(legtext, interval)

        plt.plot(XY[0,:], XY[1,:], linespec, label=label, linewidth=4)

    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    if legend:
        plt.legend()

    if show:
        plt.show()


def test_cov():
    """ Draws the confidence intervals for a given covariance matrix and samples a large number of points and shows that they fall inside the ellipse """
    # from scipy.stats import multivariate_normal as normal
    from numpy.random import multivariate_normal as normal

    P = np.array([[4,-2],[-2,4]])

    data = normal(mean=[1,-1],cov=P,size=1000)

    cov([1,-1],P,ci=[0.9],linespec='b-', fignum=1, legtext="90%")
    cov([1,-1],P,ci=[0.95],linespec='m-', fignum=1, legtext="95%")
    cov([1,-1],P,ci=[0.99],linespec='r-', fignum=1, legtext="99%")
    plt.plot(data[:,0],data[:,1],'o', alpha=0.25)
    plt.show()


def test_trajectory1d():
    t = np.linspace(0,5)
    x = np.sin(t) + np.cos(3*t)/2
    v = 1 + np.random.random(t.shape)**2*x**2
    trajectory1d(t, x, v, plot_kws={'label':'test','c':'k'}, fill_kws={'alpha': 0.3,'label': "+/- 3-sigma"})
    plt.show()


def test_trajectory2d():
    t = np.linspace(0,5, 50)
    x = np.sin(t) + np.cos(3*t)/2
    y = np.tan(t) + np.cosh(x)
    P = [np.eye(2)/(20+200*ti)+0.001*np.random.random((2,2)) for ti in t]
    trajectory2d(t, x, P, confidence_intervals=[0.90], plot_kws={'label': 'test', 'c': 'r', 'linewidth': 2})
    plt.show()    

if __name__ == "__main__":

    # test_cov()
    # test_trajectory1d()
    test_trajectory2d()
