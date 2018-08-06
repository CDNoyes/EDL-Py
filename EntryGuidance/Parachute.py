import numpy as np
import matplotlib.pyplot as plt
from .Planet import Planet


def Draw(figure=None, show=False, label=False, figsize=None, fontsize=16):
    plt.rc('font', family='serif')

    Mars = Planet(name='Mars', rho0=0, scaleHeight=0, model='exp')

    # Dynamic pressure limits
    q1 = 300
    q2 = 800

    # Mach number limits
    M1 = 1.4
    M2 = 2.2

    # Min altitude, meters, artificial limit based on mission req.
    h_min = 6e3

    h = np.linspace(h_min, 18e3, 500)

    rho, a = Mars.atmosphere(h)
    Vmax = np.minimum(a*M2, np.sqrt(2*q2/rho))
    Vmin = np.maximum(a*M1, np.sqrt(2*q1/rho))

    for i, Vminmax in enumerate(zip(Vmin, Vmax)):
        if Vminmax[0] >= Vminmax[1]:
            break

    x = np.linspace(Vmin[0], Vmax[0])
    y = np.ones_like(x)*h_min/1000

    h = h[:i]
    Vmax = Vmax[:i]
    Vmin = Vmin[:i]
    if plt.figure is None:
        plt.figure(figsize=figsize)
    else:
        plt.figure(figure, figsize=figsize)

    plt.plot(Vmax, h/1000, 'k', Vmin, h/1000, 'k', x, y, 'k')
    plt.xlabel('Velocity (m/s)', fontsize=fontsize)
    plt.ylabel('Altitude (km)', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    if label:
        # plt.rc('text', usetex=True)
        ax = plt.gca()
        ax.annotate('h > {}'.format(h_min/1000), xy=(370, 0.25 + h_min/1000), fontsize=fontsize)

        ax.annotate('M > {}'.format(M1), xy=(313, 1.2+h_min/1000), fontsize=fontsize)
        ax.annotate('M < {}'.format(M2), xy=(445, 5 + h_min/1000), fontsize=fontsize)
        ax.annotate('q < {}'.format(q2), xy=(440, 1.5 + h_min/1000), fontsize=fontsize)
        ax.annotate('q > {}'.format(q1), xy=(370, 11.5), fontsize=fontsize)

    if show:
        plt.show()


if __name__ == "__main__":
    Draw(show=True, label=True)
