import numpy as np
import matplotlib.pyplot as plt
from .Planet import Planet

def Draw(figure=None, show=False):

    Mars = Planet(name='Mars', rho0=0, scaleHeight=0, model='exp')

    # Dynamic pressure limits
    q1 = 300
    q2 = 800

    # Mach number limits
    M1 = 1.4
    M2 = 2.2

    # Min altitude, meters, artificial limit based on mission req.
    h_min = 6e3

    h = np.linspace(h_min, 18e3,500)

    rho,a = Mars.atmosphere(h)
    Vmax = np.minimum(a*M2,np.sqrt(2*q2/rho))
    Vmin = np.maximum(a*M1,np.sqrt(2*q1/rho))

    for i,Vminmax in enumerate(zip(Vmin,Vmax)):
        if Vminmax[0] >= Vminmax[1]:
            break

    x = np.linspace(Vmin[0],Vmax[0])
    y = np.ones_like(x)*h_min/1000

    h = h[:i]
    Vmax = Vmax[:i]
    Vmin = Vmin[:i]
    if plt.figure is None:
        plt.figure()
    else:
        plt.figure(figure)
    plt.plot(Vmax,h/1000,'k',Vmin,h/1000,'k',x,y,'k')
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Altitude (km)')

    if show:
        plt.show()

if __name__ == "__main__":
    Draw(show=True)
