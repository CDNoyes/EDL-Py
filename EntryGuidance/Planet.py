class Planet:
    def __init__(self, name='Mars', rho0=0, scaleHeight=0, model='exp', da=False):

        self.name = name.capitalize()
        self._da = da  # Differential algebraic inputs

        if self.name == 'Mercury':
            self.radius = float('nan')  # equatorial radius, m
            self.omega = float('nan')     # angular rate of planet rotation, rad/s
            self.mu = float('nan')      # gravitational parameter, m^3/s^2
            print('Planet model not yet implemented!')
        elif self.name == 'Venus':
            self.radius = float('nan')
            self.omega = float('nan')
            self.mu = float('nan')

        elif self.name == 'Earth':
            self.radius = 6378.1e3
            self.omega = 7.292115e-5
            self.mu = 3.98600e14

        elif self.name == 'Mars':
            self.radius = 3397e3
            self.omega = 7.095e-5
            self.mu = 4.2830e13

            if model is 'exp':
                self.rho0 = (1+rho0)*0.0158
                self.scaleHeight = (1+scaleHeight)*9354.5
                self.atmosphere = self.__exp_model_mars

            else:
                # Sample MG and store interpolators for density and speed of sound
                self.atmosphere = self.__MG_model_mars

        elif self.name == 'Saturn':
            self.radius = float('nan')
            self.omega = float('nan')
            self.mu = float('nan')

        elif self.name == 'Jupiter':
            self.radius = float('nan')
            self.omega = float('nan')
            self.mu = float('nan')

        elif self.name == 'Uranus':
            self.radius = float('nan')
            self.omega = float('nan')
            self.mu = float('nan')

        elif self.name == 'Neptune':
            self.radius = float('nan')
            self.omega = float('nan')
            self.mu = float('nan')

        else:
            print('Input planet name, '+ self.name +', is not valid')

    def __exp_model_mars(self, h):
        ''' Defines an exponential model of the atmospheric density and local speed of sound as a function of altitude. '''
        if self._da:
            from pyaudi import exp
            scalar=False
            try:
                h[0]
            except:
                scalar=True
                h = [h]
            #Density computation:
            rho = [self.rho0*exp(-hi/self.scaleHeight) for hi in h]

            # Local speed of sound computation:
            coeff = [223.8, -0.2004e-3, -1.588e-8, 1.404e-13]
            a = [sum([c*hi**i for i,c in enumerate(coeff)]) for hi in h]
            if scalar:
                a = a[0]
                rho = rho[0]

        else:
            from autograd.numpy import exp
            #Density computation:
            rho = self.rho0*exp(-h/self.scaleHeight)

            # Local speed of sound computation:
            coeff = [223.8, -0.2004e-3, -1.588e-8, 1.404e-13]
            a = sum([c*h**i for i,c in enumerate(coeff)])

        return rho,a

    def __MG_model_mars(self, h):
        ''' Interpolates data from an MG profile '''
        return self.density(h),self.speed_of_sound(h)

    def updateMG(date=[10,29,2018], latitude=0, longitude=0, dustTau=0, rpscale=0):
        ''' Calls MG '''
        return

    def range(self,lon0,lat0,heading0,lonc,latc,km=False):
        '''Computes the downrange and crossrange between two lat/lon pairs with a given initial heading.'''
        # from numpy import arccos, arcsin, sin, cos, pi, nan_to_num, zeros_like
        from numpy import pi, nan_to_num, zeros_like, real
        import numpy as np
        from numpy import sin,cos,arcsin,arccos
        # from pyaudi import sin, cos,
        from pyaudi import gdual_double as gd
        # from pyaudi import asin as arcsin
        # from pyaudi import acos as arccos

        d13 = arccos(sin(latc)*sin(lat0)+cos(latc)*cos(lat0)*cos(lonc-lon0))
        # if not isinstance(d13, gd) and np.abs(d13) < 1e-4:
        #     return 0,0
        psi12 = heading0
        PHI = np.sign(lonc-lon0)*arccos( (sin(latc) - sin(lat0)*cos(d13))/(cos(lat0)*sin(d13)) )
        psi13 = pi/2 - PHI
        CR = arcsin(sin(d13)*sin(psi12-psi13))
        DR = self.radius*arccos(cos(d13)/cos(CR))
        CR *= self.radius

        if km:
            return DR/1000., CR/1000.
        else:
            return DR, CR

    def coord(self, lon0, lat0, heading0, dr, cr):
        '''Computes the coords of a target a given downrange and crossrange from an initial location and heading.'''
        from numpy import arccos, arcsin, sin, cos, pi
        # from pyaudi import sin, cos
        # from pyaudi import asin as arcsin
        # from pyaudi import acos as arccos

        LF = arccos(cos(dr/self.radius)*cos(cr/self.radius))
        zeta = arcsin(sin(cr/self.radius)/sin(LF))
        lat = arcsin(cos(zeta-heading0+pi/2.)*cos(lat0)*sin(LF)+sin(lat0)*cos(LF))
        lon = lon0 + arcsin(sin(zeta-heading0+pi/2)*sin(LF)/cos(lat))
        return lon, lat


def getDifference(rho0, scaleHeight):
    import numpy as np

    nominal = Planet()
    dispersed = Planet(rho0=rho0, scaleHeight=scaleHeight)

    h = np.linspace(0, 127e3, 1000) # meters
    rho_nom = [nominal.atmosphere(H)[0] for H in h]
    rho_dis = [dispersed.atmosphere(H)[0] for H in h]
    diff = np.array(rho_dis)-np.array(rho_nom)
    perdiff = 100*diff/np.array(rho_nom)
    return perdiff


def compare():
    from itertools import product
    import matplotlib.pyplot as plt
    import numpy as np

    n = 2
    rho0 = np.linspace(-0.20,0.20,n)
    sh = np.linspace(-0.025,0.01,n)
    h = np.linspace(0,127,1000) # kmeters
    plt.figure()
    for rho, s in product(rho0, sh):
        perDiff = getDifference(rho, s)
        plt.plot(h, perDiff, label="\rho={}, \hs={}".format(rho, s))
    plt.legend(loc='best')
    plt.xlabel('Altitude (km)')
    plt.ylabel('Density variation (%)')
    plt.show()


if __name__ == "__main__":
    compare()
