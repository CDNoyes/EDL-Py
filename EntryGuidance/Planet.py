from math import exp


class Planet:
    def __init__(self, name = 'Mars', rho0 = 0, scaleHeight = 0):
        
        self.name = name.capitalize()
        
        if self.name == 'Mercury':
            self.radius = float('nan')  # equatorial radius, m
            self.omega = float('nan')     # angular rate of planet rotation, rad/s
            self.mu = float('nan')      # gravitational parameter, m^3/s^2
            print 'Planet model not yet implemented!'
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
            self.rho0 = (1+rho0)*0.0158
            self.scaleHeight = (1+scaleHeight)*9345.5
        
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
            print 'Input planet name, '+ self.name +', is not valid'
        
    def atmosphere(self, h):
        #Density computation:
            rho0 = self.rho0
            scaleHeight = self.scaleHeight
            rho = rho0*exp(-h/scaleHeight)
        # Local speed of sound computation:
            coeff = [223.8, -0.2004e-3, -1.588e-8, 1.404e-13]
            a = sum([c*h**i for i,c in enumerate(coeff)])
            return rho,a

            
    def range(self,lon0,lat0,heading0,lonc,latc,km=False):
        '''Computes the downrange and crossrange between two lat/lon pairs with a given initial heading.'''
        from numpy import arccos, arcsin, sin, cos, pi, nan_to_num, zeros_like
        
        LF = arccos(sin(latc)*sin(lat0)+cos(latc)*cos(lat0)*cos(lonc-lon0))
        if LF < 1e-5:
            sig = zeros_like(lonc)
        else:
            sig = arcsin(sin(lonc-lon0)*cos(latc)/sin(LF))
        zeta = sig+heading0-pi/2.
        
        DR = nan_to_num(self.radius*arccos(cos(LF)/cos(arcsin(sin(LF)*sin(zeta)))))
        CR = nan_to_num(self.radius*arcsin(sin(LF)*sin(zeta)))
        if km:
            return DR/1000., CR/1000.
        else:
            return DR,CR
        
    def coord(self,lon0,lat0,heading0,dr,cr):
        '''Computes the coords of a target a given downrange and crossrange from an initial location and heading.'''
        from numpy import arccos, arcsin, sin, cos, pi

        LF = arccos(cos(dr/self.radius)*cos(cr/self.radius))
        zeta = arcsin(sin(CR/self.radius)/sin(LF))
        lat = arcsin(cos(zeta-heading0+pi/2.)*cos(lat0)*sin(LF)+sin(lat0)*cos(LF))
        lon = lon0 + arcsin(sin(zeta-heading0+pi/2)*sin(LF)/cos(lat))
        return lon,lat