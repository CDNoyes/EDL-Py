from math import exp


class Planet:
    def __init__(self, name='Mars', rho0=0, scaleHeight=0, model='exp'):
        
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
            
            if model is 'exp':
                self.rho0 = (1+rho0)*0.0158
                self.scaleHeight = (1+scaleHeight)*9345.5
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
            print 'Input planet name, '+ self.name +', is not valid'
        

    def __exp_model_mars(self, h):
        ''' Defines an exponential model of the atmospheric density and local speed of sound as a function of altitude. '''
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
        
def getDifference(rho0, scaleHeight):
    import numpy as np
    
    nominal = Planet()
    dispersed = Planet(rho0=rho0,scaleHeight=scaleHeight)
    
    h = np.linspace(0,127e3,1000) # meters
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
    for rho,s in product(rho0,sh):
        perDiff = getDifference(rho,s)
        plt.plot(h,perDiff,label="\rho={}, \hs={}".format(rho,s))
    plt.legend(loc='best')
    plt.xlabel('Altitude (km)')
    plt.ylabel('Density variation (%)')
    plt.show()
    
if __name__ == "__main__":
    compare()