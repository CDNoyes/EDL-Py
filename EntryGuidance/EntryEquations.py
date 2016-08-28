from numpy import sin, cos, tan
import numpy as np
from EntryVehicle import EntryVehicle
from Planet import Planet
from Triggers import AltitudeTrigger
from functools import partial

class Entry:
    
    def __init__(self, PlanetModel = Planet('Mars'), VehicleModel = EntryVehicle(), Coriolis = False, DegFreedom = 3, Powered = False): #Funs to get called on trigger? To ignite for example
        self.planet = PlanetModel
        self.vehicle = VehicleModel
        self.powered = Powered
        
        if DegFreedom == 2:
            self.dyn_model = self.__entry_2dof
        elif DegFreedom == 3:
            if Coriolis:
                self.dyn_model = self.__entry_vinhs
            else:
                self.dyn_model = self.__entry_3dof
        else:
            print 'Inapproriate number of degrees of freedom.'
            
        
    def ignite(self,Trigger=None):
        self.powered = True
            
    
    def dynamics(self, u):
        if self.powered:
            return lambda x,t: self.dyn_model(x, t, u)+self.__thrust_3dof(x, u)

        else:
            return lambda x,t: self.dyn_model(x, t, u)
    
    # Dynamic Models
    
    #3DOF, Non-rotating Planet (i.e. Coriolis terms are excluded)
    def __entry_3dof(self, x, t, u):
        
        r,theta,phi,v,gamma,psi,s,m = x
        sigma,throttle,mu = u
        
        h = r - self.planet.radius

        g = self.planet.mu/r**2

        rho,a = self.planet.atmosphere(h)
        M = v/a
        cD,cL = self.vehicle.aerodynamic_coefficients(M)
        f = 0.5*rho*self.vehicle.area*v**2/self.vehicle.mass
        L = f*cL
        D = f*cD
                
        dh = v*sin(gamma)
        dtheta = v*cos(gamma)*cos(psi)/r/cos(phi)
        dphi = v*cos(gamma)*sin(psi)/r
        dv = -D - g*sin(gamma)
        dgamma = L/v*cos(sigma) + cos(gamma)*(v/r - g/v)
        dpsi = -L*sin(sigma)/v/cos(gamma) - v*cos(gamma)*cos(psi)*tan(phi)/r
        ds = -v/r*self.planet.radius*cos(gamma)
        dm = self.vehicle.mdot(throttle)

        return np.array([dh, dtheta, dphi, dv, dgamma, dpsi, ds, dm])

        
    #3DOF, Rotating Planet Model - Highest fidelity
    def __entry_vinhs(self, x, t, u):
        r,theta,phi,v,gamma,psi,s,m = x
        
        #Coriolis contributions to derivatives:
        dh = 0
        dtheta = 0
        dphi = 0
        dv = 0
        dgamma = 2*self.planet.omega*cos(psi)*cos(phi)
        dpsi =  2*self.planet.omega(tan(gamma)*cos(phi)*sin(psi)-sin(phi))
        ds = 0
        dm = 0
        
        return self.__entry_3dof(x, t, control_fun) + np.array([dh, dtheta, dphi, dv, dgamma, dpsi,ds,dm])
        
    #2DOF, Longitudinal Model
    # def __entry_2dof(self, x, t, control_fun):
        # r,s,v,gamma = x
        
        # g = self.planet.mu/r**2
        # h = r - self.planet.radius
        # rho,a = self.planet.atmosphere(h)
        # M = v/a
        # cD,cL = self.vehicle.aerodynamic_coefficients(M)
        # f = 0.5*rho*self.vehicle.area*v**2/self.vehicle.mass
        # L = f*cL
        # D = f*cD
        # sigma = control_fun(x,t)
        
        # dh = v*sin(gamma)
        # ds = v*cos(gamma)
        # dv = -D - g*sin(gamma)
        # dgamma = L/v*cos(sigma) + cos(gamma)*(v/r - g/v)
    
        # return np.array([dh,ds,dv,dgamma])
    
    def __thrust_3dof(self, x, u):
        r,theta,phi,v,gamma,psi,s,m = x
        sigma,throttle,thrustAngle = u

        return np.array([0,0,0,self.vehicle.ThrustApplied*throttle*cos(thrustAngle-gamma)/m, self.vehicle.ThrustApplied*throttle*sin(thrustAngle-gamma)/(m*v), 0, 0, self.vehicle.mdot(throttle)])
    
    # Utilities
    def altitude(self, r, km=False):
        if km:
            return (r-self.planet.radius)/1000.
        else:
            return r-self.planet.radius
            
    def energy(self, r, v, Normalized=True):
        E = 0.5*v**2 + self.planet.mu/self.planet.radius-self.planet.mu/r**2
        if Normalized:
            return (E-E[0])/(E[-1]-E[0])
        else:
            return E
            
    def aeroforces(self,r,v):
 
        g = self.planet.mu/r**2
        h = r - self.planet.radius
        L = np.zeros_like(h)
        D = np.zeros_like(h)
        for i,hi in enumerate(h):
            rho,a = self.planet.atmosphere(hi)
            M = v[i]/a
            cD,cL = self.vehicle.aerodynamic_coefficients(M)
            f = 0.5*rho*self.vehicle.area*v[i]**2/self.vehicle.mass
            L[i] = f*cL
            D[i] = f*cD
        return L,D