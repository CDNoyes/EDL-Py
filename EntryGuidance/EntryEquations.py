from numpy import sin, cos, tan
import numpy as np
from EntryVehicle import EntryVehicle
from Planet import Planet
from Triggers import DeployParachute

class Entry:
    
    def __init__(self, PlanetModel = Planet('Mars'), VehicleModel = EntryVehicle(), Coriolis = False, DegFreedom = 3, Trigger = DeployParachute):
        self.planet = PlanetModel
        self.vehicle = VehicleModel
        self.trigger = Trigger
        if DegFreedom == 2:
            self.dyn_model = self.__entry_2dof
        elif DegFreedom == 3:
            if Coriolis:
                self.dyn_model =self.__entry_vinhs
            else:
                self.dyn_model = self.__entry_3dof
        else:
            print 'Inapproriate number of degrees of freedom.'
            
    
    
    def dynamics(self, control_fun):
        return lambda x,t: self.dyn_model(x, t, control_fun)
        
    #3DOF, Non-rotating Planet (i.e. Coriolis terms are excluded)
    def __entry_3dof(self, x, t, control_fun):
        
        r,theta,phi,v,gamma,psi,s = x
        
        h = r - self.planet.radius
        
        if self.trigger is not None and self.trigger(s, h/1000., v):
            return np.full(x.shape,np.nan) # Turns out its much easier to find the nans than to find where the solution stops changing
        elif np.isnan(x).any():
            return np.full(x.shape,np.nan) # Removes annoying output due to nan's from the trigger       
        else:

            g = self.planet.mu/r**2

            rho,a = self.planet.atmosphere(h)
            M = v/a
            cD,cL = self.vehicle.aerodynamic_coefficients(M)
            f = 0.5*rho*self.vehicle.area*v**2/self.vehicle.mass
            L = f*cL
            D = f*cD
            sigma = control_fun(x,t)
                    
            dh = v*sin(gamma)
            dtheta = v*cos(gamma)*cos(psi)/r/cos(phi)
            dphi = v*cos(gamma)*sin(psi)/r
            dv = -D - g*sin(gamma)
            dgamma = L/v*cos(sigma) + cos(gamma)*(v/r - g/v)
            dpsi = -L*sin(sigma)/v/cos(gamma) - v*cos(gamma)*cos(psi)*tan(phi)/r
            ds = -v/r*self.planet.radius*cos(gamma)
            return np.array([dh, dtheta, dphi, dv, dgamma, dpsi, ds])

        
    #3DOF, Rotating Planet Model - Highest fidelity
    def __entry_vinhs(self, x, t, control_fun):
        r,theta,phi,v,gamma,psi,s = x
        
        #Coriolis contributions to derivatives:
        dh = 0
        dtheta = 0
        dphi = 0
        dv = 0
        dgamma = 2*self.planet.omega*cos(psi)*cos(phi)
        dpsi =  2*self.planet.omega(tan(gamma)*cos(phi)*sin(psi)-sin(phi))
        ds = 0

        return self.__entry_3dof(x, t, control_fun) + np.array([dh, dtheta, dphi, dv, dgamma, dpsi,ds])
        
    #2DOF, Longitudinal Model
    def __entry_2dof(self, x, t, control_fun):
        r,s,v,gamma = x
        
        g = self.planet.mu/r**2
        h = r - self.planet.radius
        rho,a = self.planet.atmosphere(h)
        M = v/a
        cD,cL = self.vehicle.aerodynamic_coefficients(M)
        f = 0.5*rho*self.vehicle.area*v**2/self.vehicle.mass
        L = f*cL
        D = f*cD
        sigma = control_fun(x,t)
        
        dh = v*sin(gamma)
        ds = v*cos(gamma)
        dv = -D - g*sin(gamma)
        dgamma = L/v*cos(sigma) + cos(gamma)*(v/r - g/v)
    
        return np.array([dh,ds,dv,dgamma])
    
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