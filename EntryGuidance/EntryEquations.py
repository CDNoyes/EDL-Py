from numpy import sin, cos, tan
import numpy as np
from functools import partial

from EntryVehicle import EntryVehicle
from Planet import Planet
from Filter import FadingMemory

class Entry:
    """  Basic equations of motion for unpowered and powered flight through an atmosphere. """
    
    def __init__(self, PlanetModel = Planet('Mars'), VehicleModel = EntryVehicle(), Coriolis = False, Powered = False):
    
        self.planet = PlanetModel
        self.vehicle = VehicleModel
        self.powered = Powered
        self.drag_ratio = 1
        self.lift_ratio = 1
        

        if Coriolis:
            self.dyn_model = self.__entry_vinhs
        else:
            self.dyn_model = self.__entry_3dof

            
    
    def update_ratios(self,LR,DR):
        self.drag_ratio = DR
        self.lift_ratio = LR
        
    
    def ignite(self):
        """ Ignites the engines to begin powered flight. """
        
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
        L = f*cL*self.lift_ratio
        D = f*cD*self.drag_ratio
                
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
        
    
    def __thrust_3dof(self, x, u):
        r,theta,phi,v,gamma,psi,s,m = x
        sigma,throttle,thrustAngle = u

        return np.array([0,0,0,self.vehicle.ThrustApplied*throttle*cos(thrustAngle-gamma)/m, self.vehicle.ThrustApplied*throttle*sin(thrustAngle-gamma)/(m*v), 0, 0, self.vehicle.mdot(throttle)])
    
    # Utilities
    def altitude(self, r, km=False):
        """ Computes the altitude from radius """
        
        if km:
            return (r-self.planet.radius)/1000.
        else:
            return r-self.planet.radius
            
    def energy(self, r, v, Normalized=True):
        """ Computes the current energy at a given altitude and velocity. """
        
        E = 0.5*v**2 + self.planet.mu/self.planet.radius-self.planet.mu/r**2
        if Normalized:
            return (E-E[0])/(E[-1]-E[0])
        else:
            return E
            
    def aeroforces(self, r, v):
        """  Returns the aerodynamic forces acting on the vehicle at a given altitude and velocity. """
        
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
        
def EDL(InputSample = np.zeros(4)):
    ''' A non-member utility to generate an EDL model for a given realization of uncertain parameters. '''
    
    CD,CL,rho0,sh = InputSample
    return Entry(PlanetModel = Planet(rho0=rho0,scaleHeight=sh), VehicleModel = EntryVehicle(CD=CD,CL=CL))  
    
    
        
class System(object):
    
    """ 
     
     A more complete EDL system with:
        - truth states                                                      0:8
        - integrated navigated state                                        8:16 
        - first order filters for lift and drag correction ratios           16,17
        - integration of the bank angle system                              18,19
        
    """
    
    def __init__(self, InputSample):

        self.model = EDL()
        self.truth = EDL(InputSample=InputSample)
        self.nav   = EDL(InputSample=InputSample) # For now, consider no knowledge error so nav = truth
        self.filter_gain  = 0.0
        self.powered = False
    
    def ignite(self):
        self.model.ignite()
        self.truth.ignite()
        self.nav.ignite()
        self.powered = True
        
    def dynamics(self, u):
        """ Returns an function integrable by odeint """
        return lambda x,t: np.hstack( (self.truth.dynamics((x[18],u[1],u[2]))(x[0:8],t), 
                                       self.nav.dynamics((x[18],u[1],u[2]))(x[8:16],t), 
                                       self.__filterUpdate(x,t),
                                       BankAngleDynamics(x[18:20], u[0]),
                                       ) ) 
    
    def __filterUpdate(self,x,t):
        """ Computes the derivatives of the aerodynamic ratios. """
        RL = x[16]
        RD = x[17]
        
        L,D   = self.model.aeroforces(np.array([x[8]]),np.array([x[11]]))
        Lm,Dm = self.nav.aeroforces(np.array([x[8]]),np.array([x[11]]))
        
        dRL = FadingMemory(currentValue=RL, measuredValue=Lm[0]/L[0], gain=self.filter_gain)
        dRD = FadingMemory(currentValue=RD, measuredValue=Dm[0]/D[0], gain=self.filter_gain)
        
        return np.array([dRL,dRD])
        
        
    def setFilterGain(self, gain):
        """ Sets the gain used in the first order lift and drag filters. """
        self.filter_gain = gain
        
        
def BankAngleDynamics(bank_state, command, kp=0.56, kd=1.3, min_bank=0, max_bank=np.pi/2, max_rate=np.radians(20), max_accel=np.radians(10)):
    """ 
        Constrained second order system subjected to minimum and maximum bank angles, max bank rate, and max acceleration.
        May cause stiffness in numerical integration, consider replacing Saturate with a smoother function like Erf
    """
    bank,rate = bank_state
    # bank = np.sign(bank)*Saturate(np.abs(bank), min_bank, max_bank)
    
    bank_dot = Saturate(rate, -max_rate, max_rate)
    rate_dot = Saturate(kp*(command-bank)-kd*rate, -max_accel, max_accel)

    return np.array([bank_dot, rate_dot])
    
def Saturate(value,min_value,max_value):
    return np.max( (np.min( (value, max_value) ), min_value))
    
def Erf(value, min_value, max_value):
    from scipy.special import erf
    input_value = (value-(max_value+min_value)*0.5)/((max_value-min_value)*0.5)
    unscaled_output = erf(np.sqrt(np.pi)/1.75*input_value)
    return unscaled_output*((max_value-min_value)*0.5) + (max_value+min_value)*0.5
    
def CompareSaturation():
    import matplotlib.pyplot as plt
    x = np.linspace(-2,2)
    
    plt.plot(x,Erf(x,-1.5,1))
    plt.plot(x,[Saturate(xx,-1.5,1) for xx in x])        
    plt.show()
    
if __name__ == "__main__":
    CompareSaturation()