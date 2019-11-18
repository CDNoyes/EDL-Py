import numpy as np

from EntryGuidance.EntryVehicle import EntryVehicle
from EntryGuidance.Planet import Planet


class Entry(object):
    """  Basic equations of motion for unpowered and powered flight through an atmosphere. """

    def __init__(self, PlanetModel=Planet('Mars'), VehicleModel=EntryVehicle(), Coriolis=False, Powered=False, Energy=False, Altitude=False, DifferentialAlgebra=False, Scale=False):


        # TODO: Simplify while generalizing the independent variable. By default it should be time, but it could accept an argument which
        # is a function that computes the derivative wrt to the independent variable. Then energy, velocity, altitude, etc could all be used 

        self.planet = PlanetModel
        self.vehicle = VehicleModel
        self.powered = Powered
        self.drag_ratio = 1
        self.lift_ratio = 1
        self.nx = 7  # [r,lon,lat,v,gamma,psi,m]
        self.nu = 3  # bank command, throttle, thrust angle
        self.__jacobian = None  # If the jacobian method is called, the Jacobian object is stored to prevent recreating it each time. It is not constructed by default.
        self.__jacobianb = None
        self._da = DifferentialAlgebra
        self.planet._da = DifferentialAlgebra

        # Non-dimensionalizing the states
        if Scale:
            self.dist_scale = self.planet.radius
            self.acc_scale = self.planet.mu/(self.dist_scale**2)
            self.time_scale = np.sqrt(self.dist_scale/self.acc_scale)
            self.vel_scale = np.sqrt(self.dist_scale*self.acc_scale)
            self.mass_scale = 1
            self._scale = np.array([self.dist_scale, 1, 1, self.vel_scale, 1, 1, 1])

        else:  # No scaling
            self.dist_scale = 1
            self.acc_scale = 1
            self.time_scale = 1
            self.vel_scale = 1
            self._scale = np.array([self.dist_scale, 1, 1, self.vel_scale, 1, 1, 1])

        if Coriolis:
            self.dyn_model = self.__entry_vinhs
        else:
            self.dyn_model = self.__entry_3dof

        self.use_energy = Energy
        if self.use_energy:
            self.dE = None
        else:
            self.dE = 1

        self.use_altitude = Altitude
        if self.use_altitude:
            self.dE = None

    def update_ratios(self, LR, DR):
        self.drag_ratio = DR
        self.lift_ratio = LR

    def DA(self, bool=None):
        if bool is None:
            return self._da
        else:
            self._da = bool
            self.planet._da = bool

    def ignite(self):
        """ Ignites the engines to begin powered flight. """
        self.powered = True

    def dynamics(self, u):
        if self.powered:
            return lambda x,t: self.dyn_model(x, t, u)+self.__thrust_3dof(x, u)

        else:
            return lambda x,t: self.dyn_model(x, t, u)

    # Dynamic Models

    # 3DOF, Non-rotating Planet (i.e. Coriolis terms are excluded)
    def __entry_3dof(self, x, t, u):
        if self._da:
            from pyaudi import sin, cos, tan
        else:
            from numpy import sin, cos, tan

        r,theta,phi,v,gamma,psi,m = x
        sigma, throttle, mu = u

        h = r - self.planet.radius/self.dist_scale

        g = self.gravity(r)

        rho, a = self.planet.atmosphere(h*self.dist_scale)
        M = v*self.vel_scale/a
        cD, cL = self.vehicle.aerodynamic_coefficients(M)
        f = np.squeeze(0.5*rho*self.vehicle.area*(v*self.vel_scale)**2/m)/self.acc_scale  # vel_scale**2/acc_scale = dist_scale 
        L = f*cL*self.lift_ratio
        D = f*cD*self.drag_ratio

        dh = v*sin(gamma)
        dtheta = v*cos(gamma)*cos(psi)/r/cos(phi)
        dphi = v*cos(gamma)*sin(psi)/r
        dv = -D - g*sin(gamma)
        dgamma = L/v*cos(sigma) + cos(gamma)*(v/r - g/v)
        dpsi = -L*sin(sigma)/v/cos(gamma) - v*cos(gamma)*cos(psi)*tan(phi)/r
        # ds = -v/r*self.planet.radius*cos(gamma)*cos(psi)/self.dist_scale
        # ds = v*cos(gamma)
        dm = np.zeros_like(dh)

        if self.use_energy:
            if np.ndim(v) <= 1:
                self.dE = -v*D
            else:
                self.dE = (-v*D)[:, None].T
        if self.use_altitude:
            self.dE = np.tile(dh, (self.nx,))

        return np.array([dh, dtheta, dphi, dv, dgamma, dpsi, dm])/self.dE

    # 3DOF, Rotating Planet Model - Highest fidelity
    def __entry_vinhs(self, x, t, u):
        if self._da:
            from pyaudi import sin, cos, tan
        else:
            from numpy import sin, cos, tan

        r,theta,phi,v,gamma,psi,s,m = x

        # Coriolis contributions to derivatives:
        dh = 0
        dtheta = 0
        dphi = 0
        dv = 0
        dgamma = 2*self.planet.omega*cos(psi)*cos(phi)
        dpsi = 2*self.planet.omega(tan(gamma)*cos(phi)*sin(psi)-sin(phi))
        # ds = 0
        dm = 0

        return self.__entry_3dof(x, t, u) + np.array([dh, dtheta, dphi, dv, dgamma, dpsi, dm])/self.dE

    def __thrust_3dof(self, x, u):
        if self._da:
            from pyaudi import sin, cos, tan
        else:
            from numpy import sin, cos, tan
        r,theta,phi,v,gamma,psi,m = x
        sigma,throttle,thrustAngle = u

        return np.array([0,0,0,self.vehicle.ThrustApplied*throttle*cos(sigma)*cos(thrustAngle-gamma)/m, self.vehicle.ThrustApplied*throttle*sin(thrustAngle-gamma)/(m*v), self.vehicle.ThrustApplied*throttle*cos(thrustAngle-gamma)*sin(sigma)/(cos(gamma)*m*v**2), self.vehicle.mdot(throttle)])/self.dE

    def _bank(self, x):
        """ Internal function used for jacobian of bank rate """

        r, theta, phi, v, gamma, psi, m, sigma, T, mu, sigma_dot = x

        if self.use_energy:
            h = r - self.planet.radius/self.dist_scale
            g = self.gravity(r)
            rho,a = self.planet.atmosphere(h*self.dist_scale)
            M = v*self.vel_scale/a
            cD,cL = self.vehicle.aerodynamic_coefficients(M)
            f = np.squeeze(0.5*rho*self.vehicle.area*(v*self.vel_scale)**2/m)/self.acc_scale
            D = f*cD*self.drag_ratio
            return sigma_dot/(-v*D)
        else:
            return sigma_dot

    def bank_jacobian(self, x, u, sigma_dot):
        # Superior DA approach - much faster
        from Utils import DA as da
        vars = ['r','theta','phi','v','fpa','psi','m','bank','T','mu','bank_rate']
        X = np.concatenate((x,u))
        X = np.append(X, sigma_dot)
        X = da.make(X, vars, 1, array=True)
        F = self._bank(X)
        return da.jacobian([F], vars)


    # Utilities
    def altitude(self, r, km=False):
        """ Computes the altitude from radius """
        if km:
            return (r-self.planet.radius)/1000.
        else:
            return r-self.planet.radius

    def radius(self, h):
            return h + self.planet.radius

    def energy(self, r, v, Normalized=True):
        """ Computes the current energy at a given radius and velocity. """

        E = 0.5*v**2 + self.planet.mu/self.planet.radius-self.planet.mu/r
        if Normalized:
            return (E-E[0])/(E[-1]-E[0])
        else:
            return E

    def scale(self, state):
        """Takes a state or array of states in physical units and returns the non-dimensional verison """
        shape = np.asarray(state).shape
        if len(shape)==1 and shape[0]==self.nx:
            return state/self._scale
        else:
            return state/np.tile(self._scale, (shape[0],1))

    def scale_time(self, time):
        return time/self.time_scale

    def unscale(self, state):
        """ Converts unitless states to states with units """
        shape = np.asarray(state).shape
        if len(shape) == 1 and shape[0] == self.nx:
            return state*self._scale
        else:
            return state*np.tile(self._scale, (shape[0], 1))

    def unscale_time(self, time):
        return time*self.time_scale

    def jacobian(self, x, u, hessian=False, vectorized=True):
        """ Returns the full jacobian of the entry dynamics model. 
            The dimension will be [nx, nx+nu].
        """
        return self._jacobian_pyaudi(x, u, hessian, vectorized)

    def _jacobian_ndt(self, x, u):
        ''' Jacobian computed via numdifftools '''
        if self.__jacobian is None:
            from numdifftools import Jacobian
            self.__jacobian = Jacobian(self.__dynamics(), method='complex')

        state = np.concatenate((x, u))
        if self.use_velocity:
            state = np.concatenate((x[:-1], u, x[-1, None]))

        return self.__jacobian(state)

    def _jacobian_pyaudi(self, x, u, hessian=False, vectorized=False):
        ''' Jacobian computed via pyaudi '''

        da_setting = self.DA()
        self.DA(True)

        from Utils import DA as da
        vars = ['r','theta','phi','v','fpa','psi','m','bank','T','mu']
        if vectorized:
            xu = np.concatenate((x.T, u.T))
        else:
            xu = np.concatenate((x, u))

        X = da.make(xu, vars, 1+hessian, array=True, vectorized=vectorized)
        f = self.__dynamics()(X)
        if hessian:
            J = da.jacobian(f, vars)
            H = da.vhessian(f, vars)
            self.DA(da_setting)
            return J, H
        else:
            J = da.jacobian(f, vars)
            self.DA(da_setting)
            return J

    def __dynamics(self):
        ''' Used in jacobian. Returns an object callable with a single combined state '''

        if self.powered:
            return lambda xu: self.dyn_model(xu[0:self.nx], xu[-1], xu[self.nx:self.nx+self.nu])+self.__thrust_3dof(xu[0:self.nx], xu[self.nx:self.nx+self.nu])
        else:
            return lambda xu: self.dyn_model(xu[0:self.nx], xu[-1], xu[self.nx:self.nx+self.nu])

    def aeroforces(self, r, v, m):
        """  Returns the aerodynamic forces acting on the vehicle at a given radius, velocity and mass. """

        h = r - self.planet.radius
        rho, a = self.planet.atmosphere(h)
        M = v/a
        cD, cL = self.vehicle.aerodynamic_coefficients(M)
        f = 0.5*rho*self.vehicle.area*v**2/m
        L = f*cL*self.lift_ratio
        D = f*cD*self.drag_ratio
        return L, D

    def gravity(self, r):
        """ Returns gravitational acceleration at a given planet radius based on quadratic model 
        
            For radius in meters, returns m/s**2
            For non-dimensional radius, returns non-dimensional gravity 

        """
        return self.planet.mu/(r*self.dist_scale)**2/self.acc_scale



def EDL(InputSample=np.zeros(4), **kwargs):
    ''' A non-member utility to generate an EDL model for a given realization of uncertain parameters. '''

    CD, CL, rho0, sh = InputSample
    return Entry(PlanetModel=Planet(rho0=rho0, scaleHeight=sh), VehicleModel=EntryVehicle(CD=CD, CL=CL), **kwargs)



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
        self.nav   = EDL(InputSample=InputSample) # For now, consider no knowledge error nor measurement error so nav = truth
        self.filter_gain  = -10.0
        self.powered = False
        self._scale = np.concatenate((self.truth._scale, self.nav._scale, np.ones((4,))*self.truth.time_scale))
        self.nx = 20

    def ignite(self):
        self.model.ignite()
        self.truth.ignite()
        self.nav.ignite()
        self.powered = True

    def scale(self, state):
        """Takes a state or array of states in physical units and returns the non-dimensional verison """
        shape = np.asarray(state).shape
        if len(shape)==1 and shape[0]==self.nx:
            return state/self._scale
        else:
            return state/np.tile(self._scale, (shape[0],1))


    def unscale(self, state):
        """ Converts unitless states to states with units """
        shape = np.asarray(state).shape
        if len(shape)==1 and shape[0]==self.nx:
            return state*self._scale
        else:
            return state*np.tile(self._scale, (shape[0],1))

    def dynamics(self, u):
        """ Returns an function integrable by odeint """
        return lambda x,t: np.hstack( (self.truth.dynamics((x[18],u[1],u[2]))(x[0:8],t),
                                       self.nav.dynamics((x[18],u[1],u[2]))(x[8:16],t),
                                       self.__filterUpdate(x,t),
                                       BankAngleDynamics(x[18:20], u[0]),
                                       ) )

    def __filterUpdate(self,x,t):
        from EntryGuidance.Filter import FadingMemory

        """ Computes the derivatives of the aerodynamic ratios. """
        RL = x[16]
        RD = x[17]

        L,D   = self.model.aeroforces(np.array([x[8]]),np.array([x[11]]),np.array([x[15]])) # Model based estimates
        Lm,Dm = self.nav.aeroforces(np.array([x[8]]),np.array([x[11]]),np.array([x[15]]))   # Measurements, possibly subject to measurement noise and attitude knowledge errors (which affect the conversion from imu data to aerodynamic accels)

        dRL = FadingMemory(currentValue=RL, measuredValue=Lm[0]/L[0], gain=self.filter_gain)
        dRD = FadingMemory(currentValue=RD, measuredValue=Dm[0]/D[0], gain=self.filter_gain)

        return np.array([dRL,dRD])


    def setFilterGain(self, gain):
        """ Sets the gain used in the first order lift and drag filters. """
        self.filter_gain = gain


def BankAngleDynamics(bank_state, command, kp=0.56, kd=1.3, min_bank=0, max_bank=np.pi/2, max_rate=np.radians(20), max_accel=np.radians(5)):
    """
        Constrained second order system subjected to minimum and maximum bank angles, max bank rate, and max acceleration.
        May cause stiffness in numerical integration, consider replacing Saturate with a smoother function like Erf
    """
    bank,rate = bank_state
    bank_dot = Saturate(rate, -max_rate, max_rate)
    rate_dot = Saturate(kp*(command-bank)-kd*rate, -max_accel, max_accel)

    return np.array([bank_dot, rate_dot])

def Saturate(value, min_value, max_value):
    return np.max( (np.min( (value, max_value) ), min_value))


def CompareJacobian():
    model = Entry()
    from InitialState import InitialState
    x = InitialState(radius=3450e3, velocity=3500)
    u = [0.3, 0, 0]

    N = 100 # Number of calls to average over

    if vel:
        ind = [0,4,6,7,3]
        print(model.dyn_model(x[ind[:-1]], x[ind[-1]], u=u).shape)
    else:
        ind = range(8)
    # print "state: {}".format(x)
    import time
    Jnum = model.jacobian(x[ind],u)
    t0 = time.time()
    for _ in range(N):
        Jnum = model.jacobian(x[ind],u)
    tnum = time.time()-t0

    model.DA(True)
    t0 = time.time()
    for _ in range(N):
        Jda = model.jacobian_(x[ind],u)
    tda = time.time()-t0

    model.DA(False)
    t0 = time.time()
    for _ in range(N):
        Jauto = model.jacobian_ad(x[ind],u)
    tauto = time.time()-t0
    print("Numerical differencing: {:.5f} s".format(tnum/N))
    print("Autograd package      : {:.5f} s\n".format(tauto/N))
    print("Differential algebra  : {:.5f} s".format(tda/N))

    print("PyAudi is approximately {:.1f}x faster than NumDiffTools".format(tnum/tda))
    print("PyAudi is approximately {:.1f}x faster than AutoGrad\n".format(tauto/tda))

    print("Conclusion: Always use pyaudi when possible!\n")

if __name__ == "__main__":
    CompareJacobian()
