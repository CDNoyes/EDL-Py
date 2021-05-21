class EntryVehicle(object):
    '''
    Defines an EntryVehicle class:

    members:
        area - the effective area, m^2
        CD   - a multiplicative offset for the vehicle's drag coefficient
        CL   - a multiplicative offset for the vehicle's lift coefficient

        SRP-related parameters:
        Thrust - the total thrust of the vehicle BEFORE efficiency losses, cant angle considerations, etc, Newtons
        ThrustFactor - mean(cos(cantAngles)*(1-superSonicEfficiencyLosses)), non-dimensional
        Isp - specific impulse, s
        g0 - sea level Earth gravity, used in mass rate of change, m/s^2

    methods:
        mdot(throttle) - computes the mass rate of change based on the current throttle setting
        aerodynamic_coefficients(Mach) - computes the values of CD and CL for the current Mach values
        BC(mass, Mach=24) - computes the vehicle's ballistic coefficient as a function of its mass. Drag coefficient is calculated by default at Mach 24.

    '''

    def __init__(self, area=15.8, CD=0., CL=0., Thrust=60375., Isp=260., ThrustFactor=1.):
        self.area = area
        self.CD = CD
        self.CL = CL

        self.Thrust = Thrust
        self.ThrustFactor = ThrustFactor
        self.ThrustApplied = self.Thrust*self.ThrustFactor
        self.g0 = 9.81
        self.isp = Isp
        self.ve = self.g0*self.isp
        cD,cL = self.aerodynamic_coefficients(24)
        self.LoD = cL/cD

    def mdot(self, throttle):
        """ Returns the mass flow rate for a given throttle setting. """
        return -self.Thrust*throttle/(self.ve)

    def aerodynamic_coefficients(self, M):
        """ Returns aero coefficients CD and CL. Supports ndarray Mach numbers. """
        pD = [2.598e4, -1022.0, -2904.0, 678.6, -44.33, 1.373]
        qD = [1.505e4, 1687.0, -2651.0, 544.1, -34.11, 1]
        pL = [1.172e4, -3654.0, 485.6, -14.61, 0.4192]
        qL = [2.53e4, -7846.0, 1086.0, -28.35, 1]

        num, den = 0, 0
        for i in range(0, len(pD)):
            num += pD[i]*M**i
        for i in range(0, len(qD)):
            den += qD[i]*M**i

        cD = num/den

        num, den = 0, 0
        for i in range(0, len(pL)):
            num += pL[i]*M**i
        for i in range(0, len(qL)):
            den += qL[i]*M**i

        cL = num/den
        return cD*(1+self.CD), cL*(1+self.CL)

    def BC(self, mass, Mach=24):
        return mass/self.area/self.aerodynamic_coefficients(Mach)[0]

class LoDAeroEV(EntryVehicle):
    def __init__(self, LoDFunction, Cd0, **EV_kwargs):
        self.Function = LoDFunction
        self.Cd0 = Cd0
        super().__init__(**EV_kwargs)

    def aerodynamic_coefficients(self, M):
        LoD = self.Function(M)*(1+self.CL)
        Cd = self.Cd0*(1+self.CD)
        Cl = LoD * Cd
        return Cd, Cl

class ConstantAeroEV(EntryVehicle):

    def __init__(self, Cd0, Cl0, **EV_kwargs):
        self.CD0 = Cd0 
        self.CL0 = Cl0 
        super().__init__(**EV_kwargs)

    def aerodynamic_coefficients(self, M):
        return self.CD0*(1+self.CD), self.CL0*(1+self.CL)


if __name__ == "__main__":
    import numpy as np 
    # print(EntryVehicle().aerodynamic_coefficients(np.linspace(1,35,100)))
    print(EntryVehicle().aerodynamic_coefficients(24))
    # print(EntryVehicle().BC(7200))
