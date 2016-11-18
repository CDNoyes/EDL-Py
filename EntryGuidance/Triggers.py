from functools import partial


class Trigger(object):
    '''
        Although purely functional triggers work, it's nice for them to encapsulate knowledge about themselves such as their type and trigger point
    '''
    def __init__(self, fun, info):
        self.__trigger = fun
        self.__info = info

    def __call__(self, input):
        return self.__trigger(**input)


    def dump(self):
        print self.__info


# Can refactor this a bit: a generic trigger where the name, setpoint, greater than or less than, and units are inputs
        
class VelocityTrigger(Trigger):
    
    def __Trigger(self, velocity, **kwargs):
        return velocity <= self.__vt
    
    def __init__(self,velTrigger):
        self.__vt = velTrigger
        super(VelocityTrigger,self).__init__(self.__Trigger, 'Velocity <= {} m/s'.format(velTrigger))

class AltitudeTrigger(Trigger):

    def __Trigger(self, altitude, **kwargs):       
        return altitude <= self.__at

    def __init__(self,altTrigger):
        self.__at = altTrigger*1000 # Assumed that the trigger is defined in km while the input from the sim will definitely be in meters
        super(AltitudeTrigger,self).__init__(self.__Trigger, 'Altitude <= {} km'.format(altTrigger))    
        
class AccelerationTrigger(Trigger):
    # Can be used with drag, lift, acc magnitude etc, useful for pre-entry
        
    def __Trigger(self, **kwargs):
        return kwargs[self.__name] >= self.__at
    
    def __init__(self, accName, accTrigger):
        self.__at =  accTrigger
        self.__name = accName
        super(AccelerationTrigger,self).__init__(self.__Trigger, '{} >= {} m/s^2'.format(accName.capitalize(),accTrigger))   
    
# class AngularTrigger(Trigger):

class MassTrigger(Trigger):
    def __Trigger(self, mass, **kwargs):
        return mass <= self.__mt
        
    def __init__(self, massTrigger):
        self.__mt = massTrigger
        super(MassTrigger,self).__init__(self.__Trigger, 'Mass <= {} kg'.format(massTrigger))

class TimeTrigger(Trigger):
    def __Trigger(self, time, **kwargs):
        return time >= self.__tt
        
    def __init__(self, timeTrigger):
        self.__tt = timeTrigger
        super(TimeTrigger,self).__init__(self.__Trigger, 'Time elapsed >= {} s'.format(timeTrigger))        
        
class RangeToGoTrigger(Trigger):
    def __Trigger(self,rangeToGo, **kwargs):
        return rangeToGo <= self.__rtg
        
    def __init__(self, rtgTrigger):
        self.__rtg = rtgTrigger
        super(RangeToGoTrigger,self).__init__(self.__Trigger,'Range to go <= {} m'.format(rtgTrigger))
        
# class LogicalTrigger(Trigger):
    # '''
    # A class for combining triggers to form more powerful logics
    # bools - links between triggers, 0 for OR, 1 for AND
    # '''
    # def __init__(self, Triggers, Bools):
        # self.__triggers = Triggers
        # self.__bools = Bools
        
        
def Parachute(alt,vel):
    '''
        Checks if the parachute should be deployed based on whether or not the current altitude (in km) and velocity (in m/s)
        satisfy the parachute's constraints on Mach number and dynamic pressure.
        Outputs:
            Satisfied - bool, true if inside the safe deployment box
            Deploy - bool, true if too low or too slow
    '''
    
    from scipy.interpolate import interp1d
    v = [438.5,487]
    h = [6,7.98]
    
    val = (alt >= h[0]) and (vel >= 310)
    if val and (vel <= v[0]):
        return True,False
        
    elif (vel > v[0]) and (vel < v[1]):
        if (alt >= interp1d(v,h)(vel)):
            v2 = [476.4,v[1]]
            h2 = [16.73,h[1]]
            if vel > v2[0]:
                return (alt<=interp1d(v2,h2)(vel)),False
            else:
                return True,False
        else:
            return False,True
    elif val:
        return False,False
    else:
        return False,True
        
        
def DeployParachute(rangeToGo,alt,vel,velBias = 0):
    Satisfied,forceDeploy = Parachute(alt,vel+velBias)
    return forceDeploy or (rangeToGo <= 0 and Satisfied)
    
    
def ParachuteTest():
    import numpy as np
    import matplotlib.pyplot as plt
    h = np.linspace(5,17)
    v = np.linspace(300,500)
    
    for alt in h:
        for vel in v:
            inside,mustDeploy=Parachute(alt,vel)
            if inside:
                plt.plot(vel,alt,marker = 'o',color='b')
            elif mustDeploy:
                plt.plot(vel,alt,marker = 'x',color='r')
                
            else:
                plt.plot(vel,alt,marker = 'x',color ='b')
            
    plt.show()
 
def DeployParachuteTest():
    import numpy as np
    import matplotlib.pyplot as plt
    h = np.linspace(5,17)
    v = np.linspace(300,500)
    s = [1,-1]
    title = ['Undershoot','Overshoot']
    for sign in s:
        plt.figure(s.index(sign))
        plt.title(title[s.index(sign)])
        for alt in h:
            for vel in v:
                Deploy=DeployParachute(sign,alt,vel)
                
                if Deploy:
                    plt.plot(vel,alt,marker = 'o',color='b')
                else:
                    plt.plot(vel,alt,marker = 'x',color='r')
                
            
    plt.show()
        
        
    

       
def findTriggerPoint(x,t):
    import numpy as np
    try:
        idx = np.where(np.isnan(x[:,0]))[0][0]
        return idx
    except:
        return len(t)
    
class TerminateSimulation(Exception):
    def __init__(self, time):
        self.time = time
    def __str__(self):
        return 'Trigger conditions met. Terminating simulation at time = {0} s.'.format(self.time)