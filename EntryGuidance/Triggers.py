from functools import partial

# def IgniteSRP():

    # return


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
        
        
def __VelocityTrigger(rangeToGo,alt,vel,velTrigger=360):
    return vel <= velTrigger

def VelocityTrigger(velTrigger=6): # These return immediately useable triggers when called with or without an argument
    return partial(__VelocityTrigger,velTrigger=velTrigger)
    
def __AltitudeTrigger(rangeToGo,alt,vel,altTrigger):       
    return alt <= altTrigger

def AltitudeTrigger(altTrigger=6):
    return partial(__AltitudeTrigger,altTrigger=altTrigger)
    
def DragTrigger(drag,dragMin = 0.2*9.81): # Useful for pre-entry
    return drag>=dragMin
       
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