''' Initial State '''

from numpy import radians, array

def InitialState(full_state=False, **kwargs):
    ''' A simple method to use a consistent nominal initial state across all programs. '''
    
    
    ind = {'radius'     : 0, 
           'r'          : 0,
            
           'lon'        : 1,
           'longitude'  : 1,
           'theta'      : 1,
           
           'lat'        : 2,
           'latitude'   : 2,
           'phi'        : 2,
           
           'vel'        : 3,
           'velocity'   : 3,
           'V'          : 3,
           
           'fpa'        : 4,
           'gamma'      : 4,
            
           'azimuth'    : 5,
           'psi'        : 5,
           
           'rangeToGo'  : 6,
           'rtg'        : 6,
           'range'      : 6,
           
           'mass'       : 7,
           'm'          : 7,
           }
    
    
    r0, theta0, phi0, v0, gamma0, psi0, s0, m0 = (3540.0e3, radians(-90.07), radians(-43.90),
                                                  5505.0,   radians(-14.15), radians(4.99),
                                                  905.65*1e3, 8500)

    x0 = [r0, theta0, phi0, v0, gamma0, psi0, s0, m0]
    
    for key in kwargs:
        x0[ind[key]] = kwargs[key]
        
    if full_state:

        return array([r0, theta0, phi0, v0, gamma0, psi0, s0, m0]*2 + [1,1] + [radians(-30),0])
    else:    
        return array(x0)
        
 
def demo():
    print "Default state:"
    print InitialState(False)
    print "Setting mass and range to go:"
    print InitialState(False, mass=-1000,range=-666)
 
if __name__ == "__main__":
    demo()