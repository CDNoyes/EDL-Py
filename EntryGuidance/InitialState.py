''' Initial State '''

from numpy import radians, array, copy

def InitialState(full_state=False, vehicle='MSL', **kwargs):
    ''' A simple function to use a consistent nominal initial state across all programs. '''


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
           'heading'    : 5,

           'mass'       : 6,
           'm'          : 6,
           }


    if False: # MSL-like numbers
        r0, theta0, phi0, v0, gamma0, psi0, m0 = (3540.0e3, radians(-90.07), radians(-43.90),
                                                      5505.0,   radians(-14.15), radians(4.99),
                                                      2804)

    elif 'msl' in vehicle.lower(): # MSL-like numbers but heading directly east
            r0, theta0, phi0, v0, gamma0, psi0, m0 = (3540.0e3, radians(0), radians(0),
                                                          5505.0,   radians(-14.15), radians(0),
                                                          2804)
    else:  # Heavy BC, equatorial flight
        r0, theta0, phi0, v0, gamma0, psi0, m0 = (3540.0e3, radians(0), radians(0),
                                                      5500,   radians(-14.5), radians(0),
                                                      8500)

    x0 = [r0, theta0, phi0, v0, gamma0, psi0, m0]

    for key in kwargs:
        try:
            x0[ind[key.lower()]] = kwargs[key]
        except:
            pass

    if 'bank' in kwargs:
        bank = kwargs['bank']
    else:
        bank = radians(-30)

    if full_state:

        return array(x0*2 + [1,1] + [bank,0])
    else:
        return array(x0)


def demo():
    print("Default state:")
    print(InitialState(False))
    print("Setting mass and range to go:")
    print(InitialState(False, mass=-1000,))

if __name__ == "__main__":
    demo()
