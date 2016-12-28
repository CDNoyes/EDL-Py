''' Initial State '''

from numpy import radians, array

def InitialState(full_state=False):
    ''' A simple method to use a consistent nominal initial state across all programs. '''
    r0, theta0, phi0, v0, gamma0, psi0, s0, m0 = (3540.0e3, radians(-90.07), radians(-43.90),
                                                  5505.0,   radians(-14.15), radians(4.99),
                                                  900*1e3, 8500)

    if full_state:

        return array([r0, theta0, phi0, v0, gamma0, psi0, s0, m0]*2 + [1,1] + [np.radians(-15),0])
    else:    
        return array([r0, theta0, phi0, v0, gamma0, psi0, s0, m0])