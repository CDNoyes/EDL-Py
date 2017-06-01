""" Feedback Linearization-based Entry Guidance """

from numpy import sin,cos 

def controller(**kwargs):
    pass 
    
    
    
def drag_dynamics(D, D_dot, g, L, r, V, gamma, rho, scaleHeight):  
    """ Estimates the nonlinear functions a,b such that the second derivative of
        drag with respect to time is given by (a+b*u)
    """

    # CD appears only in terms involving CD_dot which we assume is negligible 

    V_dot = -D-g*sin(gamma)
    g_dot = -2*g*V*sin(gamma)/r
    h_dot = V*sin(gamma)
    rho_dot = -h_dot*rho/scaleHeight

    if D_dot is None: # When using an observer, we used the observed estimate, otherwise we use this model estimate
        D_dot = D*(rho_dot/rho + 2*V_dot/V)

    a1 = D_dot*(rho_dot/rho + 2*V_dot/V) - D*(2*V_dot**2/V**2)
    a2 = -2*D/V*(D_dot+g_dot*sin(gamma))
    a3 = -2*D*g*cos(gamma)**2 * (1/r - g/V**2)
    a4 = D*rho_dot/rho/h_dot*(-g-D*sin(gamma)+V**2/r*cos(gamma)**2)
    a = a1 + a2 + a3 + a4
    

    b1 = -2*D*L*g*cos(gamma)/V**2
    b2 = D*L/h_dot*rho_dot/rho*cos(gamma)
    b = b1+b2

    return a,b
    
def drag_derivatives(u, L, D, g, r, V, gamma, rho, scaleHeight):
    
    V_dot = -D-g*sin(gamma)
    h_dot = V*sin(gamma)
    rho_dot = -h_dot*rho/scaleHeight

    D_dot = D*(rho_dot/rho + 2*V_dot/V)

    a,b = drag_dynamics(D,D_dot,g,L,r,V,gamma,rho,scaleHeight)
    D_ddot = a + b*u 
    
    return D_dot,D_ddot