"""SDC factorizations of atmospheric entry dynamics 
    in spherical coordinates, named by the independent variable 

"""

import numpy as np 
from SDCBase import SDCBase


def replace_nan(x, replace):
    """ A useful method for SDC factorizations. """
    if np.isfinite(x):
        return x
    else:
        return replace


class Energy(SDCBase):
    """
        x = [h, s, v, fpa]
        u = L/D cos sigma (Vertical L/D ratio)

        weight matrix must have the form
        [w1 0 w2 w3]
        [w1 0 w2 0] 
        [w1 0 w2 w3]
        [w1 0 w2 0] 
        
        where each row sum must be 1 

    """
    @property
    def n(self):
        """ State dimension """
        return 4

    @property
    def m(self):
        """ Control dimension """
        return 3

    def __init__(self, model, entry_mass):
        # wa = [0.3, 0, 0.3, 0.4]
        wa = [0.7, 0, 0.2, 0.1]
        wb = [0.5, 0, 0.5, 0]
        self.w = np.array( [ wa,
                    wb,
                    wa,
                    wb ])
        self.model = model 
        self.mass = entry_mass

    def set_weights(self, w):
        assert np.allclose(np.sum(sdc_model.w, axis=1), np.ones((4))), "Row weights must sum to 1"   
        self.w = w 

    def A(self, t, x):
        h, s, v, fpa = x
        r = self.model.radius(h*self.model.dist_scale)/self.model.dist_scale    # nd radius
        g = self.model.gravity(r)                                               # nd gravity 
        D = self.model.aeroforces(r*self.model.dist_scale, v*self.model.vel_scale, self.mass)[1]/self.model.acc_scale

        sg = np.sin(fpa)/D
        cg = np.cos(fpa)/D

        sg_over_fpa = replace_nan(np.sin(fpa)/fpa, 1)/D 
        cgm1_over_fpa = replace_nan((np.cos(fpa)-1)/fpa, 0)/D

        df = (1/r-g/v**2)

        #  split into two matrices - the one that gets mult by weights, and the other with 'constant' terms 
        Ah = [sg/h, 0, sg/v, sg_over_fpa]
        As = [1/h/D, 0, 1/v/D, 0]
        Av = [-g*sg/h/v, 0, -g*sg/v**2, -g*sg_over_fpa/v]
        Af = [df/D/h, 0, df/D/v, 0]
        M = np.array([Ah, As, Av, Af]) * self.w  # Apply the weights 

        C = np.zeros((4, 4))
        C[1, 3] = cgm1_over_fpa
        C[2, 2] = -1/v**2
        C[3, 3] = cgm1_over_fpa*df

        return C + M 

    def B(self, t, x):
        h, s, v, fpa = x
        r = self.model.radius(h*self.model.dist_scale) 
        L, D = self.model.aeroforces(r, v*self.model.vel_scale, self.mass) # dont need to scale since we use their ratio anyway 
        return np.array([0, 0, 0, L/D/v**2])

    def C(self, t, x):  
        return np.eye(4)


def verify():
    # Compare true dynamics and SDC factorization
    # Some sign differences must be accounted for:
    # True are wrt energy, SDC wrt energy loss (so opposing in sign)
    # True has rtg, SDC has range flown (opposite only for this state variable)
    import sys 
    sys.path.append("./Utils")
    sys.path.append("./EntryGuidance")

    from EntryEquations import Entry
    from InitialState import InitialState

    x0 = InitialState(rtg=0, r=15e3 + 3397e3)
    model = Entry(Energy=True, Scale=True)
    x0 = model.scale(x0)

    idx = [0, 6, 3, 4]  # grabs the longitudinal states in the correct order 
    print(x0[idx])

    sdc_model = Energy(model, x0[-1])
    assert np.allclose(np.sum(sdc_model.w, axis=1), np.ones((4))), "Row weights must sum to 1"      
    sigma = 0.1 

    dx = model.dynamics([sigma, 0, 0])(x0, 0)  # truth 

    print(-dx[idx])

    x0_sdc = x0[idx]
    x0_sdc[0] = model.altitude(x0_sdc[0]*model.dist_scale)/model.dist_scale
    dx_sdc = sdc_model.dynamics(np.cos(sigma))(x0_sdc, 0)

    print(dx_sdc)

if __name__ == "__main__":

    verify()


