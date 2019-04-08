"""SDC factorizations of atmospheric entry dynamics 
    in spherical coordinates, named by the independent variable 

"""

import numpy as np 
from SDCBase import SDCBase
from replace import safe_divide


class Range(SDCBase):
    """
    x = [h, v, fpa]
        u = cos(sigma)

    w = [0,  0,  1]
        [w1, w2, 1]
        [w1, w2, 0]
    """ 
    @property
    def n(self):
        """ State dimension """
        return 3

    @property
    def m(self):
        """ Control dimension """
        return 3

    def set_weights(self, w):
        # assert np.allclose(np.sum(self.w, axis=1), np.ones((4))), "Row weights must sum to 1"   
        self.w = w 

    def randomize_weights(self):
        r = np.random.random((2, 2))
        w = np.zeros((3,3))
        w[0:2,2] = 1
        T = np.sum(r, axis=1)
        w[1:3,0:2] = r/T[:, None] 
        self.set_weights(w) 


    def __init__(self, model, entry_mass):
        # Default weights:
        w = np.zeros((3,3))
        w[0:2,2] = 1
        w[1:3,0:2] = 0.5 
        self.w = w 
        # self.randomize_weights()
        self.model = model 
        self.mass = entry_mass

    def A(self, t, x):
        h, v, fpa = x
        r = self.model.radius(h*self.model.dist_scale)/self.model.dist_scale    # nd radius
        g = self.model.gravity(r)                                               # nd gravity 
        D = self.model.aeroforces(r*self.model.dist_scale, v*self.model.vel_scale, self.mass)[1]/self.model.acc_scale

        tg_over_fpa = safe_divide(np.tan(fpa), fpa, 1)

        vp = -(D/v/np.cos(fpa))
        df = (1/r-g/v**2)

        Ah = [0, 0, tg_over_fpa]
        Av = [vp/h, vp/v, -g/v*tg_over_fpa]  # This is not the only possibility 
        Af = [df/h, df/v, 0]
        M = np.array([Ah, Av, Af]) * self.w  # Apply the weights 
        return M 

    def B(self, t, x):
        h, v, fpa = x
        r = self.model.radius(h*self.model.dist_scale) 
        L = self.model.aeroforces(r, v*self.model.vel_scale, self.mass)[0]/self.model.acc_scale
        return np.array([[0, 0, L/np.cos(fpa)/v**2]]).T


class Energy(SDCBase):
    """
        Wrt to energy loss, a monotonically increasing variable 
        x = [h, s, v, fpa]
        u = cos sigma (Vertical L/D fraction)

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
        self.w = np.array([wa, wb, wa, wb])
        self.model = model 
        self.mass = entry_mass

    def set_weights(self, w):
        assert np.allclose(np.sum(self.w, axis=1), np.ones((4))), "Row weights must sum to 1"   
        self.w = w 

    def randomize_weights(self):
        r = np.random.random((4, 4))
        r[:, 1] = 0
        r[(1, 3), 3] = 0
        T = np.sum(r, axis=1)
        self.set_weights(r/T[:, None]) 

    def A(self, t, x):
        h, s, v, fpa = x
        r = self.model.radius(h*self.model.dist_scale)/self.model.dist_scale    # nd radius
        g = self.model.gravity(r)                                               # nd gravity 
        D = self.model.aeroforces(r*self.model.dist_scale, v*self.model.vel_scale, self.mass)[1]/self.model.acc_scale

        sg = np.sin(fpa)/D
        cg = np.cos(fpa)/D

        sg_over_fpa = safe_divide(np.sin(fpa), fpa, 1)/D
        cgm1_over_fpa = safe_divide(np.cos(fpa)-1, fpa, 0)/D

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
        return np.array([[0, 0, 0, L/(D*v**2)]]).T

    def C(self, t, x):  
        return np.eye(4)


def verify_energy():
    # Compare true dynamics and SDC factorization
    # Some sign differences must be accounted for:
    # True are wrt energy, SDC wrt energy loss (so opposing in sign)
    # True has rtg, SDC has range flown (opposite only for this state variable)
    import sys 
    sys.path.append("./Utils")
    sys.path.append("./EntryGuidance")

    from EntryEquations import Entry
    from InitialState import InitialState

    x0 = InitialState(fpa=0)
    model = Entry(Energy=True, Scale=True)
    x0 = model.scale(x0)

    idx = [0, 6, 3, 4]  # grabs the longitudinal states in the correct order 
    print("\nEnergy Model: \n")
    print("Longitudinal state: {}".format(x0[idx]))

    sdc_model = Energy(model, x0[-1])
    sdc_model.randomize_weights()
    sigma = 0.2

    dx = model.dynamics([sigma, 0, 0])(x0, 0)  # truth 

    print("True derivatives:   {}".format(-dx[idx]))

    x0_sdc = x0[idx]
    x0_sdc[0] = model.altitude(x0_sdc[0]*model.dist_scale)/model.dist_scale
    dx_sdc = sdc_model.dynamics([np.cos(sigma)])(x0_sdc, 0)

    print("SDC derivatives:    {}".format(dx_sdc))


def verify_range():
    # Compare true dynamics and SDC factorization
    # Some sign differences must be accounted for:
    # True has rtg, SDC has range flown 

    import sys 
    sys.path.append("./Utils")
    sys.path.append("./EntryGuidance")

    from EntryEquations import Entry
    from InitialState import InitialState

    x0 = InitialState(rtg=0, r=15e3 + 3397e3)
    model = Entry(Scale=True)
    x0 = model.scale(x0)

    idx = [0, 3, 4]  # grabs the longitudinal states in the correct order 
    print("\nRange Model: \n")
    print("Longitudinal state: {}".format(x0[idx]))

    sdc_model = Range(model, x0[-1])
    sigma = 0.2

    dx = model.dynamics([sigma, 0, 0])(x0, 0)  # truth 

    print("True derivatives:   {}".format(dx[idx]/-dx[6]))

    x0_sdc = x0[idx]
    x0_sdc[0] = model.altitude(x0_sdc[0]*model.dist_scale)/model.dist_scale
    dx_sdc = sdc_model.dynamics([np.cos(sigma)])(x0_sdc, 0)

    print("SDC derivatives:    {}".format(dx_sdc))


def test_range(randomize=False):
    from scipy.integrate import trapz 
    from matplotlib import pyplot as plt 
    import sys 
    sys.path.append("./Utils")
    sys.path.append("./EntryGuidance")

    from EntryEquations import Entry
    from InitialState import InitialState
    from RK4 import RK4
    from TSDRE import TSDRE 

    x0 = InitialState()
    model = Entry(Scale=False)
    x0 = model.scale(x0)

    idx = [0, 3, 4]  # grabs the longitudinal states in the correct order 
    x0_sdc = x0[idx]
    x0_sdc[0] = model.altitude(x0_sdc[0]*model.dist_scale)/model.dist_scale

    sdc_model = Range(model, x0[-1])   
    if randomize:
        sdc_model.randomize_weights()
    controller = TSDRE()

    # h_target = 8    
    s_range = [720, 800] # for 8 km. The higher the target, the lower the downrange needed to achieve lower velocities 
    # h_target = 3
    # s_range = [785, 885]
    def altitude_constraint(h):
        def constraint(x):
            return x[0]-h, np.array([1, 0, 0])
        return constraint 
    for h_target in [6, 8, 10]:
        for sf in np.linspace(*s_range, num=20):
            problem = {'tf': sf*1000/model.dist_scale, 'Q': lambda y: np.diag([0, 0.01, 0.1])*0, 'R': lambda x: [[1]], 'constraint': altitude_constraint(h_target*1000/model.dist_scale)}

            X = [x0_sdc]
            U = []
            S = [0]

            while True: # 1 second update interval 
                # u = [0.5]
                u = controller(S[-1], X[-1], sdc_model, problem)
                if 1:
                    u = np.clip(u, 0, 1)  # Optional saturation effects
                Snew = S[-1] + X[-1][1]*np.cos(X[-1][2])
                Snew = min(Snew, sf*1000/model.dist_scale)
                S.append(Snew)
                xi = RK4(sdc_model.dynamics(u), X[-1], np.linspace(S[-2], S[-1], 3))  # _ steps per control update 
                X.append(xi[-1])
                U.append(u)
                if np.abs(S[-1]-sf*1000) < 1:
                    break


            # Prepare the outputs     
            U.append(U[-1])
            S = np.array(S)
            x = np.array(X).T * np.array([model.dist_scale, model.vel_scale, 1])[:,None]
            keep = x[0]/1000 > 0 # min altitude 
            # keep = x[1] > 0 
            h, v, fpa = x[:, keep] 
            s = S[keep]/1000*model.dist_scale 
            u = np.array(U)[keep]

            print("Target Altitude = {:.2f} km".format(h_target))
            print("Final Range = {:.2f} km".format(sf))
            print("Final Altitude = {:.2f} km".format(h[-1]/1000))
            print("Final Velocity = {:.1f} m/s\n".format(v[-1]))
            # J = trapz(x=t, y=u.squeeze()**2)

            # Graph the results 
            plt.figure(1)
            plt.plot(s, h/1000)
            plt.xlabel("Range flown (km)")
            plt.ylabel("Altitude (km)")
            plt.figure(2)
            plt.plot(s, v)
            plt.xlabel("Range flown (km)")
            plt.ylabel("Velocity (m/s)")
            plt.figure(3)
            plt.plot(s, np.degrees(fpa))
            plt.xlabel("Range flown (km)")
            plt.ylabel("FPA")

            plt.figure(4)
            plt.plot(s, u)
            plt.xlabel("Range flown (km)")
            plt.ylabel("Control")

            plt.figure(5)
            plt.plot(sf, h_target, 'ko')
            plt.plot(s[-1], h[-1]/1000, 'rx')
            plt.figure(6)
            plt.plot(v[-1], h[-1]/1000, 'rx')
            plt.xlabel("Velocity (m/s)")
            plt.ylabel("Altitude (km)")
    plt.show()


def test_energy():
    from scipy.integrate import trapz 
    from matplotlib import pyplot as plt 
    import sys 
    sys.path.append("./Utils")
    sys.path.append("./EntryGuidance")

    from EntryEquations import Entry
    from InitialState import InitialState
    from RK4 import RK4
    from TSDRE import TSDRE 

    x0 = InitialState(rtg=0)
    model = Entry(Scale=False)
    E0 = model.energy(x0[0], x0[3], False)
    Efmin = model.energy(model.radius(3e3), 300, False)
    Efmax = model.energy(model.radius(11e3), 600, False)
    x0 = model.scale(x0)

    idx = [0, 6, 3, 4]  # grabs the longitudinal states in the correct order 
    x0_sdc = x0[idx]
    x0_sdc[0] = model.altitude(x0_sdc[0]*model.dist_scale)/model.dist_scale
    sdc_model = Energy(model, x0[-1])   
    # sdc_model.randomize_weights()
    controller = TSDRE()

    def range_constraint(s):
        def constraint(x):
            return x[1]-s, np.array([0, 1, 0, 0])
        return constraint 

    # Elist = np.linspace(Efmin, Efmax, num=3)
    Elist = [0.5*(Efmin + Efmax), Efmax]
    slist = np.linspace(660, 825, 8)
    for Ef in Elist:
        for s_target in slist:
            X = [x0_sdc]
            U = []
            Lf = E0-Ef 
            L = [0]
            problem = {'tf': Lf, 'Q': lambda y: np.diag([0, 0, 0.01, 0.1])*0, 'R': lambda x: [[1]], 'constraint': range_constraint(s_target*1000/model.dist_scale)}
            it = 0
            while True:
                D = model.aeroforces(model.radius(X[-1][0]), X[-1][2], sdc_model.mass)[1]/model.acc_scale
                if D*model.acc_scale < 0.1:
                    u = [1]
                else:
                    u = controller(L[-1], X[-1], sdc_model, problem, 10)
                if 1:
                    u = np.clip(u, 0, 1)  # Optional saturation effects
                Lnew = L[-1] + X[-1][2]*D
                Lnew = min(Lnew, Lf)
                xi = RK4(sdc_model.dynamics(u), X[-1], np.linspace(L[-1], Lnew, 3))  # _ steps per control update 
                L.append(Lnew)
                # E = model.energy(model.radius(xi[-1][0]), xi[-1][2], False)
                # print(Lnew)
                # print(E0-E)
                # L.append(E0-E)
                X.append(xi[-1])
                U.append(u)
                it += 1
                if np.abs(L[-1]-Lf) < 0.1:
                    break 
                if it > 500:
                    print("Max iter reached")
                    break


            # Prepare the outputs     
            U.append(U[-1])
            x = np.array(X).T * np.array([model.dist_scale/1000, model.dist_scale/1000, model.vel_scale, 1])[:,None]
            keep = x[0] > 0 # min altitude 
            # keep = x[1] > 0 
            h, s, v, fpa = x[:, keep] 
            u = np.array(U)[keep]
            print("Targeted Range = {:.1f} km".format(s_target))
            print("Final Range = {:.2f} km".format(s[-1]))
            print("Final Altitude = {:.2f} km".format(h[-1]))
            print("Final Velocity = {:.1f} m/s\n".format(v[-1]))
            # J = trapz(x=t, y=u.squeeze()**2)

            # Graph the results 
            plt.figure(1)
            plt.plot(s, h)
            plt.xlabel("Range flown (km)")
            plt.ylabel("Altitude (km)")
            plt.figure(2)
            plt.plot(s, v)
            plt.xlabel("Range flown (km)")
            plt.ylabel("Velocity (m/s)")
            plt.figure(3)
            plt.plot(s, np.degrees(fpa))
            plt.xlabel("Range flown (km)")
            plt.ylabel("FPA")

            plt.figure(4)
            plt.plot(s, u)
            plt.xlabel("Range flown (km)")
            plt.ylabel("Control")

            plt.figure(5)
            plt.plot(s_target*1.09, h[-1], 'ko')
            plt.plot(s[-1], h[-1], 'rx')
            plt.xlabel("Range flown (km)")
            plt.ylabel("Altitude (km)")

            plt.figure(6)
            plt.plot(v[-1], h[-1], 'rx')
            plt.xlabel("Velocity (m/s)")
            plt.ylabel("Altitude (km)")
    plt.show()

if __name__ == "__main__":

    # verify_range()
    test_range()
    # test_range(True)
    
    # verify_energy()
    # test_energy()


