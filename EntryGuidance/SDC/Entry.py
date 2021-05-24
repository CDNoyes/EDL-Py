"""SDC factorizations of atmospheric entry dynamics 
    in spherical coordinates, named by the independent variable 

"""

import numpy as np 
from SDCBase import SDCBase
from replace import safe_divide

# class EntryBase(SDCBase):

#     @property
#     def m(self):
#         """ Control dimension """
#         return 1

class Time(SDCBase):
    """
        x = [h,s,v,fpa]

            w = [0,  0, w1, w2]
                [0,  0, w1, w2]
                [w1, 0, w2, 1]
                [w1, 0, w2, 0]
    """
    @property
    def n(self):
        """ State dimension """
        return 4

    @property
    def m(self):
        """ Control dimension """
        return 1

    def set_weights(self, w):
        # assert np.allclose(np.sum(self.w[1:3,0:2], axis=1), np.ones((2))), "Row weights must sum to 1"   
        self.w = w 

    # def randomize_weights(self):
    #     r = np.random.random((2, 2))
    #     w = np.zeros((3,3))
    #     w[0:2,2] = 1
    #     T = np.sum(r, axis=1)
    #     w[1:3,0:2] = r/T[:, None] 
    #     self.set_weights(w) 


    def __init__(self, model, entry_mass):
        # Default weights:
        # w = np.zeros((3,3))
        w = np.array([[0,0,0.5,0.5],[ 0, 0, 0.5, 0.5],[0.5, 0, 0.5, 1],[0.5, 0, 0.5, 0]])
        self.w = w 
        # self.randomize_weights()
        self.model = model 
        self.mass = entry_mass

    def A(self, t, x):
        h, s, v, fpa = x
        r = self.model.radius(h*self.model.dist_scale)/self.model.dist_scale    # nd radius
        g = self.model.gravity(r)                                               # nd gravity 
        D = self.model.aeroforces(r*self.model.dist_scale, v*self.model.vel_scale, self.mass)[1]/self.model.acc_scale

        df = (v/r-g/v)*np.cos(fpa)
        sg = safe_divide(np.sin(fpa), fpa, 1)
        cgm1 = safe_divide(np.cos(fpa)-1, fpa, 0)

        Ah = [0, 0, np.sin(fpa), v*sg]
        As = [0, 0, self.w[1,2]*np.cos(fpa) + self.w[1,3], self.w[1,3]*v*cgm1]
        Av = [-D/h, 0, -D/v, -g*sg]  
        Af = [df/h, 0, df/v, 0]
        M = np.array([self.w[0]*Ah, As, self.w[2]*Av, self.w[3]*Af]) # Apply the weights 
        return M 

    def B(self, t, x):
        h, s, v, fpa = x
        r = self.model.radius(h*self.model.dist_scale) 
        L = self.model.aeroforces(r, v*self.model.vel_scale, self.mass)[0]/self.model.acc_scale
        return np.array([[0, 0, 0, L/v]]).T


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
        return 1

    def set_weights(self, w):
        assert np.allclose(np.sum(self.w[1:3, 0:2], axis=1), np.ones((2))), "Row weights must sum to 1"   
        self.w = w 

    def randomize_weights(self):
        r = np.random.random((2, 2))
        w = np.zeros((3, 3))
        w[0:2, 2] = 1
        T = np.sum(r, axis=1)
        w[1:3, 0:2] = r/T[:, None] 
        self.set_weights(w) 


    def __init__(self, model, entry_mass):
        # Default weights:
        w = np.zeros((3,3))
        w[0:2,2] = 1
        w[1:3,0:2] = 0.5 
        # w[1:3,0] = -1
        # w[1:3,1] = 2
        self.w = w 
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
        return 1

    def __init__(self, model, entry_mass):
        wa = [0.3, 0, 0.3, 0.4]

        # wa = [0.0, 0, 0.4, 0.6]
        wb = [0.5, 0, 0.5, 0]
        # wb = [-0.03, 0, 1.03, 0]
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
        # Af = [df/D/h, 0, df/D/v, 0]
        Af = [df*cg/h, 0, df*cg/v, 0] # Only use this if C[3,3] = 0
        M = np.array([Ah, As, Av, Af]) * self.w  # Apply the weights 

        C = np.zeros((4, 4))
        C[1, 3] = cgm1_over_fpa
        C[2, 2] = -1/v**2
        C[3, 3] = cgm1_over_fpa*df*0

        return C + M 

    def B(self, t, x):
        h, s, v, fpa = x
        r = self.model.radius(h*self.model.dist_scale) 
        L, D = self.model.aeroforces(r, v*self.model.vel_scale, self.mass)  # dont need to scale since we use their ratio anyway 
        return np.array([[0, 0, 0, L/(D*v**2)]]).T

    def C(self, t, x):  
        return np.eye(4)


def verify_time():
    # Compare true dynamics and SDC factorization
    # Some sign differences must be accounted for:
    # True has rtg, SDC has range flown (opposite only for this state variable)
    import sys 
    sys.path.append("./Utils")
    sys.path.append("./EntryGuidance")

    from EntryEquations import Entry
    from InitialState import InitialState

    # x0 = InitialState(fpa=0)
    x0 = InitialState(fpa=-0.1, rtg=0, r=15e3+3397e3)
    model = Entry(Energy=False, Scale=False)
    x0 = model.scale(x0)

    idx = [0, 6, 3, 4]  # grabs the longitudinal states in the correct order 
    print("\nTime Model: \n")
    print("Longitudinal state: {}".format(x0[idx]))

    sdc_model = Time(model, x0[-1])
    # sdc_model.randomize_weights()
    sigma = 0.2

    dx = model.dynamics([sigma, 0, 0])(x0, 0)  # truth 

    print("True derivatives:   {}".format(dx[idx]))

    x0_sdc = x0[idx]
    x0_sdc[0] = model.altitude(x0_sdc[0]*model.dist_scale)/model.dist_scale
    dx_sdc = sdc_model.dynamics([np.cos(sigma)])(x0_sdc, 0)

    print("SDC derivatives:    {}".format(dx_sdc))


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

    # x0 = InitialState(fpa=0)
    x0 = InitialState(fpa=-0.1, rtg=0, r=15e3+3397e3)
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
    model = Entry(Scale=False)
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
    import pandas as pd 
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
    sdc_time = Time(model, x0[-1]) 
    if randomize:
        sdc_model.randomize_weights()
        print(sdc_model.w)
    controller = TSDRE()

    # h_target = 8    
    s_range = [720, 825] # for 8 km. The higher the target, the lower the downrange needed to achieve lower velocities 
    # h_target = 3
    # s_range = [785, 885]
    def altitude_constraint(h):
        def constraint(x):
            return x[0]-h, np.array([1, 0, 0])
        return constraint 


    # for h_target in [6, 8, 10]:
    for h_target in [7, 8, 9]:
        for sf in np.linspace(*s_range, num=20):
        # for sf in [775]:
            problem = {'tf': sf*1000/model.dist_scale, 'Q': lambda y: np.diag([0, 0.01, 0.1])*0, 'R': lambda x: [[1]], 'constraint': altitude_constraint((h_target+0.05)*1000/model.dist_scale)}

            X = [x0_sdc]
            U = []
            S = [0]

            while True: # 1 second update interval 
                # u = [0.5]
                u = controller(S[-1], X[-1], sdc_model, problem, integration_steps=20)
                if 1:
                    u = np.clip(u, 0, 1)  # Optional saturation effects
                Snew = S[-1] + X[-1][1]*np.cos(X[-1][2])
                Snew = min(Snew, sf*1000/model.dist_scale)
                if 0:
                    S.append(Snew)
                    xi = RK4(sdc_model.dynamics(u), X[-1], np.linspace(S[-2], S[-1], 3))  # _ steps per control update 
                    X.append(xi[-1])
                else:
                    xc = np.array([X[-1][0], S[-1], X[-1][1], X[-1][2]])
                    xi = RK4(sdc_time.dynamics(u), xc, np.linspace(0, 1, 3))  # _ steps per control update 
                    X.append(xi[-1][[0,2,3]])
                    S.append(xi[-1][1])
                U.append(u)
                if np.abs(S[-1]-sf*1000) < 1 or S[-1]/1000 >= sf:
                    break

            # Prepare the outputs     
            U.append(U[-1])
            S = np.array(S)
            x = np.array(X).T * np.array([model.dist_scale/1000, model.vel_scale, 1])[:, None]
            keep = x[0]/1000 > 0 # min altitude 
            # keep = x[1] > 0 
            h, v, fpa = x[:, keep] 
            s = S[keep]/1000*model.dist_scale 
            u = np.array(U)[keep]

            # data = {'h': h.squeeze(), 'u': u.squeeze(), 's': s.squeeze(), 'fpa': fpa.squeeze(), 'v': v.squeeze()}
            # df = pd.DataFrame(data)
            # df.to_pickle("Range_S805_U0p5.zip")

            print("Target Altitude = {:.2f} km".format(h_target))
            print("Final Range = {:.2f} km".format(sf))
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
            plt.plot(sf, h_target, 'ko')
            plt.plot(s[-1], h[-1], 'rx')
            plt.ylabel("Altitude (km)")
            plt.xlabel("Range flown (km)")

            plt.figure(6)
            plt.plot(v[-1], h[-1], 'rx')
            plt.xlabel("Velocity (m/s)")
            plt.ylabel("Altitude (km)")

            if h[-1] < (h_target - 0.5): # 500 m error allowance, break after
                break

    plt.show()


def test_energy():
    """
        Scaled version does not work - do not use until fixed.
        Dynamics are correct - its the simulation setup that is faulty somehow.

        Further, when given the same bank profile, it returns the same trajectory
        as the range dynamics 
    """
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
    print("Drag at x0: {}".format(model.aeroforces(x0[0], x0[3], x0[-1])[1]))
    Efmin = model.energy(model.radius(3e3), 360, False)
    Efmax = model.energy(model.radius(3e3), 600, False)
    Escale = model.vel_scale**2 
    x0 = model.scale(x0)
    print("Full initial state: {}".format(x0))

    idx = [0, 6, 3, 4]  # grabs the longitudinal states in the correct order 
    x0_sdc = x0[idx]
    x0_sdc[0] = model.altitude(x0_sdc[0]*model.dist_scale)/model.dist_scale
    print("SDC initial state: {}".format(x0_sdc))

    sdc_model = Energy(model, x0[-1])   
    sdc_time = Time(model, x0[-1])
    # sdc_model.randomize_weights()
    controller = TSDRE()

    def range_constraint(s):
        def constraint(x):
            return x[1]-s, np.array([0, 1, 0, 0])
        return constraint 

    def fpa_constraint(f):
        def constraint(x):
            return x[3]-f, np.array([0, 0, 0, 1])
        return constraint 

    Evalidate = [model.energy(model.radius(12.77e3), 680.2, False)]
    svalidate = [805]
    Elist = np.linspace(Efmin, Efmax, num=10)
    # Elist = [0.5*(Efmin + Efmax), Efmax]
    # Elist = [Efmax]
    # Elist = Evalidate 
    slist = svalidate
    # slist = np.linspace(750, 825, 15)
    for Ef in Elist:
        for s_target in slist:
            X = [x0_sdc]
            U = []
            Lf = (E0-Ef)/Escale  
            # print(Lf)
            L = [0]
            T = [0]
            # problem = {'tf': Lf, 'Q': lambda y: np.diag([0, 0, 0, 0])*1., 'R': lambda x: [[1]], 'constraint': range_constraint(s_target*1000/model.dist_scale)}
            problem = {'tf': Lf, 'Q': lambda y: np.diag([0, 0, 0, 0])*1., 'R': lambda x: [[1]], 'constraint': fpa_constraint(np.radians(-9))}
            it = 0
            while True:
                D = model.aeroforces(model.radius(X[-1][0]*model.dist_scale), X[-1][2]*model.vel_scale, sdc_model.mass)[1]/model.acc_scale
                if 0:
                    print("State: {}".format(X[-1]))
                    print("Drag, unscaled = {}".format(D*model.acc_scale))
                    print("Current Energy Loss = {}".format(L[-1]))
                    print("Current Energy Loss Rate = {}\n".format(X[-1][2]*D))

                # u = [0.5]
                u = controller(L[-1], X[-1], sdc_model, problem, 10)
                if 1:
                    u = np.clip(u, 0, 1.)  # saturation effects
                # Lnew = L[-1] + X[-1][2]*D/model.time_scale
                # Lnew = min(Lnew, Lf)
                T.append(T[-1] + 1)
                xi = RK4(sdc_time.dynamics(u), X[-1], np.linspace(T[-2], T[-1], 10))  # _ steps per control update 
                E = model.energy(model.radius(xi[-1][0]*model.dist_scale), xi[-1][2], False)
                Lnew = E0- E
                L.append(Lnew)
                X.append(xi[-1])
                U.append(u)
                it += 1
                if np.abs(L[-1]-Lf) < 0.01 or L[-1] >= Lf:
                    break 
                if it > 5000:
                    print("Max iter reached")
                    break

            print(len(X))
            # Prepare the outputs     
            U.append(U[-1])
            x = np.array(X).T * np.array([model.dist_scale/1000, model.dist_scale/1000, model.vel_scale, 1])[:,None]
            keep = x[0] > 0 # min altitude 
            # keep = x[1] > 0 
            h, s, v, fpa = x[:, keep] 
            u = np.array(U)[keep]
            print("Target Energy = {:.1f}".format(Ef))
            print("Final Energy = {:.1f}".format(model.energy(model.radius(h[-1]*1000), v[-1], False)))
            print("Targeted Range = {:.1f} km".format(s_target))
            print("Final Range = {:.2f} km".format(s[-1]))
            print("Final Altitude = {:.2f} km".format(h[-1]))
            print("Final FPA = {:.2f} deg".format(np.degrees(fpa[-1])))
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

def validate():
    import pandas as pd 
    import matplotlib.pyplot as plt 
    range_data = pd.read_pickle("Range_S805_U0p5.zip")
    s = range_data['s']
    h = range_data['h']
    fpa = range_data['fpa']
    v = range_data['v']

    plt.figure(1)
    plt.plot(s, h, 'o', label="Range")
    plt.xlabel("Range flown (km)")
    plt.ylabel("Altitude (km)")
    plt.legend()
    plt.figure(2)
    plt.plot(s, v, label="Range")
    plt.legend()
    plt.xlabel("Range flown (km)")
    plt.ylabel("Velocity (m/s)")
    plt.figure(3)
    plt.plot(s, np.degrees(fpa), label="Range")
    plt.legend()
    plt.xlabel("Range flown (km)")
    plt.ylabel("FPA")

def range_mc(randomize=False):
    from scipy.integrate import trapz 
    from matplotlib import pyplot as plt 
    import pandas as pd 
    import sys 
    sys.path.append("./Utils")
    sys.path.append("./EntryGuidance")

    from EntryEquations import Entry, EDL 
    from InitialState import InitialState
    from Uncertainty import getUncertainty 
    from RK4 import RK4
    from TSDRE import TSDRE 

    x0 = InitialState()
    model = Entry(Scale=False)
    # x0 = model.scale(x0)

    idx = [0, 3, 4]  # grabs the longitudinal states in the correct order 
    x0_sdc = x0[idx]
    x0_sdc[0] = model.altitude(x0_sdc[0]*model.dist_scale)/model.dist_scale

    controller = TSDRE()

    def altitude_constraint(h):
        def constraint(x):
            return x[0]-h, np.array([1, 0, 0])
        return constraint 

    h_target = 8
    sf = 775 
    problem = {'tf': sf*1000/model.dist_scale, 'Q': lambda y: np.diag([0, 0.01, 0.1])*0, 'R': lambda x: [[1]], 'constraint': altitude_constraint((h_target+0.05)*1000/model.dist_scale)}
    unc = getUncertainty()['parametric']
    p = unc.sample(2000, 'S').T
    MC = []
    for case, delta in enumerate(p):
        # Have to construct both the model used by controller and the integration model.
        # Can model prescient knowledge of uncertainty, or keep the info from the controller by instead using a nominal model 
        print("Running {}".format(case))
        model = EDL(delta)
        sdc_model = Range(model, x0[-1])  
        sdc_time = Time(model, x0[-1]) 

        if randomize:
            sdc_model.randomize_weights()
            print(sdc_model.w)

        X = [x0_sdc]
        U = []
        S = [0]

        while True: 
            u = controller(S[-1], X[-1], sdc_model, problem, integration_steps=20)
            if 1:
                u = np.clip(u, 0, 1)  # Necessary saturation effects
            if 0:
                Snew = S[-1] + X[-1][1]*np.cos(X[-1][2])
                Snew = min(Snew, sf*1000/model.dist_scale)
                S.append(Snew)
                xi = RK4(sdc_model.dynamics(u), X[-1], np.linspace(S[-2], S[-1], 3))  # _ steps per control update 
                X.append(xi[-1])
            else:
                xc = np.array([X[-1][0], S[-1], X[-1][1], X[-1][2]])
                xi = RK4(sdc_time.dynamics(u), xc, np.linspace(0, 1, 3))  # 1 second update interval, _ steps per control update 
                X.append(xi[-1][[0,2,3]])
                S.append(xi[-1][1])
            U.append(u)
            if np.abs(S[-1]-sf*1000) < 1 or S[-1]/1000 >= sf: # within 1 m or having overshot 
                break

        # Prepare the outputs     
        U.append(U[-1])
        S = np.array(S)
        x = np.array(X).T * np.array([model.dist_scale/1000, model.vel_scale, 1])[:, None]
        h, v, fpa = x
        s = S/1000*model.dist_scale 
        u = np.array(U)

        data = {'h': h.squeeze(), 'u': u.squeeze(), 's': s.squeeze(), 'fpa': fpa.squeeze(), 'v': v.squeeze()}
        df = pd.DataFrame(data)
        MC.append(df)
        # df.to_pickle("Range_S805_U0p5.zip")

        print("Target Altitude = {:.2f} km".format(h_target))
        print("Final Range = {:.2f} km".format(sf))
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
        plt.plot(sf, h_target, 'ko')
        plt.plot(s[-1], h[-1], 'rx')
        plt.ylabel("Altitude (km)")
        plt.xlabel("Range flown (km)")

        plt.figure(6)
        plt.plot(v[-1], h[-1], 'rx')
        plt.xlabel("Velocity (m/s)")
        plt.ylabel("Altitude (km)")

    # import pdb 
    # pdb.set_trace()
    dfp = pd.DataFrame(p, columns=['cd','cl','rho0','sh'])
    DF = pd.concat(MC, axis=1)
    # print(DF)
    dfp.to_pickle("SDCRE_MC_inputs_{}".format(p.shape[0]))
    DF.to_pickle("SDRE_MC_{}".format(p.shape[0]))

    plt.show()

if __name__ == "__main__":

    # verify_time()
    # verify_range()
    # validate()
    # test_range()
    # test_range(True)
    range_mc()
    
    # verify_energy()
    # test_energy()


""" Adaptive use of TASRE for Propellant Optimal EG 

    TASRE can hit certain (h, s) pairs within the vehicle's reachable set
    The vehicle arrives with an undetermined (V, gamma)

    Goal: Determine an appropriate target state (h_target, s_target)
    such that under TSDRE control, the terminal state
    (h, s, v, fpa) is approximately equal to
    an optimal state (h*(v, fpa), s*(v, fpa))

    It may not be possible to reach exactly.

    How to adapt? 
    For fixed altitude, longer downrange means lower velocity (monotnic relationships)
    We can also alter the altitude target to achieve a different FPA 


    Integrate to s_target

"""