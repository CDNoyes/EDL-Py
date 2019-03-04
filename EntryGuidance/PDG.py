""" Defines classes for powered descent guidance 

    Each guidance should define a trigger (can be simple)
    as well as the algorithm for commanding the vehicle 


"""

import numpy as np 
import matplotlib.pyplot as plt 
import abc 


class PoweredDescentGuidanceBase(abc.ABC):

    def __init__(self):
        self.lit = False  
        self.g0 = 9.81 
        self.debug = False 

    def __call__(self, *args, **kwargs):

        if self.lit:
            # call the user defined guidance method 
            if self.debug:
                print("Calling guidance")
            return self.guidance(*args, **kwargs)
        else:
            # call the user defined trigger
            if self.debug:
                print("Calling trigger")
            return self.trigger(*args, **kwargs)

    def trigger(self, *args, **kwargs):
        raise NotImplementedError

    def guidance(self, *args, **kwargs):
        raise NotImplementedError

    def ignite(self):
        self.lit = True 


class JBG(PoweredDescentGuidanceBase):

    def __init__(self, xf=[0, 0, 0, 0]):
        super().__init__()
        self._target = np.asarray(xf) 
        self.debug = True

    def set_target(self, xf):
        assert len(xf) == 4, "Expected four final states for the algorithm target "
        self._target = xf

    def set_constants(self, g=3.71, k=1, Isp=290, Tmin=40*8500, Tmax=70*8500):     
        self.g = g
        self.k = k 
        self.Isp = Isp 
        self.Tmin = Tmin 
        self.Tmax = Tmax 

    def trigger(self, state):
        x0, y0, u0, v0, m0 = state 
        xf, yf, uf, vf = self._target
        
        du = uf-u0
        dv = vf-v0
        tf, mu = self.solve(du, dv, m0)
        self.lit = y0+self.altitude(tf, mu, u0, v0, m0) <= yf 
        return self.lit 

    def guidance(self, state):
        x0, y0, u0, v0, m0 = state 
        xf, yf, uf, vf = self._target

        du = uf-u0
        dv = vf-v0
        tf, mu = self.solve(du, dv, m0)
        return self.Tmax, mu 

    def solve(self, du, dv, m0):

        tf = (du**2 + dv**2)**0.5 / (self.Tmax/m0 - self.g)  # Will be further when the true ToF is longer (lower thrust)
        for _ in range(6):  
            mu = self.thrust_angle(tf, du, dv)
            tf = self.final_time(mu, du, dv, m0)
            if self.debug:
                print("tf = {:.5f}, mu = {:.3f}".format(tf, np.degrees(mu)))
        return tf, mu 

    def final_time(self, mu, du, dv, m0):
        return self.Isp*self.g0*m0*(1-np.exp(-du/(self.k*np.cos(mu)*self.Isp*self.g0)))/self.Tmax 

    def thrust_angle(self, tf, du, dv):
        return np.arctan2(dv+self.g*tf, du)

    def downrange(self, tf, mu, u0, v0, m0):
        mf = m0 - self.Tmax/self.Isp/self.g0*tf 
        return u0*tf + (self.Isp*self.g0)**2/self.Tmax*self.k*np.cos(mu)*(m0-mf*(1+np.log(m0/mf)))

    def altitude(self, tf, mu, u0, v0, m0):
        mf = m0 - self.Tmax/self.Isp/self.g0*tf 
        return v0*tf - 0.5*self.g*tf**2 + (self.Isp*self.g0)**2/self.Tmax*self.k*np.sin(mu)*(m0-mf*(1+np.log(m0/mf)))

    def plot_manifold(self, m0=8500):
        """ The manifold is 4D in the 2D case (and may still be thought of this way in 3D)
            because the initial mass is assumed constant
            (x,y,u,v,m) s.t. constant thrust/angle delivers the vehicle to the target 
            Assumes target is zero altitude 
        """
        from itertools import product

        self.ignite()
        debug = self.debug 
        self.debug = False 
        U0 = np.linspace(4, 700, 100)
        V0 = np.linspace(-1, -250, 100)
        UV = np.array(list(product(U0, V0)))
        DR = []
        H = []
        Mf = []
        Tf = []
        Mu = []
        for u0, v0 in UV:
            state = [0,0,u0,v0,m0]
            _, mu = self(state)
            tf = self.final_time(mu, -u0,-v0, m0)
            dr = self.downrange(tf, mu, u0, v0, m0)
            h = self.altitude(tf, mu, u0, v0, m0)
            mf = self.Tmax/self.Isp/self.g0*tf 

            DR.append(-dr)
            H.append(-h)
            Mf.append(mf)
            Tf.append(tf)
            Mu.append(mu)

        # Mf = np.array(Mf)
        # keep = Mf <= total_prop_mass 

        # C = np.linalg.lstsq(UV, np.array([DR, H]).T)[0]
        # D_approx = np.dot(UV, C).T
        aV = np.arctan2(UV.T[1], UV.T[0])
        aD = np.arctan2(H, DR)

        L = np.polyfit(aV, Mu, 1)
        fpa = aV 
        keep = np.degrees(fpa) > -25
        
        DR = np.array(DR)
        Mf = np.array(Mf)
        H = np.array(H)

        plt.figure(figsize=(14, 6))
        plt.suptitle("Optimal Ignition Surface")
        plt.subplot(1,2,1)
        plt.scatter(UV[keep, 0], UV[keep, 1], c=Mf[keep])
        plt.axis("equal")
        plt.xlabel("Horizontal Velocity at Ignition")
        plt.ylabel("Vertical Velocity at Ignition")

        plt.subplot(1,2,2)
        plt.scatter(DR[keep], H[keep], c=Mf[keep])
        cbar = plt.colorbar()
        cbar.set_label('Prop Used', rotation=270)
        # plt.scatter(D_approx[0], D_approx[1], c='k')
        plt.xlabel("Optimal Horizontal Distance")
        plt.ylabel("Optimal Altitude")
        plt.axis("equal")

        plt.figure()

        print("mu - fpa fit coeff: {}".format(L))
        plt.plot(np.degrees(aV), np.degrees(aD), label="Glideslope Angle")  # i.e. between Altitude and Downrange
        plt.plot(np.degrees(aV), np.degrees(Mu), label="Optimal Thrust Angle $\mu$")
        plt.plot(np.degrees(aV), np.degrees(np.polyval(L, aV)), label="Linear Fit to Optimal Thrust Angle")
        # plt.plot(np.degrees(aV), 180 + np.degrees(aV), 'o', label="180 + FPA")
        plt.xlabel("Ignition Flight Path Angle (deg)")
        plt.ylabel("Angle (deg)")
        plt.legend()
        # plt.plot(np.linalg.norm(UV, axis=1), Mf)
        # plt.plot(np.linalg.norm(UV, axis=1), np.linalg.norm(np.array([DR, H]))


        tf_approx1 = np.polyfit(np.linalg.norm(UV, axis=1), Tf, 1)
        tf_approx2 = np.polyfit(np.linalg.norm(UV, axis=1), Tf, 2)
        tf_constant_mass = np.linalg.norm(UV, axis=1)/(self.Tmax/m0 - self.g)

        plt.figure()
        plt.plot(np.linalg.norm(UV, axis=1), Tf, 'x')
        plt.plot(np.linalg.norm(UV, axis=1), np.polyval(tf_approx1, np.linalg.norm(UV, axis=1)), label="Linear")
        plt.plot(np.linalg.norm(UV, axis=1), np.polyval(tf_approx2, np.linalg.norm(UV, axis=1)), label="Quadratic")
        plt.plot(np.linalg.norm(UV, axis=1), tf_constant_mass, label="Constant Mass")
        plt.xlabel("Ignition velocity (m/s)")
        plt.ylabel("Optimal ToF (s)")
        plt.legend()

        plt.show()
        self.debug = debug 




def test_plot():
    controller = JBG()
    controller.set_constants()
    controller.plot_manifold()

def test_controller():
    """ Demonstrate that the controller correctly nulls the velocity vector """
    import sys 
    sys.path.append("./")
    from Utils.RK4 import RK4 

    controller = JBG()
    controller.set_constants()
    controller.ignite()
    # controller.debug = False 

    x0 = [-3200+337, 2600-1357, 625, -270, 8500]  # Chosen to approx land at the target at zero velocity 
    tf, _ = controller.solve(-x0[2], -x0[3], x0[-1])
    
    def dyn(x, t,):
        s,h,u,v,m = x 
        T,mu = controller(x)
        return np.array([u, v, T/m*np.cos(mu), T/m*np.sin(mu)-controller.g, -T/controller.Isp/controller.g0])


    t = np.linspace(0, tf, 200)
    X = RK4(dyn, x0, t)

    plt.plot(t, X[:, 0:2])
    plt.xlabel('time')
    plt.ylabel('pos')
    plt.figure()
    plt.plot(t, X[:, 2:4])
    plt.xlabel('time')
    plt.ylabel('velocity')
    plt.show()

if __name__ == "__main__":
    test_plot()
    # test_controller()
