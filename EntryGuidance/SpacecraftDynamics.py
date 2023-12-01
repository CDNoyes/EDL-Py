"""
Rotational dynamics with and without attitude kinematics 
Attitude is written using quaternions 
"""

import numpy as np 


class SpacecraftDynamics:
    """
        Constant inertia matrix angular dynamics 

        Attitude = [q1 q2 q3 q_scalar]
        Velocity = [w1 w2 w2]

        If kinematics = False, state = velocity
        Else, state = [attitude, velocity]
    """

    def __init__(self, inertia, kinematics=False):
        # Verify inertia input 
        self.__inertia(inertia)

        # Setup the appropriate dynamics model 
        self.__kinematics = kinematics 

        if kinematics:
            self.__dynamics = None  # TODO: Implement quaternion kinematics 
            self.__states = ["q0", "q1", "q2", "q3", "w1", "w2", "w3"]
        else:
            self.__dynamics = self.__dynamics_angular
            self.__states = ["w1", "w2", "w3"]

    def dynamics(self, u):
        return lambda x, t: self.__dynamics(x, t, u)

    def __dynamics_angular(self, w, t):
        wJw = np.cross(w, self.inertia.dot(w))  # Could compare performance with constructing temporary omega matrix, and using dot instead of cross
        # wJw = np.dot([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]], self.inertia.dot(w))
        return self.inertia_inverse.dot(u - wJw)

    def __inertia(self, inertia):
        J = np.asarray(inertia)
        assert J.shape[0] == 3, "SpacecraftDynamics: inertia must be an array-like of 3 principal MoI, or a 3x3 matrix."

        if J.ndim == 1:
            J = np.diag(J)

        Ji = np.linalg.inv(J)
        self.inertia, self.inertia_inverse = J, Ji


def test():
    try:
        sc = SpacecraftDynamics(inertia=[0])  # Should fail
    except AssertionError:
        print("Caught improper inertia input")



    w0 = [-0.4, 0.8, 2]
    J = [86.24, 85.07, 113.59]
    sc = SpacecraftDynamics(J, kinematics=False)