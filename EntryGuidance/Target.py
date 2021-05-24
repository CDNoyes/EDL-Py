import numpy as np 

class Target:
    def __init__(self, latitude, longitude, altitude=None, velocity=None):
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.velocity = velocity 
        
        # self.state = sphere_to_cart([])

    def coordinates(self, degrees=False):
        if degrees:
            return np.degrees([self.longitude, self.latitude])
        else:
            return np.array([self.longitude, self.latitude])

    # def position(self,):
        # return 

    # def state(self,):