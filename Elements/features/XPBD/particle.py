import numpy as np

class Particle:
    """A point mass used to simulate soft bodies"""

    position:np.array
    previous_position:np.array
    velocity:np.array
    inverse_mass:float

    is_kinematic:bool

    def __init__(self, position = np.array([0.0, 0.0, 0.0]), mass = 1.0, is_kinematic = False):
        self.position = position
        self.previous_position = position
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.mass = mass
        self.is_kinematic = is_kinematic

    @property
    def mass(self):
        return 1.0 / self.inverse_mass

    @mass.setter
    def mass(self, value):
        self.inverse_mass = 1.0 / value