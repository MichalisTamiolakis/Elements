from typing import List
class VolumeConstraint:
    """
    A tetrahedron volume constraint, defined by 4 particles.
    All other shapes can be created by primitive tetrahedrons.
    """

    affected_particle_indices:List[int]
    volume:float
    compliance:float

    def __init__(self, affected_particle_indices: List[int], volume, compliance:float) -> None:
        self.affected_particle_indices = affected_particle_indices
        self.volume = volume
        self.compliance = compliance

class DistanceConstraint:
    """A distance constraint for 2 particles"""
    
    particle_index_0:int
    particle_index_1:int
    distance:float
    compliance:float

    def __init__(self, particle_index_0:int , particle_index_1:int, distance, compliance = 100) -> None:
        self.particle_index_0 = particle_index_0
        self.particle_index_1 = particle_index_1
        self.distance = distance
        self.compliance = compliance
