import numpy as np
from Elements.utils.softbodies.constraints import VolumeConstraint, DistanceConstraint
from Elements.utils.softbodies.particle import Particle

class Solver:
    """
    A solver using Extended Position Based Dynamics (XPBD)
    Can be used to simulate any kind of constrained object, like softbody, cloth, rope, etc.
    """
    
    gravity:[float, float, float] = [0.0, -9.81, 0.0]
    substeps:int = 10

    particles:list[Particle]

    __volume_constraints:list[VolumeConstraint]
    __distance_constraints:list[DistanceConstraint]
    

    def __init__(self, particle_positions:np.array([])):
        # Calculate initial volume of each tetrahedron & initial distance of each distance constraint
        for particle in particle_positions:
            self.particles.append(Particle(particle))

    def add_volume_constraint(self, particle_indices:list[int], compliance: float)->int:
        """
        Adds a volume constraint to the solver. 
        The solver will try to keep the volume of the tetrahedron constant.
        
        Params
        ------
        particle_indices: list[int]
            The indices of the particles constituting the tetrahedron.
        compliance: float
            The compliance of the constraint. Higher values mean less stiffness.
        """

        self.__volume_constraints.append(VolumeConstraint(particle_indices, self.__tetrahedron_volume(particle_indices), compliance))
    
    def add_distance_constraint(self, particle_index_0:int, particle_index_1:int, compliance:float)->int:
        """
        Adds a distance constraint to the solver. 
        The solver will try to keep the distance between the two particles constant.
        
        Params
        ------
        particle_index_0: int
            The segment's first particle index
        particle_index_1: int
            The segment's second particle index
        compliance: float
            The compliance of the constraint. Higher values mean less stiffness.
        """

        self.__distance_constraints.append(DistanceConstraint(particle_index_0, particle_index_1, np.linalg.norm(np.subtract(np.array(self.particles[particle_index_1].position), np.array(self.particles[particle_index_1].position))), compliance))

    def step(self, dt:float):
        substepDt:float = dt / self.substeps

        for i in range(self.substeps):
            
            # Pre solve
            for particle in self.particles:
                
                particle.velocity += self.gravity * substepDt # Update velocity
                particle.previous_position = particle.position # Update previous positions
                particle.position += particle.velocity * substepDt # Update position

                if particle.position[1] < 0: # Check for collision here. Now only ground collision is implemented
                    particle.position[1] = 0
                    particle.previous_position = particle.position
                
                if particle.is_kinematic: # If kinematic undo the above
                    particle.position = particle.previous_position

            # Solve
            self.__solve_constraints()

            # Post solve
            for particle in self.particles:
                particle.velocity = (particle.position - particle.previous_position) / substepDt # Update velocity
    
    def __solve_constraints(self):
        pass

    def __tetrahedron_volume(self, particle_indices:list[int]):
        # V = 1/6((x2-x1) x (x1-x1)) . (x4-x1)
        base_particle_position = self.particles[particle_indices[0]].position

        a = np.subtract(self.particles[particle_indices[1]].position, base_particle_position)
        b = np.subtract(self.particles[particle_indices[2]].position, base_particle_position)
        c = np.subtract(self.particles[particle_indices[3]].position, base_particle_position)

        return ((1.0 / 6.0) * np.abs(np.dot(a, np.cross(b, c))))
