import numpy as np
import math
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
    
    def __solve_constraints(self, dt:float):
        self.__solve_distance_constraints(dt)
        self.__solve_volume_constraints(dt)

    def __solve_distance_constraints(self, dt:float):
        for constraint in self.__distance_constraints:
            particle_0 = self.particles[constraint.particle_index_0]
            particle_1 = self.particles[constraint.particle_index_1]

            alpha = constraint.compliance / dt / dt 
            total_weight = particle_0.inverse_mass + particle_1.inverse_mass
            
            if math.isclose(total_weight, 0):
                continue

            gradiance = np.subtract(particle_0.position, particle_1.position)
            delta = particle_1.position - particle_0.position
            delta_length = np.linalg.norm(delta)

            if math.isclose(delta_length, 0):
                continue

            gradiance *= 1.0 / delta_length
            c = delta_length - constraint.distance
            s = -c / (total_weight + alpha)

            particle_0.position += s * particle_0.inverse_mass * gradiance
            particle_1.position -= s * particle_1.inverse_mass * gradiance
        pass

    def __solve_volume_constraints(self, dt:float):
        # C = 6(V - Vrest)
        # Gradients =
        # [0] = (x3-x1) x (x2-x1)
        # [1] = (x2-x0) x (x3-x0)
        # [2] = (x3-x0) x (x1-x0)
        # [3] = (x1-x0) x (x2-x0)

        for constraint in self.__volume_constraints:
            alpha = constraint.compliance / dt / dt

            vertices = [
                self.particles[constraint.affected_particle_indices[0]].position,
                self.particles[constraint.affected_particle_indices[1]].position,
                self.particles[constraint.affected_particle_indices[2]].position,
                self.particles[constraint.affected_particle_indices[3]].position
            ]

            gradients = [
                np.cross(np.subtract(vertices[3], vertices[1]), np.subtract(vertices[2], vertices[1])),
                np.cross(np.subtract(vertices[2], vertices[0]), np.subtract(vertices[3], vertices[0])),
                np.cross(np.subtract(vertices[3], vertices[1]), np.subtract(vertices[1], vertices[0])),
                np.cross(np.subtract(vertices[1], vertices[1]), np.subtract(vertices[2], vertices[0]))
            ]

            total_weight = 0
            for i in range(4):
                total_weight += self.particles[constraint.affected_particle_indices[i]].inverse_mass

            c = self.__tetrahedron_volume(constraint.affected_particle_indices) - constraint.volume
            s = -c / (total_weight + alpha)

            # apply constraints
            for i in range(4):
                self.particles[constraint.affected_particle_indices[i]].position += (s * self.particles[constraint.affected_particle_indices[i]].inverse_mass / 6.0) * gradients[i]
            

    def __tetrahedron_volume(self, particle_indices:list[int]):
        # V = 1/6((x2-x1) x (x1-x1)) . (x4-x1)
        base_particle_position = self.particles[particle_indices[0]].position

        a = np.subtract(self.particles[particle_indices[1]].position, base_particle_position)
        b = np.subtract(self.particles[particle_indices[2]].position, base_particle_position)
        c = np.subtract(self.particles[particle_indices[3]].position, base_particle_position)

        return ((1.0 / 6.0) * np.abs(np.dot(a, np.cross(b, c))))
