import numpy as np
from dataclasses import dataclass
from fe_beam.core.boundary_conditions import NeumannBC, apply_dirichlet_harmonic
from fe_beam.utils.helpers import NXbeam, distribute_between_nodes
from fe_beam.elements.point_masses import PointMass, add_point_masses_to_M

@dataclass
class Constraints:
    min_dist: float
    max_displacement: float
    max_mass: np.ndarray

@dataclass
class Objective:
    beam_model: NXbeam
    target_moments: np.ndarray
    constraints: Constraints

    def __call__(self, x):
        # create copies of full system matrices
        K, M = self.beam_model.get_system_matrices()

        point_masses = []
        # create point masses
            # implement correct assignment x -> Li, mi
            # check if distancve and max_mass constraints are violated
            # for each point mass:
                # distribute point mass between nodes
                # point_masses.append(PointMasse(...))
        M = add_point_masses_to_M(M, self.beam_model._dof_manager, point_masses)

        # Apply harmonic Dirichlet boundary conditions
        # Apply harmonic force
            # distribute between nodes
            # neumann_bcs = [NeumannBC(...), NeumannBC(...)]



