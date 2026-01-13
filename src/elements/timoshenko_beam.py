from typing import List
import numpy as np

from elements.base_element import Element
from matrices.element_matrices import ElementMatrix
from materials.beam_material import BeamMaterial


class TimoshenkoBeamElement(Element):
    """
    Linear 2 node Timoshenko beam element (2D).
    """


    #: DOFs per node (local)
    DOF_TYPES = ["u", "v", "w", "phix", "phiy", "phiz"]

    def __init__(
        self,
        element_id: int,
        node_ids: List[int],
        material: BeamMaterial,
        stiffness_matrix: ElementMatrix,
        mass_matrix: ElementMatrix,
        shear_correction_factors: Tuple[float, float] = (1.0, 1.0),
    ):
        if len(node_ids) != 2:
            raise ValueError(
                "3D-Timoshenko-Balkenelement benötigt genau 2 Knoten."
            )

        self.id = element_id
        self.node_ids = node_ids
        self.material = material

        self._Ke = stiffness_matrix
        self._Me = mass_matrix

        # kappa_y, kappa_z (Schubkorrektur)
        self.shear_correction_factors = shear_correction_factors

    def get_stiffness_matrix(self) -> np.ndarray:
        """
        Local element stiffness matrix (12x12).
        """
        return self._Ke.matrix

    def get_mass_matrix(self) -> np.ndarray:
        """
        Local element mass matrix (12x12).
        """
        return self._Me.matrix

    def get_dof_types(self) -> List[str]:
        return self.DOF_TYPES

    def get_local_dof_mapping(self) -> List[Tuple[int, str]]:
        """
        Returns local DOF structure in CBMX order.

        [
          (n1, u), (n1, v), (n1, w), (n1, phix), (n1, phiy), (n1, phiz),
          (n2, u), (n2, v), (n2, w), (n2, phix), (n2, phiy), (n2, phiz)
        ]
        """
        mapping = []

        for node_id in self.node_ids:
            for dof in self.DOF_TYPES:
                mapping.append((node_id, dof))

        return mapping

    def validate_matrices(self) -> None:
        """
        Checks dimension and DOF consistency of matrices
        """
        ndofs = len(self.node_ids) * len(self.DOF_TYPES)

        if self._Ke.matrix.shape != (ndofs, ndofs):
            raise ValueError("Stiffness matrix must be of shape 12x12.")

        if self._Me.matrix.shape != (ndofs, ndofs):
            raise ValueError("Mass matrix must be of shape 12x12.")

        if self._Ke.dof_order != self._Me.dof_order:
            raise ValueError(
                "DOF-order of K and M is inconsistent."
            )
