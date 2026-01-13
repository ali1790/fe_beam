from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

class Element(ABC):
    """
    Abstract base class for all elements
    """

    id: int
    node_ids: List[int]

    @abstractmethod
    def get_stiffness_matrix(self) -> np.ndarray:
        """
        Returns local element stiffness matrix
        """
        pass

    @abstractmethod
    def get_mass_matrix(self) -> np.ndarray:
        """
        Returns local element mass matrix
        """
        pass
    @abstractmethod
    def get_dof_types(self) -> List[str]:
        """
        Returns DOFs per node.
        """
        pass

    @abstractmethod
    def get_local_dof_mapping(self) -> List[Tuple[int, str]]:
        """
        Returns local DOF-mapping:
        [(node_id, dof_type), ...]
        in matrix order.
        """
        pass

    def validate(self) -> None:
        """
        Optional valididation.
        """
        return
