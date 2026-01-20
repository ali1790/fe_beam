from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class ElementMatrix:
    """
    Container for a local element matrix together with its DOF ordering.

    This class decouples numerical matrix data from element logic and
    preserves the exact DOF sequence as defined by the external source
    (e.g. ANSYS CBMX format).
    """

    matrix: np.ndarray
    dof_order: List[str]

    def __post_init__(self):
        """
        Validate matrix dimensions and DOF ordering.
        """
        if self.matrix.ndim != 2:
            raise ValueError("Element matrix must be two-dimensional.")

        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Element matrix must be square.")

        if len(self.dof_order) != self.matrix.shape[0]:
            raise ValueError(
                "Length of DOF order does not match matrix dimension."
            )

    def copy(self) -> "ElementMatrix":
        """
        Return a deep copy of the element matrix.
        """
        return ElementMatrix(self.matrix.copy(), self.dof_order.copy())

    def is_symmetric(self, tol: float = 1e-12) -> bool:
        """
        Check whether the matrix is symmetric.

        Parameters
        ----------
        tol : float
            Numerical tolerance.

        Returns
        -------
        bool
            True if symmetric within tolerance.
        """
        return np.allclose(self.matrix, self.matrix.T, atol=tol)
