# core/solvers/modal_solver.py

import numpy as np
import scipy.sparse.linalg as spla

from core.solvers.base_solver import Solver

class ModalSolver(Solver):
    """
    Modal analysis solver.
    """

    def solve(
        self,
        num_modes: int,
        sigma: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve the generalized eigenvalue problem.

        Parameters
        ----------
        num_modes : int
            Number of eigenmodes to compute.
        sigma : float, optional
            Spectral shift for shift-invert mode.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Natural frequencies (rad/s) and mode shapes.
        """
        if self.M is None:
            raise ValueError("Mass matrix required for modal analysis.")

        eigvals, eigvecs = spla.eigsh(
            self.K,
            k=num_modes,
            M=self.M,
            sigma=sigma,
            which="LM"
        )

        omegas = np.sqrt(np.abs(eigvals))
        return omegas, eigvecs
