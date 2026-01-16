import numpy as np
import scipy.sparse.linalg as spla

from .base_solver import Solver

class HarmonicSolver(Solver):
    """
    Harmonic response solver.
    """

    def solve(self, f: np.ndarray, omega: float) -> np.ndarray:
        """
        Solve the harmonic system.

        Parameters
        ----------
        f : np.ndarray
            Complex load vector.
        omega : float
            Angular frequency.

        Returns
        -------
        np.ndarray
            Complex displacement amplitudes.
        """
        Kd = self.K - (omega ** 2) * self.M
        return spla.spsolve(Kd, f)
