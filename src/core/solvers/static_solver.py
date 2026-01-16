# core/solvers/static_solver.py

import numpy as np
import scipy.sparse.linalg as spla

from core.solvers.base_solver import Solver

class StaticSolver(Solver):
    """
    Linear static FEM solver.
    """

    def solve(self, f: np.ndarray) -> np.ndarray:
        """
        Solve the static system Ku = f.

        Parameters
        ----------
        f : np.ndarray
            Global load vector.

        Returns
        -------
        np.ndarray
            Displacement vector.
        """
        return spla.spsolve(self.K, f)
