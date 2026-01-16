from abc import ABC, abstractmethod
import scipy.sparse as sp
import numpy as np

class Solver(ABC):
    """
    Abstract base class for all FEM solvers.
    """

    def __init__(self, K: sp.csr_matrix, M: sp.csr_matrix | None = None):
        self.K = K
        self.M = M

    @abstractmethod
    def solve(self):
        """
        Solve the FEM system.
        """
        pass
