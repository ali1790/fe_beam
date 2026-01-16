from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import scipy.sparse as sp

from core.dof import DofManager

@dataclass(frozen=True)
class DirichletBC:
    """
    Represents a Dirichlet (essential) boundary condition.

    Attributes
    ----------
    node_id : int
        Node at which the boundary condition is applied.
    dof_type : str
        Degree of freedom type (e.g. 'u', 'v', 'w', 'phix', ...).
    value : float
        Prescribed value.
    """
    node_id: int
    dof_type: str
    value: float

def apply_dirichlet_bcs(
    K: sp.csr_matrix,
    M: Optional[sp.csr_matrix],
    f: Optional[np.ndarray],
    bcs: List[DirichletBC],
    dof_manager: DofManager
) -> Tuple[sp.csr_matrix, Optional[sp.csr_matrix], Optional[np.ndarray]]:
    """
    Apply Dirichlet boundary conditions using row/column elimination.

    Parameters
    ----------
    K : sp.csr_matrix
        Global stiffness matrix.
    M : sp.csr_matrix or None
        Global mass matrix (optional).
    f : np.ndarray or None
        Global load vector (optional).
    bcs : List[DirichletBC]
        List of Dirichlet boundary conditions.
    dof_manager : DofManager
        Global DOF manager.

    Returns
    -------
    Tuple[sp.csr_matrix, Optional[sp.csr_matrix], Optional[np.ndarray]]
        Modified (K, M, f).
    """
    K_mod = K.tolil(copy=True)
    M_mod = M.tolil(copy=True) if M is not None else None
    f_mod = f.copy() if f is not None else None

    for bc in bcs:
        dof = dof_manager.get_dof_index(bc.node_id, bc.dof_type)

        # Modify load vector
        if f_mod is not None:
            f_mod -= K_mod[:, dof].toarray().ravel() * bc.value

        # Zero row and column in stiffness matrix
        K_mod[dof, :] = 0.0
        K_mod[:, dof] = 0.0
        K_mod[dof, dof] = 1.0

        # Set prescribed value
        if f_mod is not None:
            f_mod[dof] = bc.value

        # Apply same treatment to mass matrix if present
        if M_mod is not None:
            M_mod[dof, :] = 0.0
            M_mod[:, dof] = 0.0
            M_mod[dof, dof] = 1.0

    return (
        K_mod.tocsr(),
        M_mod.tocsr() if M_mod is not None else None,
        f_mod
    )

@dataclass(frozen=True)
class NeumannBC:
    """
    Represents a Neumann (natural) boundary condition.

    Attributes
    ----------
    node_id : int
        Node at which the load is applied.
    dof_type : str
        Degree of freedom type.
    value : float
        Applied load value.
    """
    node_id: int
    dof_type: str
    value: float

def apply_neumann_bcs(
    f: np.ndarray,
    bcs: List[NeumannBC],
    dof_manager: DofManager
) -> np.ndarray:
    """
    Apply Neumann boundary conditions to the global load vector.

    Parameters
    ----------
    f : np.ndarray
        Global load vector.
    bcs : List[NeumannBC]
        List of Neumann boundary conditions.
    dof_manager : DofManager
        Global DOF manager.

    Returns
    -------
    np.ndarray
        Modified load vector.
    """
    f_mod = f.copy()

    for bc in bcs:
        dof = dof_manager.get_dof_index(bc.node_id, bc.dof_type)
        f_mod[dof] += bc.value

    return f_mod


if __name__=='__main__':
    from core.assembly import assemble_global_matrices
    from core.assembly import DofManager
    from core.mesh import Mesh
    
    # Dummies
    mesh = Mesh()
    elements = None
    dof_manager = DofManager()
    
    # typical usage for harmonic analysis
    # Assemble system
    K, M = assemble_global_matrices(mesh, elements, dof_manager)

    # Load vector
    f = np.zeros(dof_manager.number_of_dofs())

    # Boundary conditions
    dirichlet_bcs = [
        DirichletBC(node_id=1, dof_type="u", value=0.0),
        DirichletBC(node_id=1, dof_type="v", value=0.0),
        DirichletBC(node_id=1, dof_type="w", value=0.0),
    ]

    neumann_bcs = [
        NeumannBC(node_id=2, dof_type="w", value=100.0)
    ]

    f = apply_neumann_bcs(f, neumann_bcs, dof_manager)
    K_bc, M_bc, f_bc = apply_dirichlet_bcs(K, M, f, dirichlet_bcs, dof_manager)