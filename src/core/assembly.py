from typing import Iterable, Tuple
import numpy as np
import scipy.sparse as sp

from core.dof import DofManager
from core.mesh import Mesh
from elements.base_element import Element

def compute_rotation_matrix(ex: np.ndarray) -> np.ndarray:
    """
    Compute a 3x3 rotation matrix from the element axis direction.

    Parameters
    ----------
    ex : np.ndarray
        Unit vector defining the local x-axis of the element.

    Returns
    -------
    np.ndarray
        3x3 rotation matrix.
    """
    ex = ex / np.linalg.norm(ex)

    # Reference vector to define local y/z plane
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(ex, ref)) > 0.95:
        ref = np.array([0.0, 1.0, 0.0])

    ez = np.cross(ex, ref)
    ez /= np.linalg.norm(ez)

    ey = np.cross(ez, ex)

    R = np.column_stack((ex, ey, ez))
    return R

def build_transformation_matrix(R: np.ndarray) -> np.ndarray:
    """
    Build the 12x12 transformation matrix for a 3D beam element.

    The same rotation matrix is applied to translations and rotations
    at both element nodes.

    Parameters
    ----------
    R : np.ndarray
        3x3 rotation matrix.

    Returns
    -------
    np.ndarray
        12x12 transformation matrix.
    """
    T = np.zeros((12, 12))

    for i in range(4):
        T[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = R

    return T

def transform_element_matrices( element: Element, mesh: Mesh) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform local element matrices to global coordinates.

    Parameters
    ----------
    element : Element
        Finite element with local matrices.
    mesh : Mesh
        Mesh providing geometric information.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Transformed stiffness and mass matrices.
    """
    Ke_local = element.get_stiffness_matrix()
    Me_local = element.get_mass_matrix()

    # Compute element orientation
    ex = mesh.get_element_direction_cosines(element.id)
    R = compute_rotation_matrix(ex)
    T = build_transformation_matrix(R)

    Ke_global = T.T @ Ke_local @ T
    Me_global = T.T @ Me_local @ T

    return Ke_global, Me_global

def assemble_global_matrices(
    mesh: Mesh,
    elements: Iterable[Element],
    dof_manager: DofManager
) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """
    Assemble global stiffness and mass matrices.

    Parameters
    ----------
    mesh : Mesh
        FEM mesh.
    elements : Iterable[Element]
        Collection of finite elements.
    dof_manager : DofManager
        Global degree-of-freedom manager.

    Returns
    -------
    Tuple[sp.csr_matrix, sp.csr_matrix]
        Global stiffness and mass matrices (CSR format).
    """
    ndofs = dof_manager.number_of_dofs()

    K_global = sp.lil_matrix((ndofs, ndofs))
    M_global = sp.lil_matrix((ndofs, ndofs))

    for element in elements:
        Ke, Me = transform_element_matrices(element, mesh)
        dof_indices = dof_manager.get_element_dof_indices(element)

        for i_local, i_global in enumerate(dof_indices):
            for j_local, j_global in enumerate(dof_indices):
                K_global[i_global, j_global] += Ke[i_local, j_local]
                M_global[i_global, j_global] += Me[i_local, j_local]

    return K_global.tocsr(), M_global.tocsr()

def assemble_dynamic_stiffness(
    K: sp.csr_matrix,
    M: sp.csr_matrix,
    omega: float
) -> sp.csr_matrix:
    """
    Compute the dynamic stiffness matrix for harmonic analysis.

    Parameters
    ----------
    K : sp.csr_matrix
        Global stiffness matrix.
    M : sp.csr_matrix
        Global mass matrix.
    omega : float
        Angular frequency.

    Returns
    -------
    sp.csr_matrix
        Dynamic stiffness matrix.
    """
    return K - (omega ** 2) * M

def typical_use(mesh, elements):
# Enumerate DOFs
    dof_manager = DofManager()
    dof_manager.enumerate_dofs(elements)

    # Assemble matrices
    K, M = assemble_global_matrices(mesh, elements, dof_manager)

    # Harmonic analysis at frequency omega
    omega = 100.0
    Kd = assemble_dynamic_stiffness(K, M, omega)
