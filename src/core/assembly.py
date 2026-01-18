# assembly.py
#
# Global matrix assembly for 3D Timoshenko beam elements with local-to-global
# coordinate transformation.
#
# English comments are used by request.
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

try:
    from scipy.sparse import lil_matrix, csr_matrix
except ImportError as e:
    raise ImportError(
        "scipy is required for sparse assembly (scipy.sparse). "
        "Install scipy or replace sparse matrices with dense numpy arrays."
    ) from e


# -----------------------------
# Transformation utilities
# -----------------------------

def _normalize(v: np.ndarray, tol: float = 1e-14) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < tol:
        raise ValueError("Cannot normalize a near-zero vector.")
    return v / n


def compute_beam_rotation_matrix_3d(
    x1: np.ndarray,
    x2: np.ndarray,
    reference: Optional[np.ndarray] = None,
    *,
    tol_parallel: float = 1e-10,
) -> np.ndarray:
    """
    Build a right-handed local coordinate system for a 3D beam.

    Local axes:
      - e1: along the element axis (node1 -> node2)
      - e2: in the plane spanned by e1 and the given reference vector (as far as possible)
      - e3: completes the right-handed triad

    Returns:
      R (3x3) with columns being local basis vectors expressed in global coordinates,
      such that:
        v_global = R @ v_local
      and R is orthonormal (R.T == R^{-1}).
    """
    x1 = np.asarray(x1, dtype=float).reshape(3)
    x2 = np.asarray(x2, dtype=float).reshape(3)

    e1 = _normalize(x2 - x1)

    # Default reference vector: global Z, and if nearly parallel to e1, use global Y.
    if reference is None:
        ref = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(ref, e1))) > 1.0 - tol_parallel:
            ref = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        ref = np.asarray(reference, dtype=float).reshape(3)

    # Remove the component of ref along e1 to get a vector perpendicular to e1.
    ref_perp = ref - float(np.dot(ref, e1)) * e1
    ref_norm = float(np.linalg.norm(ref_perp))

    # If reference is parallel to e1, choose an alternate reference automatically.
    if ref_norm < tol_parallel:
        # Pick a vector not parallel to e1 by selecting the smallest component axis.
        if abs(e1[0]) < abs(e1[1]) and abs(e1[0]) < abs(e1[2]):
            alt = np.array([1.0, 0.0, 0.0], dtype=float)
        elif abs(e1[1]) < abs(e1[2]):
            alt = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            alt = np.array([0.0, 0.0, 1.0], dtype=float)

        ref_perp = alt - float(np.dot(alt, e1)) * e1
        ref_perp = _normalize(ref_perp)
    else:
        ref_perp = ref_perp / ref_norm

    e2 = ref_perp
    e3 = np.cross(e1, e2)
    e3 = _normalize(e3)

    # Re-orthogonalize e2 to avoid accumulation of numerical error.
    e2 = np.cross(e3, e1)
    e2 = _normalize(e2)

    R = np.column_stack((e1, e2, e3))  # columns = local axes in global coords
    return R


def build_beam_T_12x12(R: np.ndarray) -> np.ndarray:
    """
    Build the 12x12 transformation matrix for a 2-node 3D beam element with DOFs:
      [u, v, w, phix, phiy, phiz] per node.

    With R mapping local -> global for 3D vectors:
      v_global = R @ v_local

    Then the element DOF vector transforms as:
      q_global = T @ q_local

    For an orthonormal R, inverse is T.T.
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError("R must be 3x3.")

    # 6x6 per node: translations and rotations both transform as 3D vectors
    T6 = np.zeros((6, 6), dtype=float)
    T6[0:3, 0:3] = R
    T6[3:6, 3:6] = R

    # 12x12 for two nodes
    T = np.zeros((12, 12), dtype=float)
    T[0:6, 0:6] = T6
    T[6:12, 6:12] = T6
    return T


# -----------------------------
# Assembly
# -----------------------------

@dataclass(frozen=True)
class AssembledSystem:
    """
    Container for assembled global matrices.
    """
    K: csr_matrix
    M: csr_matrix


def assemble_global_matrices(
    mesh,
    elements: Iterable,
    dof_manager,
    *,
    transform_to_global: bool = True,
    reference_vectors: Optional[Dict[int, np.ndarray]] = None,
) -> AssembledSystem:
    """
    Assemble global stiffness and mass matrices.

    Parameters
    ----------
    mesh:
        Must provide:
          - get_element_nodes(element_id) -> [Node, Node]
          - (optional) get_element_length(element_id) (not used here)
        Nodes must provide coordinates() -> np.ndarray shape (3,)

    elements:
        Iterable of element objects providing:
          - id (int)
          - node_ids (len==2)
          - get_stiffness_matrix() -> (12,12) local matrix
          - get_mass_matrix() -> (12,12) local matrix
          - get_local_dof_mapping() consistent with dof_manager

    dof_manager:
        Must provide:
          - number_of_dofs()
          - get_element_dof_indices(element) -> list[int] of length 12

    transform_to_global:
        If True, apply 3D local-to-global transformation for each element:
          K_global_e = T @ K_local @ T.T
          M_global_e = T @ M_local @ T.T

    reference_vectors:
        Optional dict: element_id -> reference vector (3,)
        Used to define the local y/z axes (controls twist about the beam axis).
        If not provided, a robust default is used (global Z, fallback global Y).

    Returns
    -------
    AssembledSystem with sparse CSR matrices K, M.
    """
    ndof = int(dof_manager.number_of_dofs())
    K = lil_matrix((ndof, ndof), dtype=float)
    M = lil_matrix((ndof, ndof), dtype=float)

    ref_map = reference_vectors or {}

    for elem in elements:
        # Global DOF indices for this element (length 12)
        edofs = dof_manager.get_element_dof_indices(elem)
        if len(edofs) != 12:
            raise ValueError(
                f"Element {getattr(elem, 'id', '?')} must have 12 DOFs in 3D."
            )

        Ke = np.asarray(elem.get_stiffness_matrix(), dtype=float)
        Me = np.asarray(elem.get_mass_matrix(), dtype=float)

        if Ke.shape != (12, 12):
            raise ValueError(f"Element {elem.id}: stiffness matrix must be 12x12.")
        if Me.shape != (12, 12):
            raise ValueError(f"Element {elem.id}: mass matrix must be 12x12.")

        if transform_to_global:
            # Build element rotation from its end node coordinates
            n1, n2 = mesh.get_element_nodes(elem.id)
            x1 = n1.coordinates()
            x2 = n2.coordinates()

            ref = ref_map.get(elem.id, None)
            R = compute_beam_rotation_matrix_3d(x1, x2, reference=ref)
            T = build_beam_T_12x12(R)

            # With q_global = T q_local:
            # K_global = T K_local T^T (same for M)
            Ke = T @ Ke @ T.T
            Me = T @ Me @ T.T

        # Sparse assembly: add element contributions
        for a_local, A in enumerate(edofs):
            for b_local, B in enumerate(edofs):
                K[A, B] += Ke[a_local, b_local]
                M[A, B] += Me[a_local, b_local]

    return AssembledSystem(K=K.tocsr(), M=M.tocsr())


def assemble_global_load_vector(
    dof_manager,
    *,
    nodal_loads: Optional[Dict[Tuple[int, str], float]] = None,
) -> np.ndarray:
    """
    Assemble a global load vector f from nodal loads.

    Parameters
    ----------
    dof_manager:
        Must provide get_dof_index(node_id, dof_type) and number_of_dofs().

    nodal_loads:
        Dict keyed by (node_id, dof_type) -> value.
        Example for 3D:
          (10, "u") = 100.0
          (10, "phiz") = 5.0

    Returns
    -------
    f: ndarray shape (ndof,)
    """
    ndof = int(dof_manager.number_of_dofs())
    f = np.zeros(ndof, dtype=float)

    if not nodal_loads:
        return f

    for (node_id, dof_type), value in nodal_loads.items():
        idx = dof_manager.get_dof_index(node_id, dof_type)
        f[idx] += float(value)

    return f

#Notes you will care about (practical FEM details):

#Twist control: reference_vectors[element_id] = np.array([...]) lets you define the element’s local  y/z
 #orientation (important for non-prismatic beams, anisotropic sections, or when you want deterministic rotation about the beam axis).

#Transformation convention: this file uses q_global = T @ q_local, therefore K_global = T @ K_local @ T.T. If you prefer the opposite convention, swap accordingly (and keep it consistent everywhere).

#This assumes each 3D beam element returns local Ke, Me in the DOF order:
#[u1, v1, w1, phix1, phiy1, phiz1, u2, v2, w2, phix2, phiy2, phiz2]