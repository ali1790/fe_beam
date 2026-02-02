# postprocessing.py
#
# Postprocessing utilities for 3D beam elements:
#   - element end forces (internal nodal force resultants) in local and global coordinates
#   - support reactions (static) in global coordinates
#
# Conventions used:
#   - DOF order per node: [u, v, w, phix, phiy, phiz]
#   - Element DOF vector (12): [node1 6 dofs, node2 6 dofs]
#   - Transformation convention consistent with assembly.py:
#       q_global = T @ q_local
#     Therefore:
#       q_local  = T.T @ q_global  (T orthonormal)
#       f_global = T @ f_local
#       f_local  = T.T @ f_global
#
# "Element end forces" returned here are the internal nodal force resultants
# that satisfy (in local system):
#   f_int_local = K_local * q_local - f_eq_local   (static)
#   f_int_local = Z_local(omega) * q_local - f_eq_local (harmonic)
#
# English comments are used by request.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from scipy.sparse import csr_matrix, issparse
except ImportError as e:
    raise ImportError(
        "scipy is required for postprocessing (scipy.sparse)."
    ) from e

from fe_beam.core.assembly import compute_beam_rotation_matrix_3d, build_beam_T_12x12
from fe_beam.core.boundary_conditions import DirichletBC


Number = Union[float, complex, np.floating, np.complexfloating]


# -----------------------------
# Data containers
# -----------------------------

@dataclass(frozen=True)
class EndForces:
    """
    Element end forces (internal nodal force resultants).

    local:  12-vector in local beam coordinates:
      [Fx1,Fy1,Fz1,Mx1,My1,Mz1, Fx2,Fy2,Fz2,Mx2,My2,Mz2]

    global: 12-vector in global coordinates (same block structure).
    """
    element_id: int
    local: np.ndarray
    global_: np.ndarray  # 'global' is reserved in some contexts, so use global_


@dataclass(frozen=True)
class SupportReactions:
    """
    Support reactions in full DOF space (global coordinates).

    r is full-size vector (ndof,), nonzero primarily at constrained DOFs.
    """
    r: np.ndarray
    fixed_dofs: np.ndarray


# -----------------------------
# Helpers
# -----------------------------

def _to_csr(A) -> csr_matrix:
    if issparse(A):
        return A.tocsr()
    return csr_matrix(A)


def _extract_element_q_global(dof_manager, element, u_full: np.ndarray) -> np.ndarray:
    edofs = dof_manager.get_element_dof_indices(element)
    if len(edofs) != 12:
        raise ValueError(f"Element {getattr(element, 'id', '?')} does not have 12 DOFs.")
    return np.asarray(u_full)[edofs]


def _build_T_for_element(mesh, element, reference_vector: Optional[np.ndarray] = None) -> np.ndarray:
    n1, n2 = mesh.get_element_nodes(element.id)
    R = compute_beam_rotation_matrix_3d(n1.coordinates(), n2.coordinates(), reference=reference_vector)
    return build_beam_T_12x12(R)


def _default_eq_load_12(dtype) -> np.ndarray:
    return np.zeros(12, dtype=dtype)


# -----------------------------
# Element end forces (static)
# -----------------------------

def compute_element_end_forces_static(
    mesh,
    elements: Iterable,
    dof_manager,
    u_full: np.ndarray,
    *,
    reference_vectors: Optional[Dict[int, np.ndarray]] = None,
    element_equivalent_loads_local: Optional[Dict[int, np.ndarray]] = None,
) -> Dict[int, EndForces]:
    """
    Compute internal element end forces for a static solution.

    Parameters
    ----------
    u_full:
        Full displacement vector (ndof,).
    reference_vectors:
        Optional element_id -> reference vector (3,) to define the local y/z orientation.
    element_equivalent_loads_local:
        Optional element_id -> f_eq_local (12,), to subtract distributed-load equivalents etc.
        If not provided, treated as zero.

    Returns
    -------
    Dict: element_id -> EndForces(local, global_)
    """
    ref_map = reference_vectors or {}
    feq_map = element_equivalent_loads_local or {}

    out: Dict[int, EndForces] = {}

    for e in elements:
        T = _build_T_for_element(mesh, e, reference_vector=ref_map.get(e.id))

        q_g = _extract_element_q_global(dof_manager, e, u_full.astype(float))
        q_l = T.T @ q_g

        Ke_l = np.asarray(e.get_stiffness_matrix(), dtype=float)
        if Ke_l.shape != (12, 12):
            raise ValueError(f"Element {e.id}: stiffness matrix must be 12x12.")

        f_eq_l = feq_map.get(e.id, _default_eq_load_12(float))
        f_eq_l = np.asarray(f_eq_l, dtype=float).reshape(12)

        f_int_l = Ke_l @ q_l - f_eq_l
        f_int_g = T @ f_int_l

        out[e.id] = EndForces(element_id=e.id, local=f_int_l, global_=f_int_g)

    return out


# -----------------------------
# Element end forces (harmonic)
# -----------------------------

def compute_element_end_forces_harmonic(
    mesh,
    elements: Iterable,
    dof_manager,
    u_full: np.ndarray,
    omega: float,
    *,
    reference_vectors: Optional[Dict[int, np.ndarray]] = None,
    element_equivalent_loads_local: Optional[Dict[int, np.ndarray]] = None,
    element_damping_local: Optional[Dict[int, np.ndarray]] = None,
    structural_damping_g: float = 0.0,
) -> Dict[int, EndForces]:
    """
    Compute internal element end forces for a harmonic solution (complex amplitudes).

    Model:
      f_int_local = Z_local(omega) * q_local - f_eq_local
      Z_local = (1 + i*g) K_local + i*omega*C_local - omega^2*M_local

    Parameters
    ----------
    u_full:
        Full complex displacement vector (ndof,).
    element_damping_local:
        Optional element_id -> C_local (12,12) viscous damping for that element.
    structural_damping_g:
        Structural (hysteretic) damping coefficient g applied as (1 + i*g)K.
        If you want DMPRAT-like approximation for full harmonic, a common mapping is g = 2*zeta.

    Returns
    -------
    Dict: element_id -> EndForces(local, global_)
    """
    ref_map = reference_vectors or {}
    feq_map = element_equivalent_loads_local or {}
    c_map = element_damping_local or {}

    out: Dict[int, EndForces] = {}

    for e in elements:
        T = _build_T_for_element(mesh, e, reference_vector=ref_map.get(e.id))

        q_g = _extract_element_q_global(dof_manager, e, np.asarray(u_full, dtype=complex))
        q_l = T.T @ q_g

        Ke = np.asarray(e.get_stiffness_matrix(), dtype=float)
        Me = np.asarray(e.get_mass_matrix(), dtype=float)

        if Ke.shape != (12, 12):
            raise ValueError(f"Element {e.id}: stiffness matrix must be 12x12.")
        if Me.shape != (12, 12):
            raise ValueError(f"Element {e.id}: mass matrix must be 12x12.")

        Kc = (1.0 + 1j * float(structural_damping_g)) * Ke.astype(complex)
        Z = Kc - (float(omega) ** 2) * Me.astype(complex)

        if e.id in c_map and c_map[e.id] is not None:
            Ce = np.asarray(c_map[e.id], dtype=complex)
            if Ce.shape != (12, 12):
                raise ValueError(f"Element {e.id}: damping matrix must be 12x12.")
            Z = Z + 1j * float(omega) * Ce

        f_eq_l = feq_map.get(e.id, _default_eq_load_12(complex))
        f_eq_l = np.asarray(f_eq_l, dtype=complex).reshape(12)

        f_int_l = Z @ q_l - f_eq_l
        f_int_g = T @ f_int_l

        out[e.id] = EndForces(element_id=e.id, local=f_int_l, global_=f_int_g)

    return out


# -----------------------------
# Support reactions (static)
# -----------------------------

def compute_support_reactions_static(
    K,
    u_full: np.ndarray,
    f_full: np.ndarray,
    dof_manager,
    dirichlet_bcs: Sequence[DirichletBC],
    *,
    zero_free_dofs: bool = True,
) -> SupportReactions:
    """
    Compute support reactions for a static solution.

    Reaction definition:
      r = K u - f

    Parameters
    ----------
    K:
        Global stiffness matrix (ndof x ndof) sparse or dense.
    u_full:
        Full displacement vector (ndof,).
    f_full:
        Full applied load vector (ndof,).
    dirichlet_bcs:
        Used to identify fixed DOFs.
    zero_free_dofs:
        If True, sets reactions at non-fixed DOFs to zero for clarity.

    Returns
    -------
    SupportReactions with full reaction vector and fixed DOF indices.
    """
    K = _to_csr(K)
    u = np.asarray(u_full, dtype=np.result_type(K.dtype, u_full.dtype, np.float64))
    f = np.asarray(f_full, dtype=np.result_type(K.dtype, f_full.dtype, np.float64))

    ndof = K.shape[0]
    if u.shape[0] != ndof or f.shape[0] != ndof:
        raise ValueError("Incompatible sizes for K, u_full, f_full.")

    fixed = []
    for bc in dirichlet_bcs:
        fixed.append(int(dof_manager.get_dof_index(bc.node_id, bc.dof_type)))
    fixed_dofs = np.array(sorted(set(fixed)), dtype=int)

    r = (K @ u) - f
    r = np.asarray(r).reshape(-1)

    if zero_free_dofs:
        mask = np.ones(ndof, dtype=bool)
        mask[fixed_dofs] = False
        r[mask] = 0.0

    return SupportReactions(r=r, fixed_dofs=fixed_dofs)
