# boundary_conditions.py
#
# Boundary conditions (Dirichlet / Neumann) for:
#   - static analysis:            K u = f
#   - harmonic (frequency) analysis: (K + i*omega*C - omega^2*M) u = f(omega)
#
# This module focuses on a robust, reusable "elimination" (partition) approach:
# it reduces matrices/vectors by enforcing Dirichlet constraints exactly, and
# provides reconstruction of the full displacement vector.
#
# English comments are used by request.

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union, Dict

import numpy as np

try:
    from scipy.sparse import csr_matrix, issparse
except ImportError as e:
    raise ImportError(
        "scipy is required for sparse boundary condition handling (scipy.sparse). "
        "Install scipy or replace sparse matrices with dense numpy arrays."
    ) from e


Number = Union[float, complex, np.floating, np.complexfloating]


# -----------------------------
# Boundary condition data types
# -----------------------------

@dataclass(frozen=True)
class DirichletBC:
    """
    Essential boundary condition: enforce u(node_id, dof_type) = value.
    value can be real (static) or complex amplitude (harmonic).
    """
    node_id: int
    dof_type: str
    value: Number = 0.0


@dataclass(frozen=True)
class NeumannBC:
    """
    Natural boundary condition: apply external nodal load f(node_id, dof_type) += value.
    value can be real (static) or complex amplitude (harmonic).
    """
    node_id: int
    dof_type: str
    value: Number


@dataclass(frozen=True)
class ReducedSystem:
    """
    Reduced linear system after Dirichlet elimination.

    A_reduced * u_free = f_reduced

    Also stores:
      - free_dofs: indices in the full system that remain unknown
      - fixed_dofs: indices in the full system that were prescribed
      - u_fixed: full-size vector containing prescribed values at fixed DOFs
    """
    A: csr_matrix
    f: np.ndarray
    free_dofs: np.ndarray
    fixed_dofs: np.ndarray
    u_fixed: np.ndarray

    def reconstruct_full_solution(self, u_free: np.ndarray) -> np.ndarray:
        """
        Reconstruct the full-size displacement vector from the reduced solution.
        """
        u_full = np.array(self.u_fixed, copy=True)
        u_full[self.free_dofs] = u_free
        return u_full


# -----------------------------
# Helpers
# -----------------------------

def _to_csr(A) -> csr_matrix:
    if issparse(A):
        return A.tocsr()
    return csr_matrix(A)


def _collect_dirichlet(
    dof_manager,
    ndof: int,
    bcs: Sequence[DirichletBC],
    *,
    dtype: np.dtype,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      fixed_dofs: (k,) int indices
      u_fixed:    (ndof,) vector with prescribed values at fixed dofs, zero elsewhere
    """
    fixed = []
    u_fixed = np.zeros(ndof, dtype=dtype)

    for bc in bcs:
        idx = int(dof_manager.get_dof_index(bc.node_id, bc.dof_type))
        fixed.append(idx)
        u_fixed[idx] = dtype.type(bc.value) if hasattr(dtype, "type") else bc.value

    fixed_dofs = np.array(sorted(set(fixed)), dtype=int)
    return fixed_dofs, u_fixed


def _free_dofs_from_fixed(ndof: int, fixed_dofs: np.ndarray) -> np.ndarray:
    mask = np.ones(ndof, dtype=bool)
    mask[fixed_dofs] = False
    return np.nonzero(mask)[0]


def _reduce_system_by_dirichlet(
    A: csr_matrix,
    f: np.ndarray,
    fixed_dofs: np.ndarray,
    u_fixed: np.ndarray,
) -> ReducedSystem:
    """
    Generic Dirichlet elimination for a linear system A u = f.

    Partition DOFs into free (f) and fixed (c). With u_c prescribed:
      A_ff u_f + A_fc u_c = f_f
      => A_ff u_f = f_f - A_fc u_c

    Returns ReducedSystem with A_ff and adjusted RHS.
    """
    A = _to_csr(A)
    ndof = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")
    if f.shape[0] != ndof:
        raise ValueError("f has incompatible size.")

    fixed_dofs = np.array(fixed_dofs, dtype=int)
    free_dofs = _free_dofs_from_fixed(ndof, fixed_dofs)

    # Extract partitions
    A_ff = A[free_dofs, :][:, free_dofs].tocsr()
    A_fc = A[free_dofs, :][:, fixed_dofs].tocsr()

    f_f = f[free_dofs]

    # Adjust RHS: f_f - A_fc * u_c
    u_c = u_fixed[fixed_dofs]
    f_reduced = f_f - (A_fc @ u_c)

    return ReducedSystem(
        A=A_ff,
        f=f_reduced,
        free_dofs=free_dofs,
        fixed_dofs=fixed_dofs,
        u_fixed=u_fixed,
    )


# -----------------------------
# Neumann (loads)
# -----------------------------

def apply_neumann_bcs(
    dof_manager,
    f: np.ndarray,
    bcs: Sequence[NeumannBC],
) -> np.ndarray:
    """
    Adds nodal loads to an existing global load vector f.

    Works for static (real) and harmonic (complex) loads.
    """
    f_out = np.array(f, copy=True)
    for bc in bcs:
        idx = int(dof_manager.get_dof_index(bc.node_id, bc.dof_type))
        f_out[idx] += bc.value
    return f_out


def build_load_vector(
    dof_manager,
    *,
    ndof: int,
    neumann_bcs: Optional[Sequence[NeumannBC]] = None,
    dtype: Union[np.dtype, type] = float,
) -> np.ndarray:
    """
    Convenience builder for a global load vector f from Neumann BCs.
    """
    f = np.zeros(ndof, dtype=dtype)
    if neumann_bcs:
        f = apply_neumann_bcs(dof_manager, f, neumann_bcs)
    return f


# -----------------------------
# Dirichlet elimination: static
# -----------------------------

def apply_dirichlet_static(
    K,
    f: np.ndarray,
    dof_manager,
    dirichlet_bcs: Sequence[DirichletBC],
) -> ReducedSystem:
    """
    Apply Dirichlet BCs to the static system:
      K u = f

    Returns reduced system for unknown DOFs.
    """
    K = _to_csr(K)
    ndof = K.shape[0]
    dtype = np.result_type(K.dtype, f.dtype, np.float64)

    f = np.asarray(f, dtype=dtype)

    fixed_dofs, u_fixed = _collect_dirichlet(
        dof_manager, ndof, dirichlet_bcs, dtype=dtype
    )

    return _reduce_system_by_dirichlet(K, f, fixed_dofs, u_fixed)


# -----------------------------
# Dirichlet elimination: harmonic
# -----------------------------

def build_dynamic_stiffness(
    K,
    M,
    omega: float,
    *,
    C=None,
    dmprat: float = 0.0
) -> csr_matrix:
    """
    Build the (complex) dynamic stiffness:
      Z(omega) = K + i*omega*C - omega^2*M

    If C is None, it is treated as zero.
    """
    # Interpret DMPRAT as structural damping coefficient via g = 2*dmprat
    g = 2.0 * float(dmprat)
    Kc = _to_csr(K).astype(complex) * (1.0 + 1j * g)
    Mc = _to_csr(M).astype(complex)
    if C is None:
        return (Kc - (omega**2) * Mc).tocsr()

    Cc = _to_csr(C).astype(complex)
    return (Kc + 1j * omega * Cc - (omega**2) * Mc).tocsr()


def apply_dirichlet_harmonic(
    K,
    M,
    omega: float,
    f: np.ndarray,
    dof_manager,
    dirichlet_bcs: Sequence[DirichletBC],
    *,
    C=None,
) -> ReducedSystem:
    """
    Apply Dirichlet BCs to the harmonic system:
      (K + i*omega*C - omega^2*M) u = f

    Returns reduced complex system for unknown DOFs.
    """
    Z = build_dynamic_stiffness(K, M, omega, C=C)
    ndof = Z.shape[0]

    # Harmonic analysis is naturally complex-valued.
    f = np.asarray(f, dtype=complex)

    fixed_dofs, u_fixed = _collect_dirichlet(
        dof_manager, ndof, dirichlet_bcs, dtype=np.dtype(complex)
    )

    return _reduce_system_by_dirichlet(Z, f, fixed_dofs, u_fixed)


def apply_dirichlet_harmonic_many_frequencies_prepare(
    K,
    M,
    dof_manager,
    dirichlet_bcs: Sequence[DirichletBC],
    *,
    C=None,
) -> Dict[str, object]:
    """
    Optional helper for frequency sweeps.

    Idea:
      - Precompute the DOF partition and the fixed displacement vector once
      - For each omega, build Z(omega) and reduce using the same partition

    Returns a dict with:
      - fixed_dofs, free_dofs, u_fixed, K, M, C as CSR (not reduced yet)
    """
    K = _to_csr(K)
    M = _to_csr(M)
    Cc = _to_csr(C) if C is not None else None

    ndof = K.shape[0]
    fixed_dofs, u_fixed = _collect_dirichlet(
        dof_manager, ndof, dirichlet_bcs, dtype=np.dtype(complex)
    )
    free_dofs = _free_dofs_from_fixed(ndof, fixed_dofs)

    return {
        "K": K,
        "M": M,
        "C": Cc,
        "fixed_dofs": fixed_dofs,
        "free_dofs": free_dofs,
        "u_fixed": u_fixed,
    }


def apply_dirichlet_harmonic_many_frequencies(
    prepared: Dict[str, object],
    omega: float,
    f: np.ndarray,
) -> ReducedSystem:
    """
    Use output of apply_dirichlet_harmonic_many_frequencies_prepare(...)
    to reduce the harmonic system for a specific omega.
    """
    K: csr_matrix = prepared["K"]  # type: ignore[assignment]
    M: csr_matrix = prepared["M"]  # type: ignore[assignment]
    C: Optional[csr_matrix] = prepared["C"]  # type: ignore[assignment]
    fixed_dofs: np.ndarray = prepared["fixed_dofs"]  # type: ignore[assignment]
    u_fixed: np.ndarray = prepared["u_fixed"]  # type: ignore[assignment]

    Z = build_dynamic_stiffness(K, M, omega, C=C)
    f = np.asarray(f, dtype=complex)

    # Reuse generic reducer
    return _reduce_system_by_dirichlet(Z, f, fixed_dofs, u_fixed)
