# solvers.py
#
# Solvers for:
#   - Static analysis:                K u = f
#   - Harmonic (frequency) analysis:  (K + i*omega*C - omega^2*M) u = f(omega)
#   - Modal analysis:                 K phi = lambda M phi
#
# This module is designed to work with:
#   - assembly.py (AssembledSystem)
#   - boundary_conditions.py (Dirichlet/Neumann application + reduction)
#
# English comments are used by request.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from scipy.sparse import csr_matrix, issparse
    from scipy.sparse.linalg import spsolve, splu, eigsh
except ImportError as e:
    raise ImportError(
        "scipy is required for solvers (scipy.sparse, scipy.sparse.linalg)."
    ) from e

from fe_beam.core.boundary_conditions import (
    DirichletBC,
    ReducedSystem,
    apply_dirichlet_static,
    apply_dirichlet_harmonic,
    build_dynamic_stiffness,
)


Number = Union[float, complex, np.floating, np.complexfloating]


# -----------------------------
# Results containers
# -----------------------------

@dataclass(frozen=True)
class StaticResult:
    u: np.ndarray  # full displacement vector (ndof,)
    reactions: Optional[np.ndarray] = None  # optional, see compute_reactions()


@dataclass(frozen=True)
class HarmonicResult:
    omega: float
    u: np.ndarray  # full complex displacement vector (ndof,)


@dataclass(frozen=True)
class HarmonicSweepResult:
    omegas: np.ndarray          # (n_omega,)
    U: np.ndarray               # (n_omega, ndof) complex
    # Optional convenience: e.g. response at a single dof extracted externally.


@dataclass(frozen=True)
class ModalResult:
    omegas: np.ndarray          # natural circular frequencies (rad/s), (n_modes,)
    frequencies_hz: np.ndarray  # natural frequencies (Hz), (n_modes,)
    modes: np.ndarray           # mode shapes (ndof, n_modes) in full DOF space


# -----------------------------
# Helper utilities
# -----------------------------

def _to_csr(A) -> csr_matrix:
    if issparse(A):
        return A.tocsr()
    return csr_matrix(A)


def compute_reactions_static(
    K,
    u_full: np.ndarray,
    f_full: np.ndarray,
    fixed_dofs: np.ndarray,
) -> np.ndarray:
    """
    Compute reaction forces at Dirichlet DOFs for a static solution:
      r = K u - f
    Returns a full-size reaction vector (ndof,), nonzero primarily at constrained DOFs.
    """
    K = _to_csr(K)
    r = (K @ u_full) - f_full
    # Optionally zero out free dofs for clarity:
    mask = np.ones_like(r, dtype=bool)
    mask[fixed_dofs] = False
    r[mask] = 0.0
    return np.asarray(r)


# -----------------------------
# Static solver
# -----------------------------

class StaticSolver:
    """
    Static linear solver:
      K u = f
    with exact Dirichlet enforcement via elimination (partitioning).
    """

    def solve(
        self,
        K,
        f: np.ndarray,
        dof_manager,
        dirichlet_bcs: Sequence[DirichletBC],
        *,
        compute_reactions: bool = True,
    ) -> StaticResult:
        K = _to_csr(K)
        f = np.asarray(f, dtype=np.result_type(K.dtype, f.dtype, np.float64))

        reduced: ReducedSystem = apply_dirichlet_static(
            K=K, f=f, dof_manager=dof_manager, dirichlet_bcs=dirichlet_bcs
        )

        # Solve reduced system
        u_free = spsolve(reduced.A, reduced.f)
        u_full = reduced.reconstruct_full_solution(u_free)

        reactions = None
        if compute_reactions:
            reactions = compute_reactions_static(
                K=K,
                u_full=u_full,
                f_full=f,
                fixed_dofs=reduced.fixed_dofs,
            )

        return StaticResult(u=u_full, reactions=reactions)


# -----------------------------
# Harmonic solver
# -----------------------------

class HarmonicSolver:
    """
    Harmonic response solver:
      (K + i*omega*C - omega^2*M) u(omega) = f(omega)

    Dirichlet constraints enforced by elimination for each omega.
    """

    def solve_frequency(
        self,
        K,
        M,
        omega: float,
        f: np.ndarray,
        dof_manager,
        dirichlet_bcs: Sequence[DirichletBC],
        *,
        C=None,
        eta = 0
    ) -> HarmonicResult:
        f = np.asarray(f, dtype=complex)

        reduced = apply_dirichlet_harmonic(
            K=complex(1, eta) * K, M=M, omega=float(omega), f=f,
            dof_manager=dof_manager,
            dirichlet_bcs=dirichlet_bcs,
            C=C,
        )

        u_free = spsolve(reduced.A, reduced.f)
        u_full = reduced.reconstruct_full_solution(u_free)

        return HarmonicResult(omega=float(omega), u=u_full)

    def solve_sweep(
        self,
        K,
        M,
        omegas: Sequence[float],
        f_of_omega,
        dof_manager,
        dirichlet_bcs: Sequence[DirichletBC],
        *,
        C=None,
        reuse_factorization: bool = False,
    ) -> HarmonicSweepResult:
        """
        Frequency sweep.

        Parameters
        ----------
        omegas:
            Iterable of circular frequencies (rad/s).
        f_of_omega:
            Callable omega -> full load vector f(omega) (complex ndarray).
            Allows frequency-dependent excitation.
        reuse_factorization:
            If True, attempts to reuse LU for identical reduced matrices.
            Note: in general Z(omega) changes with omega, so reuse is only helpful
            if you sweep repeated frequencies or special cases.
        """
        omegas_arr = np.asarray(list(omegas), dtype=float)
        if omegas_arr.ndim != 1:
            raise ValueError("omegas must be a 1D sequence.")

        # Determine ndof from dof_manager
        ndof = int(dof_manager.number_of_dofs())
        U = np.zeros((len(omegas_arr), ndof), dtype=complex)

        last_A = None
        last_lu = None

        for i, om in enumerate(omegas_arr):
            f = np.asarray(f_of_omega(float(om)), dtype=complex)
            res = apply_dirichlet_harmonic(
                K=K, M=M, omega=float(om), f=f,
                dof_manager=dof_manager,
                dirichlet_bcs=dirichlet_bcs,
                C=C,
            )

            # Optional LU reuse (rarely beneficial unless matrices repeat)
            if reuse_factorization and last_A is not None and (res.A != last_A).nnz == 0:
                lu = last_lu
            else:
                lu = splu(res.A.tocsc())
                last_A = res.A
                last_lu = lu

            u_free = lu.solve(res.f)
            U[i, :] = res.reconstruct_full_solution(u_free)

        return HarmonicSweepResult(omegas=omegas_arr, U=U)


# -----------------------------
# Modal solver
# -----------------------------

class ModalSolver:
    """
    Modal analysis:
      K phi = lambda M phi
      omega = sqrt(lambda)
      f = omega / (2*pi)

    Dirichlet constraints enforced by elimination on both K and M.

    Notes:
      - Uses scipy.sparse.linalg.eigsh (symmetric generalized eigenproblem).
      - Requires K and M to be (numerically) symmetric and M positive definite
        on the free DOFs.
    """

    def solve(
        self,
        K,
        M,
        dof_manager,
        dirichlet_bcs: Sequence[DirichletBC],
        *,
        n_modes: int = 6,
        sigma: Optional[float] = None,
        which: str = "SM",
        mass_normalize: bool = True,
    ) -> ModalResult:
        K = _to_csr(K)
        M = _to_csr(M)

        ndof = K.shape[0]
        if M.shape != (ndof, ndof):
            raise ValueError("K and M must have the same shape.")

        # Reduce K and M with Dirichlet elimination.
        # For eigenproblems, we enforce u_fixed = 0 (typical for modal shapes).
        # If non-zero prescribed displacements exist, modal interpretation becomes nonstandard.
        reduced_K = apply_dirichlet_static(
            K=K, f=np.zeros(ndof, dtype=float),
            dof_manager=dof_manager,
            dirichlet_bcs=dirichlet_bcs,
        )
        # Reuse the same fixed/free partition, but reduce M similarly:
        fixed = reduced_K.fixed_dofs
        free = reduced_K.free_dofs

        K_ff = K[free, :][:, free].tocsr()
        M_ff = M[free, :][:, free].tocsr()
        k_scale = float(np.max(np.abs(K_ff.diagonal())))
        m_scale = float(np.max(np.abs(M_ff.diagonal())))

        K_s = K_ff / k_scale
        M_s = M_ff / m_scale
        # Solve generalized eigenproblem on free DOFs:
        #   K_ff v = lambda M_ff v
        # Using eigsh: smallest magnitude eigenvalues by default with which="SM".
        if sigma is not None:
            # Shift-invert around sigma to target eigenvalues near sigma.
            evals_s, evecs = eigsh(K_s, k=n_modes, M=M_s, sigma=float(sigma), which="LM")
        else:
            n = K_s.shape[0]
            v0 = np.ones(n)  # deterministic

            # Shift-invert around sigma=0 targets the lowest eigenvalues robustly
            evals_s, evecs = eigsh(
                K_s, k=n_modes, M=M_s,
                sigma=0.0, which="LM",
                v0=v0,
                tol=1e-10,
                maxiter=50000,
            )
            #evals, evecs = eigsh(K_ff, k=n_modes, M=M_ff, which=which)
        evals = evals_s * (k_scale / m_scale)
        # Clean and sort eigenvalues (ascending)
        evals = np.asarray(evals, dtype=float)
        evecs = np.asarray(evecs, dtype=float)

        order = np.argsort(evals)
        evals = evals[order]
        evecs = evecs[:, order]

        # Convert to natural circular frequencies (rad/s)
        # Guard against tiny negative values from numerical noise.
        evals[evals < 0.0] = 0.0
        omegas = np.sqrt(evals)
        freqs_hz = omegas / (2.0 * np.pi)

        # Reconstruct full DOF mode shapes
        modes_full = np.zeros((ndof, n_modes), dtype=float)
        modes_full[free, :] = evecs
        # fixed DOFs remain zero

        if mass_normalize:
            # Mass-normalize each mode: phi^T M phi = 1 (on full DOFs)
            for j in range(n_modes):
                phi = modes_full[:, j]
                mnorm = float(phi.T @ (M @ phi))
                if mnorm > 0.0:
                    modes_full[:, j] = phi / np.sqrt(mnorm)

        return ModalResult(omegas=omegas, frequencies_hz=freqs_hz, modes=modes_full)
