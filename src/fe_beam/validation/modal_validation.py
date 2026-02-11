# validation/modal_validation.py
#
# Modal validation harness for a cantilever beam:
#   1) Build a straight cantilever (x-axis) with Ne beam elements
#   2) Run modal analysis (lowest modes) with robust ARPACK settings
#   3) Compare first bending eigenfrequency against Euler–Bernoulli analytical formula
#   4) Perform mesh convergence study: Ne = 1,2,4,8,16,...
#
# Notes for anisotropic sections:
#   - This harness starts with an ISOTROPIC / diagonal section (no coupling terms),
#     because it is the fastest way to validate:
#       * DOFs, assembly, BC elimination, mass/stiffness consistency, solver stability
#   - Once this passes, introduce anisotropy and then validate orientation/twist.
#
# IMPORTANT:
#   Your current SectionConstitutive / element mass formulation must be consistent.
#   If your element expects section.C in the velocity basis [u_dot,v_dot,w_dot,phix_dot,phiy_dot,phiz_dot],
#   use that. If it expects a different basis, adjust build_section_matrices(...) accordingly.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

# ---- your package imports (adapt paths if needed) ----
from fe_beam.core.mesh import Mesh, Node, ElementConnectivity
from fe_beam.core.dof import DofManager
from fe_beam.core.assembly import assemble_global_matrices
from fe_beam.core.boundary_conditions import DirichletBC
from fe_beam.elements.timoshenko_beam import SectionConstitutive, TimoshenkoBeamElement


# -----------------------------------------------------------------------------
# Analytical Euler–Bernoulli cantilever frequencies
# -----------------------------------------------------------------------------

_BETA_CANTILEVER = np.array([
    1.875104068711961,
    4.694091132974174,
    7.854757438237612,
    10.995540734875466,
    14.13716839104647,
    17.27875965739948,
], dtype=float)


def analytical_cantilever_eb(
    E: float,
    I: float,
    rho: float,
    A: float,
    L: float,
    n_modes: int = 3
) -> np.ndarray:
    """
    Euler–Bernoulli cantilever bending frequencies (Hz) for one bending plane:
        omega_n = (beta_n^2) * sqrt(EI / (rho*A)) / L^2
        f_n = omega_n / (2*pi)
    """
    betas = _BETA_CANTILEVER[:n_modes]
    omega = (betas**2) * np.sqrt(E * I / (rho * A)) / (L**2)
    return omega / (2.0 * np.pi)


# -----------------------------------------------------------------------------
# Section matrix builder (isotropic baseline)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class IsotropicBeamParams:
    # material
    E: float
    nu: float
    rho: float
    # geometry (rectangular default)
    b: float   # width
    h: float   # height
    # shear correction factors
    kappa_y: float = 5.0 / 6.0
    kappa_z: float = 5.0 / 6.0


def rectangular_section_props(b: float, h: float) -> Dict[str, float]:
    """
    Returns A, Iy, Iz, J (approx) for a rectangular section.
    J uses a common approximation; good enough for validation.
    """
    A = b * h
    Iy = (b * h**3) / 12.0
    Iz = (h * b**3) / 12.0

    # Saint-Venant torsion constant approximation for rectangle:
    # J ≈ b*h^3*(1/3 - 0.21*(h/b)*(1 - (h^4)/(12*b^4))) for b>=h
    # Use symmetric handling:
    a = max(b, h)
    t = min(b, h)
    ratio = t / a
    J = a * t**3 * (1.0/3.0 - 0.21*ratio*(1.0 - (ratio**4)/12.0))
    return {"A": A, "Iy": Iy, "Iz": Iz, "J": J}


def build_section_matrices_isotropic(
    p: IsotropicBeamParams,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build baseline diagonal 6x6 section matrices.

    Stiffness (CBMX-like), in base order:
        ["ex", "kx", "ky", "kz", "gy", "gz"]
      N  = EA * ex
      Mx = GJ * kx
      My = EIy * ky
      Mz = EIz * kz
      Vy = kappa_y*GA * gy
      Vz = kappa_z*GA * gz

    Mass (CBMD-like):
      This depends on your element implementation.
      The most common choice for consistent beam inertia per unit length is a 6x6 in the
      generalized velocity basis [u_dot, v_dot, w_dot, phix_dot, phiy_dot, phiz_dot].
      If your element expects that, then define C accordingly and set section.order to that basis.

      HOWEVER, if your current element code uses the SAME "order" for stiffness and mass (ex,kx,ky,kz,gy,gz),
      then you MUST align this to your implementation (fix the element if possible).

    Here we provide a practical default for *velocity basis* and clearly mark the needed hook.
    """
    props = rectangular_section_props(p.b, p.h)
    A, Iy, Iz, J = props["A"], props["Iy"], props["Iz"], props["J"]

    G = p.E / (2.0 * (1.0 + p.nu))

    # 6x6 stiffness in ["ex","kx","ky","kz","gy","gz"]
    S = np.diag([
        p.E * A,            # EA
        G * J,              # GJ
        p.E * Iy,           # EIy
        p.E * Iz,           # EIz
        p.kappa_y * G * A,  # kappa*GA (shear y)
        p.kappa_z * G * A,  # kappa*GA (shear z)
    ]).astype(float)

    # --- Mass matrix (per unit length) in VELOCITY basis [u,v,w,phix,phiy,phiz] ---
    # Translational inertia per length: rho*A
    # Rotational inertia per length about local axes:
    #   about x: rho*Jp, where Jp ~ Iy+Iz (polar second moment)
    #   about y: rho*Iy
    #   about z: rho*Iz
    #
    # This is a reasonable baseline for modal validation.
    Jp = Iy + Iz

    C_vel = np.diag([
        p.rho * A,   # u_dot
        p.rho * A,   # v_dot
        p.rho * A,   # w_dot
        p.rho * Jp,  # phix_dot
        p.rho * Iy,  # phiy_dot
        p.rho * Iz,  # phiz_dot
    ]).astype(float)

    return S, C_vel


# -----------------------------------------------------------------------------
# Robust modal solve helper (shift-invert + scaling + deterministic v0)
# -----------------------------------------------------------------------------

def solve_modal_robust(
    K: csr_matrix,
    M: csr_matrix,
    fixed_dofs: np.ndarray,
    n_modes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve K phi = lambda M phi on free DOFs robustly.

    Returns:
      evals (lambda), modes_full (ndof x n_modes)
    """
    ndof = K.shape[0]
    mask = np.ones(ndof, dtype=bool)
    mask[fixed_dofs] = False
    free = np.nonzero(mask)[0]

    K_ff = K[free, :][:, free].tocsr()
    M_ff = M[free, :][:, free].tocsr()

    # Symmetrize (harmless if already symmetric)
    K_ff = (K_ff + K_ff.T) * 0.5
    M_ff = (M_ff + M_ff.T) * 0.5

    # Diagonal scaling to improve conditioning
    k_scale = float(np.max(np.abs(K_ff.diagonal())))
    m_scale = float(np.max(np.abs(M_ff.diagonal())))
    if k_scale <= 0.0 or m_scale <= 0.0:
        raise ValueError("Non-positive scaling detected (check K/M).")

    K_s = K_ff / k_scale
    M_s = M_ff / m_scale

    n = K_s.shape[0]
    v0 = np.ones(n)  # deterministic start vector

    # Shift-invert around sigma=0 to obtain lowest eigenvalues robustly
    evals_s, evecs = eigsh(
        K_s,
        k=n_modes,
        M=M_s,
        sigma=0.0,
        which="LM",
        v0=v0,
        tol=1e-10,
        maxiter=50000,
    )

    # Back-scale eigenvalues: lambda = lambda_s * (k_scale/m_scale)
    evals = np.asarray(evals_s, dtype=float) * (k_scale / m_scale)

    # Sort ascending
    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]

    # Reconstruct full modes
    modes_full = np.zeros((ndof, n_modes), dtype=float)
    modes_full[free, :] = evecs

    return evals, modes_full


# -----------------------------------------------------------------------------
# Model builder: straight cantilever on x-axis
# -----------------------------------------------------------------------------

def build_cantilever_model(
    L: float,
    n_elems: int,
    section_S: np.ndarray,
    section_C: np.ndarray,
    *,
    ref_vector_global: np.ndarray = np.array([0.0, 0.0, 1.0]),
    twist_psi: float = 0.0,
) -> Tuple[Mesh, List[TimoshenkoBeamElement3DSection], DofManager, csr_matrix, csr_matrix, List[DirichletBC]]:
    """
    Builds mesh, elements, dofs, assembles K, M.
    """
    mesh = Mesh()

    # Nodes along x-axis
    for i in range(n_elems + 1):
        x = (L * i) / n_elems
        mesh.add_node(Node(id=i + 1, x=x, y=0.0, z=0.0))

    # Elements
    for e in range(n_elems):
        mesh.add_element(ElementConnectivity(element_id=e + 1, node_ids=[e + 1, e + 2]))

    # Uniform section
    section = SectionConstitutive(
        S=section_S,
        C=section_C,
        # IMPORTANT: Set this to what your element expects.
        # If your element expects stiffness order ("ex","kx","ky","kz","gy","gz") and
        # mass order ("u","v","w","phix","phiy","phiz"), you should extend SectionConstitutive accordingly.
        #
        # For baseline validation, assume your element is set up to accept velocity basis for C via this order:
        order=("u", "v", "w", "phix", "phiy", "phiz"),
    )

    elements: List[TimoshenkoBeamElement] = []
    for eid in range(1, n_elems + 1):
        n1, n2 = mesh.get_element_nodes(eid)
        Le = float(np.linalg.norm(n2.coordinates() - n1.coordinates()))

        # NOTE: This assumes your element constructor supports x1/x2/ref_vector_global/twist_psi.
        # If your class signature is different, adapt here.
        elem = TimoshenkoBeamElement(
            element_id=eid,
            node_ids=[n1.id, n2.id],
            L=Le,
            section=section,
            gauss_points=2,
        )
        elements.append(elem)

    # DOFs
    dof_manager = DofManager()
    dof_manager.enumerate_dofs(elements)

    # Assemble K,M (global)
    system = assemble_global_matrices(
        mesh=mesh,
        elements=elements,
        dof_manager=dof_manager,
        transform_to_global=True,
        # NOTE: assembly may be set to use element-provided frame; if not, it needs reference_vectors
    )
    K = system.K.tocsr()
    M = system.M.tocsr()

    # Cantilever clamp at node 1 (all 6 dofs)
    dofs = ["u", "v", "w", "phix", "phiy", "phiz"]
    dirichlet_bcs = [DirichletBC(1, dof, 0.0) for dof in dofs]

    return mesh, elements, dof_manager, K, M, dirichlet_bcs


def dirichlet_to_fixed_dofs(dof_manager: DofManager, bcs: Sequence[DirichletBC]) -> np.ndarray:
    fixed = [int(dof_manager.get_dof_index(bc.node_id, bc.dof_type)) for bc in bcs]
    return np.array(sorted(set(fixed)), dtype=int)


# -----------------------------------------------------------------------------
# Main validation routine
# -----------------------------------------------------------------------------

def run_mesh_convergence_validation():
    # --- Beam definition (choose something slender) ---
    L = 1.0
    params = IsotropicBeamParams(
        E=210e9,
        nu=0.3,
        rho=7800.0,
        b=0.02,
        h=0.04,
    )

    section_S, section_C = build_section_matrices_isotropic(params)
    props = rectangular_section_props(params.b, params.h)
    A, Iy, Iz = props["A"], props["Iy"], props["Iz"]

    # Analytical EB for bending about z or y; here pick Iy (bending about y uses Iz, etc.)
    # For a cantilever along x-axis:
    # - transverse displacement in z corresponds to bending about local y -> EI_y ~ E*Iy (if y is "up")
    # - transverse displacement in y corresponds to bending about local z -> EI_z ~ E*Iz
    # In this simple axis setup, we report both analytical targets:
    f_eb_y = analytical_cantilever_eb(params.E, Iy, params.rho, A, L, n_modes=3)
    f_eb_z = analytical_cantilever_eb(params.E, Iz, params.rho, A, L, n_modes=3)

    print("\nEuler–Bernoulli (cantilever) analytical bending frequencies (Hz):")
    print("  Using EI=E*Iy:", f_eb_y)
    print("  Using EI=E*Iz:", f_eb_z)
    print("\nMesh convergence (computed modes will include bending/torsion/axial; identify by inspection):")
    print("  Ne   f1[Hz]     f2[Hz]     f3[Hz]    (first three computed)")

    for ne in [1, 2, 4, 8, 16, 32]:
        mesh, elements, dof_manager, K, M, dbc = build_cantilever_model(
            L=L,
            n_elems=ne,
            section_S=section_S,
            section_C=section_C,
            ref_vector_global=np.array([0.0, 0.0, 1.0]),
            twist_psi=0.0,
        )

        fixed_dofs = dirichlet_to_fixed_dofs(dof_manager, dbc)

        evals, modes = solve_modal_robust(K, M, fixed_dofs, n_modes=6)
        omegas = np.sqrt(np.maximum(evals, 0.0))
        freqs = omegas / (2.0 * np.pi)

        print(f"  {ne:<4d} {freqs[0]:<10.4f} {freqs[1]:<10.4f} {freqs[2]:<10.4f}")

    print("\nInterpretation tips:")
    print("  - Two low modes are typically bending in orthogonal planes.")
    print("  - A torsional mode may appear depending on stiffness/inertia ratios.")
    print("  - Results should approach a limit as Ne increases (monotone or near-monotone).")
    print("  - If frequencies jump around with Ne, check orientation continuity and mass definition.")


if __name__ == "__main__":
    run_mesh_convergence_validation()
