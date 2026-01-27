# point_masses.py
#
# Utilities to account for lumped (concentrated) point masses at nodes.
#
# Supported:
#   - translational lumped mass m added to DOFs (u, v, w)
#   - optional rotational inertia tensor I (3x3) added to DOFs (phix, phiy, phiz)
#   - optional translation-rotation coupling due to eccentricity r (3,) between node
#     reference point and mass center (rigid attachment assumption)
#
# Notes:
#   - Point masses contribute to the global mass matrix M (not to stiffness K).
#   - For harmonic and modal analyses, they are essential.
#   - For static analyses, they do not affect the solution (unless gravity loads
#     are modeled separately as external forces).
#
# English comments are used by request.

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Union

import numpy as np

try:
    from scipy.sparse import csr_matrix, lil_matrix, issparse
except ImportError as e:
    raise ImportError(
        "scipy is required for point mass handling (scipy.sparse)."
    ) from e


Number = Union[float, np.floating]


@dataclass(frozen=True)
class PointMass:
    """
    Concentrated mass attached to a node.

    Parameters
    ----------
    node_id:
        Node at which the point mass is attached.

    mass:
        Translational lumped mass m [kg]. Added to (u, v, w).

    inertia:
        Optional 3x3 rotational inertia tensor [kg*m^2] about the node reference point
        (or about the mass center if eccentricity is provided together with proper transformation).
        Added to (phix, phiy, phiz). If None, only translational mass is added.

        The inertia must be expressed in GLOBAL coordinates.

    eccentricity:
        Optional vector r (3,) [m] from node reference point to mass center in GLOBAL coordinates.
        If provided, translation-rotation coupling terms are added:
            M_tr = -m * S(r)
            M_rt =  m * S(r)
        and the rotational inertia about the node can be augmented if you provide inertia about the mass center.
        (See add_point_masses_to_M docstring for details.)
    """
    node_id: int
    mass: Number
    inertia: Optional[np.ndarray] = None
    eccentricity: Optional[np.ndarray] = None

    def __post_init__(self):
        if float(self.mass) < 0.0:
            raise ValueError("PointMass.mass must be non-negative.")

        if self.inertia is not None:
            I = np.asarray(self.inertia, dtype=float)
            if I.shape != (3, 3):
                raise ValueError("PointMass.inertia must be a 3x3 tensor.")

        if self.eccentricity is not None:
            r = np.asarray(self.eccentricity, dtype=float).reshape(-1)
            if r.shape != (3,):
                raise ValueError("PointMass.eccentricity must be a 3-vector.")


def skew(r: np.ndarray) -> np.ndarray:
    """
    Skew-symmetric matrix S(r) such that S(r) * a = r x a.
    """
    r = np.asarray(r, dtype=float).reshape(3)
    rx, ry, rz = float(r[0]), float(r[1]), float(r[2])
    return np.array(
        [
            [0.0, -rz,  ry],
            [rz,  0.0, -rx],
            [-ry, rx,  0.0],
        ],
        dtype=float,
    )


def add_point_masses_to_M(
    M,
    dof_manager,
    point_masses: Iterable[PointMass],
    *,
    dof_translation: Sequence[str] = ("u", "v", "w"),
    dof_rotation: Sequence[str] = ("phix", "phiy", "phiz"),
    assume_inertia_about_node: bool = True,
) -> csr_matrix:
    """
    Add point masses to the global mass matrix.

    Translational part:
      M_tt += m * I3

    Rotational part (if inertia provided):
      M_rr += I

    Translation-rotation coupling (if eccentricity r provided):
      For a rigidly attached point mass at offset r, using small-angle kinematics:
         v_mass = v_node + omega x r
      => kinetic energy introduces coupling terms:
         M_tr = -m * S(r)
         M_rt =  m * S(r)
      and (if you provide inertia about the mass center) the inertia about the node is:
         I_node = I_cm + m * (||r||^2 I3 - r r^T)   (parallel axis theorem)

    Parameters
    ----------
    M:
        Global mass matrix (ndof x ndof), sparse or dense.

    dof_manager:
        Must provide get_dof_index(node_id, dof_type).

    point_masses:
        Iterable of PointMass.

    dof_translation / dof_rotation:
        DOF names used in your model.

    assume_inertia_about_node:
        If True (default), the provided inertia tensor is assumed to already be about the node point.
        If False and eccentricity is provided, then inertia is treated as about the mass center and
        will be shifted to the node using the parallel axis theorem.

    Returns
    -------
    M_out:
        Updated global mass matrix in CSR format.
    """
    if issparse(M):
        M_out = M.tocsr().tolil()
    else:
        M_out = lil_matrix(np.asarray(M, dtype=float))

    for pm in point_masses:
        nid = pm.node_id
        m = float(pm.mass)

        # Indices: translations
        it = [int(dof_manager.get_dof_index(nid, d)) for d in dof_translation]
        # Add translational lumped mass
        for a in range(3):
            M_out[it[a], it[a]] += m

        # Indices: rotations
        ir = [int(dof_manager.get_dof_index(nid, d)) for d in dof_rotation]

        # Optional eccentricity coupling
        r = None
        if pm.eccentricity is not None:
            r = np.asarray(pm.eccentricity, dtype=float).reshape(3)
            Sr = skew(r)
            # Coupling blocks:
            # [ M_tt  M_tr ]
            # [ M_rt  M_rr ]
            # with M_tr = -m S(r), M_rt = +m S(r)
            M_tr = -m * Sr
            M_rt =  m * Sr

            for a in range(3):
                for b in range(3):
                    M_out[it[a], ir[b]] += M_tr[a, b]
                    M_out[ir[a], it[b]] += M_rt[a, b]

        # Optional rotational inertia
        if pm.inertia is not None:
            I = np.asarray(pm.inertia, dtype=float)

            # If inertia is given about CM and eccentricity is provided, shift to node
            if (not assume_inertia_about_node) and (r is not None):
                rrT = np.outer(r, r)
                I_node = I + m * ((float(r @ r) * np.eye(3)) - rrT)
            else:
                I_node = I

            for a in range(3):
                for b in range(3):
                    M_out[ir[a], ir[b]] += I_node[a, b]

    return M_out.tocsr()

if __name__=='__main__':
    # typical use
    point_masses = [
        PointMass(node_id=2, mass=5.0),  # 5 kg am freien Ende
        ]
    I = np.diag([0.01, 0.02, 0.02])
    point_masses = [
    PointMass(node_id=2, mass=5.0, inertia=I),
]

    M = None
    dof_manager=None

    M = add_point_masses_to_M(
    M=M,
    dof_manager=dof_manager,
    point_masses=point_masses,
)
    pass