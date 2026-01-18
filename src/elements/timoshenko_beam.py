from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence
import numpy as np

from elements.base_element import Element
from matrices.element_matrices import ElementMatrix
#from materials.beam_material import BeamMaterial


@dataclass(frozen=True)
class SectionConstitutive:
    """
    Querschnittskennwerte aus ANSYS CBMX/CBMD.
    S: 6x6 Steifigkeit (generalized forces = S * generalized strains)
    C: 6x6 Masse      (generalized momenta = C * generalized velocities)
    order: Bedeutungsreihenfolge der 6 Komponenten (Strings), z.B.
           ["ex", "kx", "ky", "kz", "gy", "gz"].
    """
    S: np.ndarray
    C: np.ndarray
    order: Tuple[str, str, str, str, str, str] = ("ex", "kx", "ky", "kz", "gy", "gz")

    def __post_init__(self):
        if self.S.shape != (6, 6):
            raise ValueError("Section S muss 6x6 sein.")
        if self.C.shape != (6, 6):
            raise ValueError("Section C muss 6x6 sein.")


class TimoshenkoBeamElement(Element):
    """
    2-Knoten 3D-Timoshenko-Balken:
      - nimmt Sectionmatrizen S (CBMX) und C (CBMD) als Input (je 6x6)
      - erzeugt intern Ke/Me (12x12) via numerischer Integration
      - lokale DOFs pro Knoten: [u, v, w, phix, phiy, phiz]
    """

    DOF_TYPES = ["u", "v", "w", "phix", "phiy", "phiz"]

    def __init__(
        self,
        element_id: int,
        node_ids: List[int],
        L: float,
        section: SectionConstitutive,
        *,
        gauss_points: int = 2,
    ):
        if len(node_ids) != 2:
            raise ValueError("3D-Balkenelement benötigt genau 2 Knoten.")
        if L <= 0.0:
            raise ValueError("Elementlänge L muss > 0 sein.")

        self.id = element_id
        self.node_ids = node_ids
        self.L = float(L)
        self.section = section
        self.gauss_points = int(gauss_points)

        Ke = self._build_stiffness_12x12()
        Me = self._build_mass_12x12()

        self._Ke = ElementMatrix(Ke, self._default_dof_order())
        self._Me = ElementMatrix(Me, self._default_dof_order())

    # --- Element-Interface ---
    def get_stiffness_matrix(self) -> np.ndarray:
        return self._Ke.matrix

    def get_mass_matrix(self) -> np.ndarray:
        return self._Me.matrix

    def get_dof_types(self) -> List[str]:
        return self.DOF_TYPES

    def get_local_dof_mapping(self):
        mapping = []
        for nid in self.node_ids:
            for dof in self.DOF_TYPES:
                mapping.append((nid, dof))
        return mapping

    # --- Intern: DOF-Order ---
    def _default_dof_order(self) -> List[str]:
        out = []
        for i in (1, 2):
            for d in self.DOF_TYPES:
                out.append(f"{d}{i}")
        return out

    # --- Numerik: Gauss-Integration auf [0, L] ---
    def _gauss_rule(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gauss-Legendre auf [-1, 1].
        """
        if n == 1:
            xi = np.array([0.0])
            w = np.array([2.0])
        elif n == 2:
            a = 1.0 / np.sqrt(3.0)
            xi = np.array([-a, a])
            w = np.array([1.0, 1.0])
        elif n == 3:
            xi = np.array([-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)])
            w = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])
        else:
            raise ValueError("Nur gauss_points = 1..3 implementiert.")
        return xi, w

    def _map_xi_to_x(self, xi: float) -> Tuple[float, float]:
        """
        xi in [-1,1] -> x in [0,L]
        Rückgabe: x, jacobian dx/dxi
        """
        x = 0.5 * (xi + 1.0) * self.L
        jac = 0.5 * self.L
        return x, jac

    # --- Shape Functions für 2-Knoten linear ---
    def _N(self, x: float) -> np.ndarray:
        L = self.L
        N1 = 1.0 - x / L
        N2 = x / L
        return np.array([N1, N2], dtype=float)

    def _dN_dx(self) -> np.ndarray:
        L = self.L
        return np.array([-1.0 / L, 1.0 / L], dtype=float)

    # --- Kinematik: B-Matrix (6 x 12) in Basisorder ["ex","kx","ky","kz","gy","gz"] ---
    def _B_base(self, x: float) -> np.ndarray:
        """
        Baut B so, dass:
          eps = B * q
        q = [u1 v1 w1 phix1 phiy1 phiz1 u2 v2 w2 phix2 phiy2 phiz2]^T
        eps_base = [ex, kx, ky, kz, gy, gz]^T
        """
        N = self._N(x)          # (2,)
        dN = self._dN_dx()      # (2,)

        B = np.zeros((6, 12), dtype=float)

        # ex = du/dx
        # u(x) = N1 u1 + N2 u2
        B[0, 0] = dN[0]  # u1
        B[0, 6] = dN[1]  # u2

        # kx = d(phix)/dx
        B[1, 3] = dN[0]  # phix1
        B[1, 9] = dN[1]  # phix2

        # ky = d(phiy)/dx
        B[2, 4] = dN[0]  # phiy1
        B[2, 10] = dN[1] # phiy2

        # kz = d(phiz)/dx
        B[3, 5] = dN[0]  # phiz1
        B[3, 11] = dN[1] # phiz2

        # gy = dv/dx - phiz
        B[4, 1] = dN[0]  # v1
        B[4, 7] = dN[1]  # v2
        B[4, 5] = -N[0]  # phiz1
        B[4, 11] = -N[1] # phiz2

        # gz = dw/dx + phiy
        B[5, 2] = dN[0]  # w1
        B[5, 8] = dN[1]  # w2
        B[5, 4] =  N[0]  # phiy1
        B[5, 10] = N[1]  # phiy2

        return B

    # --- Shape für Mass: Nq (6 x 12) für generalized velocities [u_dot, v_dot, w_dot, phix_dot, phiy_dot, phiz_dot] ---
    def _Nq(self, x: float) -> np.ndarray:
        """
        Baut Nq so, dass:
          v_gen = Nq * q_dot
        v_gen = [u_dot, v_dot, w_dot, phix_dot, phiy_dot, phiz_dot]^T (lokal)
        """
        N = self._N(x)
        Nq = np.zeros((6, 12), dtype=float)

        # u, v, w
        Nq[0, 0] = N[0]
        Nq[0, 6] = N[1]

        Nq[1, 1] = N[0]
        Nq[1, 7] = N[1]

        Nq[2, 2] = N[0]
        Nq[2, 8] = N[1]

        # phix, phiy, phiz
        Nq[3, 3] = N[0]
        Nq[3, 9] = N[1]

        Nq[4, 4] = N[0]
        Nq[4, 10] = N[1]

        Nq[5, 5] = N[0]
        Nq[5, 11] = N[1]

        return Nq

    # --- Order-Mapping: ANSYS/CBMX order -> base order ---
    def _permute_6(self, order: Sequence[str]) -> np.ndarray:
        """
        Liefert Permutationsmatrix P (6x6) so, dass:
          vec_in_base = P * vec_in_order
        bzw. für Matrizen:
          A_in_base = P * A_in_order * P^T
        """
        base = ("ex", "kx", "ky", "kz", "gy", "gz")
        if len(order) != 6:
            raise ValueError("order muss 6 Einträge haben.")
        idx = {name: i for i, name in enumerate(order)}
        P = np.zeros((6, 6), dtype=float)
        for i_base, name in enumerate(base):
            if name not in idx:
                raise ValueError(f"order enthält '{name}' nicht. order={order}")
            P[i_base, idx[name]] = 1.0
        return P

    def _S_in_base(self) -> np.ndarray:
        P = self._permute_6(self.section.order)
        return P @ self.section.S @ P.T

    def _C_in_base(self) -> np.ndarray:
        P = self._permute_6(self.section.order)
        return P @ self.section.C @ P.T

    # --- Build 12x12 matrices ---
    def _build_stiffness_12x12(self) -> np.ndarray:
        xi, wi = self._gauss_rule(self.gauss_points)
        S = self._S_in_base()

        Ke = np.zeros((12, 12), dtype=float)

        for xii, w in zip(xi, wi):
            x, jac = self._map_xi_to_x(float(xii))
            B = self._B_base(x)  # 6x12 (base order)
            Ke += (B.T @ S @ B) * w * jac

        return Ke

    def _build_mass_12x12(self) -> np.ndarray:
        xi, wi = self._gauss_rule(self.gauss_points)
        C = self._C_in_base()

        Me = np.zeros((12, 12), dtype=float)

        for xii, w in zip(xi, wi):
            x, jac = self._map_xi_to_x(float(xii))
            Nq = self._Nq(x)  # 6x12 (u,v,w,phix,phiy,phiz)
            Me += (Nq.T @ C @ Nq) * w * jac

        return Me


class TimoshenkoBeamElement_old(Element):
    """
    Linear 2 node Timoshenko beam element (2D).
    """


    #: DOFs per node (local)
    DOF_TYPES = ["u", "v", "w", "phix", "phiy", "phiz"]

    def __init__(
        self,
        element_id: int,
        node_ids: List[int],
        stiffness_matrix: ElementMatrix,
        mass_matrix: ElementMatrix,
        shear_correction_factors: Tuple[float, float] = (1.0, 1.0),
    ):
        if len(node_ids) != 2:
            raise ValueError(
                "3D-Timoshenko-Balkenelement benötigt genau 2 Knoten."
            )

        self.id = element_id
        self.node_ids = node_ids

        self._Ke = stiffness_matrix
        self._Me = mass_matrix

        # kappa_y, kappa_z (Schubkorrektur)
        self.shear_correction_factors = shear_correction_factors

    def get_stiffness_matrix(self) -> np.ndarray:
        """
        Local element stiffness matrix (12x12).
        """
        return self._Ke#.matrix

    def get_mass_matrix(self) -> np.ndarray:
        """
        Local element mass matrix (12x12).
        """
        return self._Me#.matrix

    def get_dof_types(self) -> List[str]:
        return self.DOF_TYPES

    def get_local_dof_mapping(self) -> List[Tuple[int, str]]:
        """
        Returns local DOF structure in CBMX order.

        [
          (n1, u), (n1, v), (n1, w), (n1, phix), (n1, phiy), (n1, phiz),
          (n2, u), (n2, v), (n2, w), (n2, phix), (n2, phiy), (n2, phiz)
        ]
        """
        mapping = []

        for node_id in self.node_ids:
            for dof in self.DOF_TYPES:
                mapping.append((node_id, dof))

        return mapping

    def validate_matrices(self) -> None:
        """
        Checks dimension and DOF consistency of matrices
        """
        ndofs = len(self.node_ids) * len(self.DOF_TYPES)

        if self._Ke.matrix.shape != (ndofs, ndofs):
            raise ValueError("Stiffness matrix must be of shape 12x12.")

        if self._Me.matrix.shape != (ndofs, ndofs):
            raise ValueError("Mass matrix must be of shape 12x12.")

        if self._Ke.dof_order != self._Me.dof_order:
            raise ValueError(
                "DOF-order of K and M is inconsistent."
            )
