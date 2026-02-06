import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from dataclasses import dataclass
from fe_beam.utils.helpers import NXbeam, distribute_between_nodes, BeamProperties
from fe_beam.core.mesh import Mesh, Node, ElementConnectivity
from fe_beam.core.dof import DofManager
from fe_beam.core.boundary_conditions import DirichletBC, apply_dirichlet_harmonic, NeumannBC, build_load_vector
from fe_beam.elements.point_masses import add_point_masses_to_M, PointMass
from fe_beam.core.solvers import ModalSolver, HarmonicSolver
from fe_beam.core.postprocessing import compute_element_end_forces_harmonic, get_force_indices
#from fatiguetestoptimizer.Settings import UserInput, Constraints
from scipy.sparse import issparse

COMPONENT_INDEX_LOCAL = get_force_indices()

def max_asym(A):
    if issparse(A):
        D = A - A.T
        return 0.0 if D.nnz == 0 else float(abs(D).max())
    else:
        return float(np.max(np.abs(A - A.T)))


class Constraints:
    def __init__(self):
        self.n_masses = 2
        self.fixed_dofs = {}
        self.min_distance = 2.
        self.masses_at_lengths = [[[0., 80.]], [[0, 10E3]]]
        self.u_allowed = 1.
    
    def get_u_allowed(self):
        return self.u_allowed

    def get_mass_distance(self):
        return self.min_distance

class UserInput:
    def __init__(self):
        self.target_moments_path = r'/home/alex/Projects/FatigueTestOptimizer/src/fatiguetestoptimizer/assets/Input_Examples/NR87p5/NR87p5_Edgewise_Testloads.txt'
        self.eval_start = 0.0
        self.eval_end  = 43.0
        self.freq_percentage = .99
        self.test_direction = 'Flap'
        self.damprat = 0.
        pass

def calculate_clamp_mass(l: float, m_extra: float, unit='kg') -> float:
    """Calculate total mass of load clamp 

    Args:
        l (float or array- like): exciter position in meters 
        m_extra (float or array- like): Extra mass in t
        unit (str, optional): Defines the unit (kg or t) in which the mass is returened. Defaults to 'kg'.

    Returns:
        float or array like: Total clamp mass
    """    
    #''''''
    #m_loadclamp = x[2] * 1E3 + 0.4803 * x[0]**2. - 69.132 * x[0] + 3303.9
    if isinstance(m_extra, float):
        check = m_extra>1E3
    else:
        check = any( m_extra>1E3 ) 
    if check:
        print(f'WARNING: m_extra should be given in t:\nm_extra {m_extra} it will be converted to t'*100)
        m_extra*=1E-3

    m_base = 0.1699 * l**2. - 44.281 * l + 2891.4
    m_clamp = m_extra * 1E3
    if unit=='kg':
        return  m_base + m_clamp
    elif unit=='t':
        return  (m_base + m_clamp) / 10


    pass

def create_point_masses(masses: List, mesh: Mesh, lengthwise_coordinate: int):
    point_masses = []
    for (position, mass) in masses:
        node_ids, values = distribute_between_nodes(position, mass, mesh, lengthwise_coordinate)
        point_masses.append(PointMass(node_ids[0], values[0]))
        point_masses.append(PointMass(node_ids[1], values[1]))
    return point_masses


@dataclass
class FatigueTestSingle: 
    fe_problem: NXbeam
    user_input: UserInput
    constraints: Constraints
    
    def distance_constraint(self, x: List)->bool:
        """Check if there is sufficient distance between load clamps and extra masses.

        Args:
            x (dict): degrees of freedom of optimization problem

        Returns:
            bool: True if constraint violated
        """
        positions = [x['L_F']] + [x[f'L{i}'] for i in range(1, self.n_masses+1)]
        labels = ['L_F'] + [f'L{i}' for i in range(1, self.n_masses+1)]

        d_min = self.constraints.get_mass_distance()
        for labela, xa in zip( labels, positions):
            for labelb, xb in zip( labels, positions):
                if labela!=labelb and np.abs(xa - xb)<d_min:
                    return True
        return False

        pass

    def excitation_constraint(self, x: list, displacement)->bool:
        """Check constraint of max displacement amplitude at force introduction point. 
           Return True if constraint violated.
        Args:
            x (dict): degrees of freedom of optimization problem
            displacement (array): Array containing displacement solution and  coordinates.

        Returns:
            bool: True if constraint violated
        """
        l_coordinates = displacement[:, 0]
        u_excitation = np.abs(displacement[:, 1])
        u_at_excitation = np.interp((x['L_F'],), l_coordinates, u_excitation)[0]
        u_ratio = np.abs(u_at_excitation / self.constraints.get_u_allowed())
        if  u_ratio > 1.:
            print(f'Max excitation constraint violated\n u_max/u_allowed: {u_ratio}')
        return u_ratio>1

    def mass_constraints(self, x: List)->bool:
        """Check if masses comply to position dependent max given by the user.
        Args:
            x (list): degrees of freedom of optimization problem

        Returns:
            bool: True if constraint violated
        """
        l_ranges = self.constraints.masses_at_lengths[0]
        m_ranges = self.constraints.masses_at_lengths[1]
        mass_positions = [x['L_F']] + [x[f'L{i}'] for i in range(1,self.n_masses+1)]#list(x[-2 * self.n_masses::2])
        masses = [calculate_clamp_mass(x['L_F'], x['mF']) * 1E-3] + [x[f'm{i}'] for i in range(1,self.n_masses+1)]#list(x[-2 * self.n_masses + 1::2])

        for l_m, m in zip(mass_positions, masses):
            for n, l_r in enumerate(l_ranges):
                if l_m>=l_r[0] and l_m<l_r[1] and m>m_ranges[n][1]:
                    return True
        return False

    def min_moment_constraint(self, resulting_moment)->bool:
        """Check if the test moment is larger then the target moment at every position.

        Args:
            resulting_moment (array): Array containing bending moment amplitudes and coordinates

        Returns:
            bool: True if constraint is violated
        """
        abs_ratio = np.abs(resulting_moment[:, 1]) / np.abs(self.target_moments[:, 1])
        return_val = np.min(abs_ratio[self.eval_mask])
        if  return_val<1.:
            violation_pos = resulting_moment[abs_ratio==return_val, 0][0]
            print(f'Minimal moment constraint violated:\n M_test / M_target = {return_val:.4e} @ {violation_pos:.2f} m')
        return return_val<1.

    def read_target_moments(self, target_moments_path:str, plot=False):
        '''Read target moment ranges from file'''
        model_coordinates = np.array([node.coordinates()[self.fe_problem.lengthwise_coordinate] for node in self.fe_problem.nodes])
        model_coordinates = np.sort(model_coordinates)

        target_moments = np.loadtxt(target_moments_path, skiprows=2)
        target_moments[:, 1] = target_moments[:, 1] * 500.
        target_moments = target_moments[target_moments[:, 0].argsort()]
        print('Target moments are multiplied by 1000 for unit system consistency! kNm -> Nm')
        print('Target moments are devided by 2: range -> amplitude')

        if np.all(np.diff(model_coordinates)>0) and np.all(np.diff(target_moments[:, 0])>0):
            y_interp = np.interp(model_coordinates, target_moments[:, 0], target_moments[:, 1])
            interpolated_moments = np.vstack((model_coordinates, y_interp)).T
        else:
            print( model_coordinates )
            print( target_moments[:, 0] )
            raise AssertionError('x-values not sorted before interpolation.')

        if plot:
            plt.plot(model_coordinates, [5] * len(model_coordinates), 'x')
            plt.plot(interpolated_moments[0], interpolated_moments[1], 'o')
            plt.plot(target_moments[:, 0], target_moments[:, 1], 'o', label='original')
            plt.legend()
            plt.show()
        return interpolated_moments

    def get_displacement_amplitudes(self, u_full: np.ndarray, dof: str)->np.ndarray:
        nodes = self.fe_problem.mesh.get_all_nodes()
        node_ids = []
        tmp_l = []
        for node in nodes:
            node_ids.append(node.id)
            tmp_l.append(node.coordinates()[self.ids['L']])
        l = np.array(tmp_l)
        idx = np.array([self.fe_problem._dof_manager.get_dof_index(nid, dof) for nid in node_ids], dtype=int)
        amp = np.abs(u_full[idx])
        order = np.argsort(l)

        return np.column_stack([l[order], amp[order]])#, np.array(node_ids, dtype=int)[order]

    def get_moment_amplitudes(self, u_full: np.ndarray, omega: float, damprat: float, component: str, average: bool=True):
        elements = self.fe_problem.beam_elements
        end_forces = compute_element_end_forces_harmonic(
            mesh = self.fe_problem.mesh,
            elements= elements,
            dof_manager= self.fe_problem._dof_manager,
            u_full=u_full,
            omega=float(omega),
            structural_damping_g=damprat,
            reference_vectors=self.fe_problem.orientation_vectors
        )
        segs = []
        for e in elements:
            n1, n2 = e.node_ids
            x1 = float(self.fe_problem.mesh.nodes[n1].coordinates()[self.fe_problem.lengthwise_coordinate])
            x2 = float(self.fe_problem.mesh.nodes[n2].coordinates()[self.fe_problem.lengthwise_coordinate])

            fl = end_forces[e.id].local  # complex 12-vector
            segs.append((x1, x2, fl))
        i1, i2 = COMPONENT_INDEX_LOCAL[component]

        xs = []
        ys = []

        # Sort segments by x1 for a nice diagram
        segs_sorted = sorted(segs, key=lambda s: min(s[0], s[1]))

        for (x1, x2, fl) in segs_sorted:
            q1 = fl[i1]
            q2 = fl[i2]
            y1, y2 = abs(q1), abs(q2)

            xs.extend([x1, x2])
            ys.extend([y1, y2])
        # average at nodes 
        moments_unaveraged = np.vstack([xs, ys]).T
        if not average:
            return moments_unaveraged
        else:
            unique_l = sorted(list( set(moments_unaveraged[:, 0]) ))
            moments_averaged = np.zeros((len(unique_l), 2))
            for i, ul in enumerate( unique_l ):
                mask = moments_unaveraged[:, 0] == ul
                moments_averaged[i, 0] = ul
                moments_averaged[i, 1] = np.average(moments_unaveraged[mask, 1])
            return moments_averaged

    def run_analysis(self, x):
        K, M = self.fe_problem.get_system_matrices()

        #1. Create masses in kg
        m_loadclamp = calculate_clamp_mass( x['L_F'], x['mF'] )
        masses = [[x['L_F'], m_loadclamp]] +  [[ x[f'L{i+1}'], x[f'm{i+1}']*1E3] for i in range(self.n_masses)]
        point_masses = create_point_masses(masses, self.fe_problem.mesh, self.fe_problem.lengthwise_coordinate)
        M_w_pm = add_point_masses_to_M(M, self.fe_problem._dof_manager, point_masses)

        #2. Create fixed_support
        dofs = ['u', 'v', 'w', 'phix', 'phiy', 'phiz'] # will get them from somewhere else later
        fixed_support_bcs = [DirichletBC(self.fe_problem.start_node, dof) for dof in dofs]

        # 3. Run modal analysis
        t1 = time.time()
        ndof = self.fe_problem._dof_manager.number_of_dofs()
        modal = ModalSolver()
        modal_result =  modal.solve(
            K = K.copy(),
            M=M_w_pm.copy(),
            dof_manager=self.fe_problem._dof_manager,
            dirichlet_bcs=fixed_support_bcs,
            n_modes=2
        )
        omega_test = self.user_input.freq_percentage * modal_result.omegas[self.ids['frequency']]

        # 4. Harmonic response analysis
        force_components = ['N', 'Fy', 'Fz','T', 'My', 'Mz']

        force = [0.]*3
        force[self.ids['F']] = x['Fampl']
        node_ids, values = distribute_between_nodes(x['L_F'], force, self.fe_problem.mesh, self.ids['L'])
        neumann_bcs = [NeumannBC(node_ids[0], dofs[self.ids['F']], values[0][self.ids['F']]),
                       NeumannBC(node_ids[1], dofs[self.ids['F']], values[1][self.ids['F']])]
        f_harm = build_load_vector(
            dof_manager=self.fe_problem._dof_manager,
            ndof=ndof,
            neumann_bcs=neumann_bcs,
            dtype=complex
        )
        harm_solver = HarmonicSolver()

        harm_result = harm_solver.solve_frequency(
        K=K.copy(),
        M=M_w_pm.copy(),
        omega=omega_test,
        f=f_harm,
        dof_manager=self.fe_problem._dof_manager,
        dirichlet_bcs=fixed_support_bcs,
        C=None,          # no viscous damping
        eta=0.0,         # no structural damping
        )
        u_full = harm_result.u
        displacement_amplitudes = self.get_displacement_amplitudes(u_full, dofs[self.ids['U']])
        
        moment_amplitudes = self.get_moment_amplitudes(u_full, omega_test, self.user_input.damprat, force_components[3+self.ids['M']])
        return displacement_amplitudes, moment_amplitudes
        moment_amplitudesX = self.get_moment_amplitudes(u_full, omega_test, self.user_input.damprat, force_components[3+0])
        moment_amplitudesY = self.get_moment_amplitudes(u_full, omega_test, self.user_input.damprat, force_components[3+1])
        moment_amplitudesZ = self.get_moment_amplitudes(u_full, omega_test, self.user_input.damprat, force_components[3+2])
        plt.plot(moment_amplitudesX[:, 0], moment_amplitudesX[:, 1], label='X')
        plt.plot(moment_amplitudesY[:, 0], moment_amplitudesY[:, 1], label='Y')
        plt.plot(moment_amplitudesZ[:, 0], moment_amplitudesZ[:, 1], label='Z')
        plt.legend()
        plt.show()

    def __post_init__(self):
        self.n_masses = self.constraints.n_masses
        self.mesh = self.fe_problem.mesh
        dof_order = ['L_F', 'Fampl', 'mF']

        for i in range(self.n_masses):
            dof_order+=[f'L{i + 1}', f'm{i + 1}']
        self.dof_var = [d for d in dof_order if d not in self.constraints.fixed_dofs.keys()]

        self.target_moments = self.read_target_moments(self.user_input.target_moments_path)
        self.eval_mask = (self.target_moments[:, 0] >= self.user_input.eval_start) & (self.target_moments[:, 0] <= self.user_input.eval_end)

        self.ids = {'L': self.fe_problem.lengthwise_coordinate }
        if self.user_input.test_direction == 'Flap':
            self.ids['frequency'] = 1
        else: 
            self.ids['frequency'] = 0

        # Directions to be used based on global coordinate system and test direction
        if self.fe_problem.lengthwise_coordinate==0 and self.user_input.test_direction=='Flap':
            self.ids['U'] = 2
            self.ids['M'] = 1
        elif self.fe_problem.lengthwise_coordinate==0 and self.user_input.test_direction=='Edge':
            self.ids['U'] = 1
            self.ids['M'] = 2
        elif self.fe_problem.lengthwise_coordinate==2 and self.user_input.test_direction=='Flap':
            self.ids['U'] = 0
            self.ids['M'] = 1
        elif self.fe_problem.lengthwise_coordinate==2 and self.user_input.test_direction=='Edge':
            self.ids['U'] = 1
            self.ids['M'] = 0
        
        self.ids['F'] = self.ids['U']

    def __call__(self, x_tup):
        x_var = {k: v for k, v in zip(self.dof_var, x_tup)}
        x_dict = {**x_var, **self.constraints.fixed_dofs}

        self.mass_constraints(x_dict)
        if self.distance_constraint(x_dict) or self.mass_constraints(x_dict):
            return np.inf

        resulting_disp, resulting_moments = self.run_analysis(x_dict )
        # Calculate normalised least square error between target moment and given model

        # Max ratio between result and target
        max_overload = np.max( np.abs( resulting_moments[self.eval_mask, 1] ) / np.abs( self.target_moments[self.eval_mask, 1] ) )

        # List of evaluated constraints
        if self.excitation_constraint(x_dict, resulting_disp) or self.min_moment_constraint(resulting_moments):
            return np.inf
            
        print(max_overload)
        return max_overload

if __name__=='__main__':
    apdl_file = r'/home/alex/Projects/FatigueTestOptimizer/src/fatiguetestoptimizer/assets/Input_Examples/NR87p5/NR87p5_S21_L75_woFinish.apdl'
    beam_geometry = BeamProperties(apdl_file)
    nodes = [Node(k, v[0], v[1], v[2]) for k, v in beam_geometry.nodes.items()]
    connectivities = [ElementConnectivity(k, beam_geometry.elements[k])
                  for k in sorted(beam_geometry.elements.keys())]
    #sectional_properties = {}
    #for element_id, sec_props in beam_geometry.section_properties.items():
        #cbmx = np.zeros((6,6))
        #cbmd = np.zeros((6,6))
        #for i in range(6):
            #cbmx[i, i:] = sec_props['CBMX'][i]
            #cbmd[i, i:] = sec_props['CBMD'][i]
        #sectional_properties[element_id] = SectionConstitutive(cbmx, cbmd)
    nx_beam = NXbeam(nodes, connectivities, beam_geometry.section_properties)
    constraints = Constraints()
    user_input = UserInput()
    objective = FatigueTestSingle(nx_beam, user_input, constraints)
    test_input = {'L_F': 40, 'Fampl': 1E2, 'mF': 1E3, 'L1': 15, 'm1': 500, 'L2': 60, 'm2': 100}
    objective(test_input.values())
