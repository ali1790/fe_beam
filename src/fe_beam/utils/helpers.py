import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Tuple
from fe_beam.core.mesh import Mesh, Node, ElementConnectivity
from fe_beam.elements.timoshenko_beam import SectionConstitutive, TimoshenkoBeamElement
from fe_beam.core.dof import DofManager
from fe_beam.core.assembly import assemble_global_matrices

def create_mesh(nodes: List[Node], connectivities: List[ElementConnectivity]):
    mesh = Mesh()
    for node in nodes:
        mesh.add_node(node)
    for connectivity in connectivities:
        mesh.add_element(connectivity)
    return mesh

def assign_properties_to_elements(mesh: Mesh, sectional_properties: Dict[int, SectionConstitutive]):
    '''
    Docstring for assign_properties_to_elements
    
    :param mesh: Mesh object
    :type mesh: Mesh
    :param sectional_properties: Dictionary linking element_ids and sectional properties
    :type sectional_properties: Dict[element_id, SectionConstitutive]
    '''

    beam_elements = []
    for element in mesh.get_all_elements():
        element_id = element.element_id
        beam_elements.append( 
            TimoshenkoBeamElement(
                element_id,
                element.node_ids,
                mesh.get_element_length(element_id),
                sectional_properties[element_id]
            )
        )
    return beam_elements

def create_system_matrices(mesh: Mesh, elements: List[TimoshenkoBeamElement], orientation_vectors: Dict[int, np.ndarray]=None):
    dof_manager = DofManager()
    dof_manager.enumerate_dofs(elements)
    system = assemble_global_matrices(mesh, elements, dof_manager, reference_vectors=orientation_vectors)
    return system
    pass

def setup_problem(nodes: List[Node], connectivities: List[ElementConnectivity], sectional_properties: Dict[int, SectionConstitutive]):
    mesh = create_mesh(nodes, connectivities)
    beam_elements = assign_properties_to_elements(mesh, sectional_properties)
    system = create_system_matrices(mesh, beam_elements)
    return system

def distribute_between_nodes(position: float, value_to_distribute, mesh: Mesh, lengthwise_coordinate: int):
    """Finds closest nodes to  and distributes load value to  nodes

    Args:
        position (float): Desired position 
        value_to_distribute (float or list): Value(s) to distribute
        lengthwise coordinate: id of lengthwise coordinate

    Returns:
        id_near (list): ids of two nodes with load introduction
        node1_force (list): Amplitude of load components at node1
        node2_force (list): Amplitude of load components at node2
    """
    all_nodes = mesh.get_all_nodes()

    tmp = np.array([ [node.id] + list(node.coordinates()) for node in all_nodes])
    tmp = tmp[tmp[:, 1+lengthwise_coordinate].argsort()]

    if position >= np.max( tmp[: , 1 + lengthwise_coordinate] ):
        node_ids = ( int(tmp[-2, 0] ), int( tmp[-1, 0] ) )
        if isinstance(value_to_distribute, list):
            node1_val = [0]*3
        else:
            node1_val = 0.
        node2_val = value_to_distribute
        return node_ids, node1_val, node2_val
    elif np.any(tmp[:, 1 + lengthwise_coordinate] == position):
        row_id = np.where(tmp[:, 1 + lengthwise_coordinate] == position)[0][0]
        node_ids = [tmp[row_id-1, 0], tmp[row_id, 0]]
        if isinstance(value_to_distribute, list):
            node1_val = [0.]*3
        else:
            node1_val = 0.
        node2_val = value_to_distribute
        return node_ids, [node1_val, node2_val]
    else:
        lower_nodes = tmp[ tmp[:, 1 + lengthwise_coordinate] < position, :]
        upper_nodes = tmp[ tmp[:, 1 + lengthwise_coordinate] > position, :]

        lower_node = lower_nodes[-1, :]
        upper_node = upper_nodes[0, :]
        node_ids = ( int( lower_node[0]), int(upper_node[0]) )

        l1 = lower_node[ 1 + lengthwise_coordinate]
        l2 = upper_node[ 1 + lengthwise_coordinate]
        ratio = (position - l1) / ( l2 - l1 )
        if isinstance(value_to_distribute, list):
            node1_val = [ ( 1. - ratio ) * f for f in value_to_distribute]
            node2_val = [ ratio * f for f in value_to_distribute]
        else:
            node1_val = ( 1. - ratio ) * value_to_distribute
            node2_val = ratio * value_to_distribute
        return node_ids, [node1_val, node2_val]
        # Find 2 closest nodes 
        # Check if one is to the left and other to the right

class BeamProperties(object):
    '''Reads geometry and BECAS material parameters from .cdb-files and stores the them 
       in dictionaries.'''
    def __init__(self, source_file) -> None:
        self.source_file = source_file

        if source_file.split('.')[-1] == 'apdl':
            print(f'Reading beam properties from {source_file}')
            self.read_section_properties_apdl()
            self.read_beam_elements_apdl()
        elif source_file.split('.')[-1] == 'cdb':
            print(f'Reading beam properties from {source_file}')
            self.read_section_properties_cdb()
            self.read_beam_elements_cdb()
        self.read_beam_nodes()

    def read_beam_elements_apdl(self) -> None:
        with open(self.source_file, 'r', encoding='utf8') as i: content = i.readlines()
        self.elements = {}

        eblock_start = False
        eblock_end = False
        i=0
        while not eblock_end:
            line = content[i]
            if eblock_start and not eblock_end:
                tmp = line.split()
                if len(tmp) == 14:
                    sec_num = int(tmp[3])
                    if sec_num in self.section_properties.keys():
                        self.elements[sec_num] = [int(tmp[11]), int(tmp[12])]
                    i+=1
                else:
                    eblock_end = True
            elif 'EBLOCK' in line:
                eblock_start = True
                i+=2
            else:
                i+=1

    def read_beam_elements_cdb(self)->None:
        """Reads element definitions from .cdb file..
        """
        n_trigger = 'EBLOCK'

        e_start = False
        e_end = False

        self.elements = {}
        element_id=1
        with open(self._epath, 'r', encoding='utf8') as i:
            line = i.readline()
            while not e_end:
                if e_start and not e_end and len( line.split() ) != 14:
                    e_end = True
                elif e_start and not e_end and '!' not in line:
                    linelist = line.split()
                    #self.elements[int(linelist[-4])] = [int( linelist[-3] ), int( linelist[-2] )]
                    self.elements[int(element_id)] = [int( linelist[-3] ), int( linelist[-2] )]
                    element_id+=1
                    #self.elements[int(linelist[3])] = [int( linelist[-3] ), int( linelist[-2] )]
                    line = i.readline()
                elif n_trigger in line:
                    line = i.readline()
                    line = i.readline()
                    e_start = True
                else:
                    line = i.readline()

    def read_beam_nodes(self)->None:
        """Reads coordinates of nodes from .cdb file..
        """
        n_trigger = 'NBLOCK'
    
        needed_nodes = []
        for v in self.elements.values():
            needed_nodes+=v
        n_start = False
        n_end = False

        self.nodes = {}
        with open(self.source_file, 'r', encoding='utf8') as i:
            line = i.readline()
            while not n_end:
                if n_start and not n_end and len( line.split() ) != 6:
                    n_end = True
                elif n_start and not n_end and '!' not in line:
                    linelist = line.split()
                    node_id = int(linelist[0])
                    if node_id in needed_nodes:
                        self.nodes[node_id] = [float(s) for s in linelist[3:] ]
                    line = i.readline()
                elif n_trigger in line:
                    line = i.readline()
                    line = i.readline()
                    n_start = True
                else:
                    line = i.readline()

    def read_section_properties_apdl(self) -> None:
        with open(self.source_file, 'r', encoding='utf8') as i: content = i.readlines()
        tmp_section_properties = {}
        seclines = {}
        for n, c in enumerate(content):
            if 'sectype' in c and '!' not in c.split('sectype')[0]:
                secnum = int(c.replace(' ','').split(',')[1])
                seclines[secnum] = n
        
        for secnum, secline in seclines.items():
            tmp_section_properties[secnum] = {'CBMX': [None]*6, 'CBMD': [None]*6}
            end = False
            i = 1
            while not end:
                line = content[secline + i].rstrip().replace(' ','').split(',')
                if line[0] in ['CBMX', 'CBMD']:
                    tag = line[0]
                    entry_id = int(line[1])
                    tmp_section_properties[secnum][tag][entry_id - 1] = [float(x) for x in line[2:2+7-entry_id]]
                    i+=1
                else:
                    end = True

        self.section_properties = {}
        for element_id, sec_props in tmp_section_properties.items():
            cbmx = np.zeros((6,6))
            cbmd = np.zeros((6,6))
            for i in range(6):
                cbmx[i, i:] = sec_props['CBMX'][i]
                cbmd[i, i:] = sec_props['CBMD'][i]
            self.section_properties[element_id] = SectionConstitutive(cbmx, cbmd)

    def read_section_properties_cdb(self):
        """Reads section properties from .cdb file..
        """
        self.section_properties = {}
        with open(self._secprops_path, 'r') as i: content = i.readlines()
        cx_raw = []
        cd_raw = []
        for n, c in enumerate(content):
            if 'CBMX' in c and '!' not in c and len(c.split(','))>1:
                if int(c.replace(' ','').split(',')[1])<7:
                    cx_raw.append(c.replace(' ','').split(',')[1:])
            if 'CBMD' in c and '!' not in c and len(c.split(','))>1:
                if int(c.replace(' ','').split(',')[1])<7:
                    cd_raw.append(c.replace(' ','').split(',')[1:])
            
        for i in range(0, len(cx_raw), 6):
            section_number = int((i+6)/6)
            cbmx_i = []
            cbmd_i = []
            for cx, cd in zip(cx_raw[i:i+6], cd_raw[i:i+6] ):
                cbmx_i.append([float(x) for x in cx[1:8-int(cx[0])]])
                cbmd_i.append([float(x) for x in cd[1:8-int(cd[0])]])
        cbmx = np.zeros((6,6))
        cbmd = np.zeros((6,6))
        for i in range(6):
            cbmx[i, i:] = cbmx_i[i]
            cbmd[i, i:] = cbmd_i[i]
        print(cbmx)
        self.section_properties[section_number] = SectionConstitutive(cbmx, cbmd)

    def get_geometry(self):
        return self.nodes, self.elements, self.section_properties
    
    def get_beam_limits(self):
        """Gets  min/max coordinates of the beam

        Returns:
            _type_: _description_
        """        
        v_max = [-np.inf]*3
        v_min = [np.inf]*3
        for nodepair in self.elements.values():
            for nodeid in nodepair:
                v = self.nodes[nodeid]
                for i in range(3):
                    if v[i]>v_max[i]:
                        v_max[i] = v[i]
                    if v[i]<v_min[i]:
                        v_min[i] = v[i]
        return (v_min[0], v_max[0], v_min[1], v_max[1], v_min[2], v_max[2] )

    def get_mass(self):
        mass = 0
        for k, v in self.section_properties.items():
            mass_per_length = v['CBMD'][0][0]
            node1, node2 = self.elements[k]
            v1 = self.nodes[node1]
            v2 = self.nodes[node2]
            element_length = np.sqrt( (v1[0] - v2[0])**2. + (v1[1] - v2[1])**2. + (v1[2] - v2[2])**2.)
            mass+=mass_per_length * element_length
        return mass

    def get_static_moment(self, length_id):
        g = 9.81
        L = np.array( sorted( [ v[length_id] for v in self.nodes.values() ] ) )
        M = np.zeros(L.shape)
        for k, v in self.section_properties.items():
            mass_per_length = v['CBMD'][0][0]
            node1, node2 = self.elements[k]
            v1 = self.nodes[node1]
            v2 = self.nodes[node2]
            element_length = np.sqrt( (v1[0] - v2[0])**2. + (v1[1] - v2[1])**2. + (v1[2] - v2[2])**2.)
            center = .5 * ( v1[length_id] + v2[length_id] )
            F = element_length * mass_per_length * g
            M+=np.array([ F * max([(center - xi), 0]) for xi in L])
        return np.column_stack([L, M])

@dataclass
class NXbeam:
    nodes: List[Node]
    connectivities: List[ElementConnectivity]
    section_properties: Dict[int, SectionConstitutive]

    def find_lengthwise_coordinate(self):
        '''
        Finds lengthwise coordinate of beam model in global coordinates
        '''
        tmp_coordinates = np.array([node.coordinates() for node in self.nodes])        
        ranges = tmp_coordinates.max(axis=0) - tmp_coordinates.min(axis=0)
        self.lengthwise_coordinate = np.argmax(ranges)
        return self.lengthwise_coordinate
    
    def find_start_end(self):
        start = np.inf
        end = -np.inf
        start_node = None
        end_node = None
        for node in self.nodes:
            l = node.coordinates()[self.lengthwise_coordinate]
            if l<=start:
                start = l
                start_node = node.id
            if l>=end:
                end = l
                end_node = node.id
        self.start_node = start_node
        self.end_node = end_node

    def get_system_matrices(self):
        return self.K_global.copy(), self.M_global.copy()

    def create_additional_masses(self, additional_masses, positions):
        pass

    def apply_harmonic_force():
        pass

    def __post_init__(self):
        self.find_lengthwise_coordinate()
        self.find_start_end()
        self.mesh = Mesh()
        for node in tqdm(self.nodes, desc='Adding nodes:', unit='nodes'):
            self.mesh.add_node(node)
        self.beam_elements = []
        self.orientation_vectors = {}
        if self.lengthwise_coordinate == 0:
            print('foo')
            orientation_vector = np.array([0., 1., 0.])
        elif self.lengthwise_coordinate == 2:
            print('bar')
            orientation_vector = np.array([0., -1., 0.])

        for connectivity in tqdm(self.connectivities, desc='Creating elements:', unit='elements'):
            self.mesh.add_element(connectivity)
            element_id = connectivity.element_id
            self.orientation_vectors[element_id] = orientation_vector
            self.beam_elements.append( 
                TimoshenkoBeamElement(
                    element_id,
                    connectivity.node_ids,
                    self.mesh.get_element_length(element_id),
                    self.section_properties[element_id]
                )
            )
        print('Creating global system matrices')
        self._dof_manager = DofManager()
        self._dof_manager.enumerate_dofs(self.beam_elements)
        system = create_system_matrices(self.mesh, self.beam_elements, self.orientation_vectors)
        self.K_global = system.K
        self.M_global = system.M
        pass

        

if __name__=='__main__':
    # Define geometry
    l_beam = 10.
    n_nodes = 3
    x = np.zeros(n_nodes)
    y = np.zeros(n_nodes)
    z = np.linspace(0, l_beam, endpoint=True, num=n_nodes)

    # nodes and connectivities
    nodes = [ Node(k, v[0], v[1], v[2]) for k, v in enumerate(zip(x, y, z) )  ]
    connectivities = [ElementConnectivity(k, [k, k+1]) for k in range(n_nodes-1)]

    # Material properties
    cbmx = np.diag([1e7, 1e5, 1e5, 1e5, 5e4, 5e4])
    cbmd = np.diag([10.0, 1.0, 1.0, 1.0, 0.5, 0.5])
    sectional_properties = {k: SectionConstitutive(cbmx, cbmd) for k in range(n_nodes-1)}

    nx_beam = NXbeam(nodes, connectivities, sectional_properties)

    print(nx_beam.start_node)
    print(nx_beam.end_node)
    #print(nx_beam.mesh.get_limits())
    #mins, maxs = nx_beam.mesh.get_limits()
    #print(mins)
    #print(maxs)
    #print( distribute_between_nodes(2.5, 1., nx_beam.mesh, 2))

    #system = setup_problem(nodes, connectivities, sectional_properties)
