from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class ElementConnectivity:
    '''Contains element topology'''
    element_id: int
    node_ids: List[int]
    
    def __post_int__(self):
        if len(self.node_ids) < 2:
            raise ValueError("An element must have at least two nodes!")

@dataclass(frozen=True)
class Node:
    '''Mesh node in 3d space'''
    id: int 
    z: float
    x: float 
    y: float 
    
    def coordinates(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    def __str__(self):
        return f'NodeID: {self.id}\nx: {self.x}\ny: {self.y}\nz: {self.z}'

class Mesh:
    '''FE mesh defined by nodes and element connectivities'''
    def __init__(self,nodes:Dict={}, element_connectivity:Dict={} ):
        self.nodes: Dict[int, Node] = nodes
        self.elements: Dict[int, ElementConnectivity] = element_connectivity
        
    def add_element(self, connectivity: ElementConnectivity) -> None:
        
        if connectivity.element_id in self.elements.keys():
            raise KeyError(
                f"Element {connectivity.element_id} does already exist."
                    )

        for nid in connectivity.node_ids:
            if nid not in self.nodes:
                raise KeyError(f"Node-ID {nid} not defined in mesh.")

        self.elements[connectivity.element_id] = connectivity

    def add_node(self, node: Node) -> None:
        '''Adds Node to mesh'''
        if node.id in self.nodes.keys():
            raise KeyError(f'Node {node.id} does already exist!')
        self.nodes[node.id] = node

    def show(self) -> None:
        fig, axs = plt.subplots(nrows=3)
        legend_entry = False
        for elementid, conn in self.elements.items():
            node_ids = conn.node_ids
            x1 = self.nodes[node_ids[0]].coordinates()[0]
            y1 = self.nodes[node_ids[0]].coordinates()[1]
            z1 = self.nodes[node_ids[0]].coordinates()[2]
            x2 = self.nodes[node_ids[1]].coordinates()[0]
            y2 = self.nodes[node_ids[1]].coordinates()[1]
            z2 = self.nodes[node_ids[1]].coordinates()[2]
            if not legend_entry:
                axs[0].plot([x1, x2], [y1, y2], ls='-', color='green', label='Element')
                legend_entry=True
            else:
                axs[0].plot([x1, x2], [y1, y2], ls='-', color='green')
            axs[1].plot([z1, z2], [y1, y2], ls='-', color='green')
            axs[2].plot([z1, z2], [x1, x2], ls='-', color='green')
            axs[0].text( (x1 + x2) / 2, (y1 + y2) / 2, f'{elementid}', color='green', ha='center', va='bottom')
            axs[1].text( (z1 + z2) / 2, (y1 + y2) / 2, f'{elementid}', color='green', ha='center', va='bottom')
            axs[2].text( (z1 + z2) / 2, (x1 + x2) / 2, f'{elementid}', color='green', ha='center', va='bottom')

        legend_entry = False
        for nodeid, node in self.nodes.items():
            if not legend_entry:
                axs[0].plot(node.coordinates()[0], node.coordinates()[1], 'o', ls='',color='red', label='Node')
                legend_entry=True
            else:
                axs[0].plot(node.coordinates()[0], node.coordinates()[1], 'o', ls='', color='red')
            axs[1].plot(node.coordinates()[2], node.coordinates()[0], 'o', ls='', color='red')
            axs[2].plot(node.coordinates()[2], node.coordinates()[1], 'o', ls='', color='red')

            axs[0].text( node.coordinates()[0], node.coordinates()[1], f'{nodeid}', color='red', ha='center', va='bottom')
            axs[1].text( node.coordinates()[2], node.coordinates()[1], f'{nodeid}', color='red', ha='center', va='bottom')
            axs[2].text( node.coordinates()[2], node.coordinates()[0], f'{nodeid}', color='red', ha='center', va='bottom')

        axs[0].set_xlabel('X [m]')
        axs[0].set_ylabel('Y [m]')
        axs[1].set_xlabel('Z [m]')
        axs[1].set_ylabel('X [m]')
        axs[2].set_xlabel('Z [m]')
        axs[2].set_ylabel('Y [m]')
        axs[0].legend()
        plt.show()
        pass

    def get_element_direction(self, element_id: int) -> np.ndarray:
        """
        Calculates unit vector in element direction.
        """
        nodes = self.get_element_nodes(element_id)

        if len(nodes) != 2:
            raise ValueError(
                "Only defined for linear elements."
            )

        x1 = nodes[0].coordinates()
        x2 = nodes[1].coordinates()

        v = x2 - x1
        L = np.linalg.norm(v)

        if L <= 0.0:
            raise ValueError("Element length  not positive.")

        return v / L

    def get_element_length(self, element_id: int) -> float:
        """
        Calculates length of linear beam element
        """
        nodes = self.get_element_nodes(element_id)

        if len(nodes) != 2:
            raise ValueError(
                "Elementlänge ist nur für 2-Knoten-Balkenelemente definiert."
            )

        x1 = nodes[0].coordinates()
        x2 = nodes[1].coordinates()

        return float(np.linalg.norm(x2 - x1))

    def get_element_nodes(self, element_id: int) -> List[Node]:
        """Returns list of Node objects connected to element element_id

        Args:
            element_id (int)

        Raises:
            KeyError: raises if element is not existing

        Returns:
            List[Node]: List of Node objects connected to element
        """        '''
        '''
        if element_id not in self.elements:
            raise KeyError(f"Element {element_id} does not exist.")

        conn = self.elements[element_id]
        return [self.nodes[nid] for nid in conn.node_ids]

    def number_of_nodes(self) -> int:
        return len(self.nodes)

    def number_of_elements(self) -> int:
        return len(self.elements)

    def get_all_nodes(self) -> List[Node]:
        return list(self.nodes.values())

    def get_all_elements(self) -> List[ElementConnectivity]:
        return list(self.elements.values())

    def show(self):
        node_coordinates = np.row_stack([node.coordinates() for node in self.nodes.values()])
        fig, axs = plt.subplots(nrows=3)

        for i , element in enumerate( self.elements.values()):
            n1 = element.node_ids[0]
            n2 = element.node_ids[1]
            v1 = self.nodes[n1].coordinates()
            v2 = self.nodes[n2].coordinates()

            if i == 0:
                axs[0].plot([v1[0], v2[0]], [v1[1], v2[1]], ls='-', color='green', label='Elements')
            else:
                axs[0].plot([v1[0], v2[0]], [v1[1], v2[1]], ls='-', color='green')
            axs[1].plot([v1[2], v2[2]], [v1[0], v2[0]], ls='-', color='green')
            axs[2].plot([v1[2], v2[2]], [v1[1], v2[1]], ls='-', color='green')
            # Display element ids
            axs[0].text( (v1[0] + v2[0]) / 2., (v1[0] + v2[0]) / 2., f'{element.element_id}', color='green', va='bottom')
            axs[1].text( (v1[2] + v2[2]) / 2., (v1[0] + v2[0]) / 2., f'{element.element_id}', color='green', va='bottom')
            axs[2].text( (v1[2] + v2[2]) / 2., (v1[1] + v2[1]) / 2., f'{element.element_id}', color='green', va='bottom')

        axs[0].plot(node_coordinates[:, 0], node_coordinates[:, 1], 'x' , color='red', label='Nodes')
        axs[0].set_xlabel('X [m]')
        axs[0].set_ylabel('Y [m]')
        axs[0].legend()
        axs[0].grid()

        axs[1].plot(node_coordinates[:, 2], node_coordinates[:, 0], 'x', color='red')
        axs[1].set_xlabel('Z [m]')
        axs[1].set_ylabel('X [m]')
        axs[1].grid()

        axs[2].plot(node_coordinates[:, 2], node_coordinates[:, 1], 'x', color='red' )
        axs[2].set_xlabel('Z [m]')
        axs[2].set_ylabel('Y [m]')
        axs[2].grid()
        for node_id, node in self.nodes.items():
            axs[0].text(node.coordinates()[0], node.coordinates()[1], f'{node_id}', color='red', va='top')
            axs[1].text(node.coordinates()[2], node.coordinates()[0], f'{node_id}', color='red', va='top')
            axs[2].text(node.coordinates()[2], node.coordinates()[1], f'{node_id}', color='red', va='top')
        plt.show()
        
def read_ansys_mesh(path):
    '''Reads mesh from apdl file''' 
    pass

def create_test_mesh():
    z =  np.linspace(0, 87.5, endpoint=True)

    mesh = Mesh()
    element = 0
    nodes  = {k: Node(k, v, x=0, y=0) for k, v in enumerate(z)}

    for v in nodes.values():
        mesh.add_node(v)
    
    for element_id, node_id in enumerate( list(nodes.keys())[:-1:] ):
        element = ElementConnectivity(element_id, [node_id, node_id+1])
        mesh.add_element(element)
    mesh.show()

if __name__=='__main__':
    create_test_mesh()


