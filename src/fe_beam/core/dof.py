from typing import Dict, List, Tuple, Iterable

class DegreeOfFreedom:
    """
    Represents single degree of freedom
    """

    def __init__(self, node_id: int, dof_type: str):
        self.node_id = node_id
        self.dof_type = dof_type

    def key(self) -> Tuple[int, str]:
        return (self.node_id, self.dof_type)

    def __repr__(self) -> str:
        return f"DOF(node={self.node_id}, type='{self.dof_type}')"

class DofManager:
    """
    Handles global DOFs
    """
    def __init__(self):
        self._dof_map: Dict[Tuple[int, str], int] = {}
        self._inverse_map: Dict[int, Tuple[int, str]] = {}
        self._is_enumerated: bool = False

    def enumerate_dofs(self, elements: Iterable) -> None:
        """
        Creates consistent global DOF enumeration from a list of elements.
        """
        if self._is_enumerated:
            raise RuntimeError("DOFs wurden bereits nummeriert.")

        index = 0

        for element in elements:
            for node_id, dof_type in element.get_local_dof_mapping():
                key = (node_id, dof_type)
                if key not in self._dof_map:
                    self._dof_map[key] = index
                    self._inverse_map[index] = key
                    index += 1

        self._is_enumerated = True

    def get_dof_index(self, node_id: int, dof_type: str) -> int:
        """
        Returns global index of DOF 
        """
        key = (node_id, dof_type)

        if key not in self._dof_map:
            raise KeyError(f"DOF {key} ist nicht definiert.")

        return self._dof_map[key]

    def get_dof_from_index(self, index: int) -> Tuple[int, str]:
        if index not in self._inverse_map:
            raise KeyError(f"Ungültiger DOF-Index {index}.")
        return self._inverse_map[index]

    def get_element_dof_indices(self, element) -> List[int]:
        """
        Returns global DOF indices of an element in local order.
        """
        indices = []

        for node_id, dof_type in element.get_local_dof_mapping():
            indices.append(self.get_dof_index(node_id, dof_type))

        return indices

    def iter_dofs(self):
        for index, key in self._inverse_map.items():
            yield index, key

    def number_of_dofs(self) -> int:
        return len(self._dof_map)

def use_example():
    pass
    # 1. Elemente erzeugen
    #elements = [beam1, beam2, beam3]

    # 2. DOFs global nummerieren
    #dof_manager = DofManager()
    #dof_manager.enumerate_dofs(elements)

    # 3. Assemblierung vorbereiten
    #for e in elements:
        #idx = dof_manager.get_element_dof_indices(e)
#
        #pass


