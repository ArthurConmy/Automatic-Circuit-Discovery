from acdc.acdc_utils import TorchIndex, Edge, EdgeType
from typing import List, Dict, Optional, Tuple, Union, Set, Callable, TypeVar, Iterable, Any

class TLACDCInterpNode:
    """Represents one node in the computational graph, similar to ACDCInterpNode from the rust_circuit code
    
    But WARNING this has nodes closer to the input tokens as *parents* of nodes closer to the output tokens, the opposite of the rust_circuit code
    
    Params:
        name: name of the node
        index: the index of the tensor that this node represents
        mode: how we deal with this node when we bump into it as a parent of another node. Addition: it's summed to make up the child. Direct_computation: it's the sole node used to compute the child. Off: it's not the parent of a child ever."""
        
    def __init__(self, name: str, index: TorchIndex):
        
        self.name = name
        self.index = index
        
        self.parents: List["TLACDCInterpNode"] = []
        self.children: List["TLACDCInterpNode"] = []

    def _add_child(self, child_node: "TLACDCInterpNode"):
        """Use the method on TLACDCCorrespondence instead of this one"""
        self.children.append(child_node)

    def _add_parent(self, parent_node: "TLACDCInterpNode"):
        """Use the method on TLACDCCorrespondence instead of this one"""
        self.parents.append(parent_node)

    def __repr__(self):
        return f"TLACDCInterpNode({self.name}, {self.index})"

    def __str__(self) -> str:
        index_str = "" if len(self.index.hashable_tuple) < 3 else f"_{self.index.hashable_tuple[2]}"
        return f"{self.name}{self.index}"
