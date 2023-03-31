import collections
from collections import defaultdict
from enum import Enum
from typing import Any, Literal, Dict, Tuple, Union, List, Optional

class OrderedDefaultdict(collections.OrderedDict):
    """ A defaultdict with OrderedDict as its base class. 
    Thanks to https://stackoverflow.com/a/6190500/1090562"""

    def __init__(self, default_factory=None, *args, **kwargs):
        if not (default_factory is None or callable(default_factory)):
            raise TypeError('first argument must be callable or None')
        super(OrderedDefaultdict, self).__init__(*args, **kwargs)
        self.default_factory = default_factory  # called by __missing__()

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key,)
        self[key] = value = self.default_factory()
        return value

    def __repr__(self):  # Optional.
        return '%s(%r, %r)' % (self.__class__.__name__, self.default_factory, self.items())

class EdgeType(Enum):
    """Property of edges in the computational graph - either 
    
    1. The parent adds to the child (e.g residual stream)
    2. The *single* child is a function of and only of the parent (e.g the value hooked by hook_q is a function of what hook_q_input saves)
    3. Generally like 2. but where there are generally multiple parents, so it's hard to separate effects. E.g hook_result is a function of hook_q, _k and _v but once we've computed hook_result it's difficult to edit one input"""

    ADDITION = 0
    DIRECT_COMPUTATION = 1
    ALWAYS_INCLUDED = 2

class Edge:
    def __init__(
        self,
        edge_type: EdgeType,
        present: bool = True,
    ):
        self.edge_type = edge_type
        self.present = present


# TODO attrs.frozen???
class TorchIndex:
    """There is not a clean bijection between things we 
    want in the computational graph, and things that are hooked
    (e.g hook_result covers all heads in a layer)
    
    `HookReference`s are essentially indices that say which part of the tensor is being affected. 
    
    E.g (slice(None), slice(None), 3) means index [:, :, 3]"""

    def __init__(
        self, 
        list_of_things_in_tuple
    ):
        for arg in list_of_things_in_tuple: # TODO write this less verbosely. Just typehint + check typeguard saves us??
            if type(arg) in [type(None), int]:
                continue
            else:
                assert isinstance(arg, list)
                assert all([type(x) == int for x in arg])

        self.as_index = tuple([slice(None) if x is None else x for x in list_of_things_in_tuple])
        self.hashable_tuple = tuple(list_of_things_in_tuple)

    def __hash__(self):
        return hash(self.hashable_tuple)

    def __repr__(self) -> str:
        return f"TorchIndex({self.hashable_tuple})"

class TLACDCInterpNode:
    """Represents one node in the computational graph, similar to ACDCInterpNode from the rust_circuit code
    
    But WARNING this has nodes closer to the input tokens as *parents* of nodes closer to the output tokens, the opposite of the rust_circuit code
    
    Params:
        name: name of the node
        index: the index of the tensor that this node represents
        mode: how we deal with this node when we bump into it as a parent of another node. Addition: it's summed to make up the child. Direct_computation: it's the sole node used to compute the child. Off: it's not the parent of a child ever."""
        
    def __init__(self, name: str, index: TorchIndex, mode: Literal["addition", "direct_computation", "off"] = "off"):
        
        self.name = name
        self.index = index
        self.mode = mode
        
        self.parents: List["TLACDCInterpNode"] = []
        self.children: List["TLACDCInterpNode"] = []

    def _add_child(self, child_node: "TLACDCInterpNode"):
        """Use the method on TLACDCCorrespondence instead of this one"""
        self.children.append(child_node)

    def _add_parent(self, parent_node: "TLACDCInterpNode"):
        """Use the method on TLACDCCorrespondence instead of this one"""
        self.parents.append(parent_node)

class TLACDCCorrespondence:
    """Stores the full computational graph, similar to ACDCCorrespondence from the rust_circuit code"""
        
    def __init__(self):
        self.graph: Dict[str, List[TLACDCInterpNode]] = defaultdict(list) # TODO maybe another

    def nodes(self) -> List[TLACDCInterpNode]:
        """Concatenate all nodes in the graph"""
        return [node for node_list in self.graph.values() for node in node_list]
    

    def add_node(self, node: TLACDCInterpNode):
        assert node not in self.graph, f"Node {node} already in graph"
        self.graph[node.name].append(node)

    def add_edge(
        self,
        parent_node: TLACDCInterpNode,
        child_node: TLACDCInterpNode,
    ):
        if parent_node not in self.nodes(): # TODO could be slow ???
            self.add_node(parent_node)
        if child_node not in self.nodes():
            self.add_node(child_node)
        
        parent_node._add_child(child_node)
        child_node._add_parent(parent_node)


def make_nd_dict(end_type, n = 3) -> Any:
    """Make biiig default dicts : ) : )"""

    if n not in [3, 4]:
        raise NotImplementedError("Only implemented for 3/4")
        
    if n == 3:
        return OrderedDefaultdict(lambda: defaultdict(lambda: defaultdict(end_type)))

    if n == 4:
        return OrderedDefaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(end_type))))
    
def count_no_edges(graph):
    num_edges = 0
    for receiver_name in graph.keys():
        for receiver_slice_tuple in graph[receiver_name].keys():
            for sender_hook_name in graph[receiver_name][receiver_slice_tuple].keys():
                for sender_slice_tuple in graph[receiver_name][receiver_slice_tuple][sender_hook_name]:
                    edge = graph[receiver_name][receiver_slice_tuple][sender_hook_name][sender_slice_tuple]

                    if not edge.edge_type == EdgeType.ALWAYS_INCLUDED and edge.present:
                        num_edges += 1
    return num_edges