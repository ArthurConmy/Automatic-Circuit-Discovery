from transformer_lens.acdc.TLACDCInterpNode import TLACDCInterpNode
from collections import OrderedDict
from transformer_lens.acdc.utils import TorchIndex, Edge, EdgeType, OrderedDefaultdict, make_nd_dict
from typing import List, Dict, Optional, Tuple, Union, Set, Callable, TypeVar, Iterable, Any


class TLACDCCorrespondence:
    """Stores the full computational graph, similar to ACDCCorrespondence from the rust_circuit code"""
        
    def __init__(self):
        self.graph: OrderedDict[str, OrderedDict[TorchIndex, TLACDCInterpNode]] = OrderedDefaultdict(OrderedDict) # TODO rename "nodes?"
 
        self.edges: OrderedDict[str, OrderedDict[TorchIndex, OrderedDict[str, OrderedDict[TorchIndex, Optional[Edge]]]]] = make_nd_dict(end_type=None, n=4)

    def nodes(self) -> List[TLACDCInterpNode]:
        """Concatenate all nodes in the graph"""
        return [node for by_index_list in self.graph.values() for node in by_index_list.values()]
    
    def all_edges(self) -> Dict[Tuple[str, TorchIndex, str, TorchIndex], Edge]:
        """Concatenate all edges in the graph"""
        
        big_dict = {}

        for child_name, rest1 in self.edges.items():
            for child_index, rest2 in rest1.items():
                for parent_name, rest3 in rest2.items():
                    for parent_index, edge in rest3.items():
                        assert edge is not None, (child_name, child_index, parent_name, parent_index, "sldkfdj")

                        big_dict[(child_name, child_index, parent_name, parent_index)] = edge
        
        return big_dict

    def add_node(self, node: TLACDCInterpNode):
        assert node not in self.nodes(), f"Node {node} already in graph"
        self.graph[node.name][node.index] = node

    def add_edge(
        self,
        parent_node: TLACDCInterpNode,
        child_node: TLACDCInterpNode,
        edge: Edge,
    ):
        if parent_node not in self.nodes(): # TODO could be slow ???
            self.add_node(parent_node)
        if child_node not in self.nodes():
            self.add_node(child_node)
        
        parent_node._add_child(child_node)
        child_node._add_parent(parent_node)

        self.edges[child_node.name][child_node.index][parent_node.name][parent_node.index] = edge

    def setup_correspondence(self, model):
        # TODO would be to just straight up create the thing from the Transformer Lens model here!
        pass