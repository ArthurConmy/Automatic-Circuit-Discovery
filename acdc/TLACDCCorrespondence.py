from acdc.TLACDCInterpNode import TLACDCInterpNode
from collections import OrderedDict
from acdc.acdc_utils import TorchIndex, Edge, EdgeType, OrderedDefaultdict, make_nd_dict
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

    def add_node(self, node: TLACDCInterpNode, safe=True):
        if safe:
            assert node not in self.nodes(), f"Node {node} already in graph"
        self.graph[node.name][node.index] = node

    def add_edge(
        self,
        parent_node: TLACDCInterpNode,
        child_node: TLACDCInterpNode,
        edge: Edge,
        safe=True,
    ):
        if safe:
            if parent_node not in self.nodes(): # TODO could be slow ???
                self.add_node(parent_node)
            if child_node not in self.nodes():
                self.add_node(child_node)
        
        assert child_node.incoming_edge_type == edge.edge_type, (child_node.incoming_edge_type, edge.edge_type)
        
        parent_node._add_child(child_node)
        child_node._add_parent(parent_node)

        self.edges[child_node.name][child_node.index][parent_node.name][parent_node.index] = edge
    
    @classmethod
    def setup_from_model(cls, model):
        correspondence = cls()

        downstream_residual_nodes: List[TLACDCInterpNode] = []
        logits_node = TLACDCInterpNode(
            name=f"blocks.{model.cfg.n_layers-1}.hook_resid_post",
            index=TorchIndex([None]),
            incoming_edge_type = EdgeType.ADDITION,
        )
        correspondence.add_node(logits_node) 
        downstream_residual_nodes.append(logits_node)
        new_downstream_residual_nodes: List[TLACDCInterpNode] = []

        for layer_idx in range(model.cfg.n_layers - 1, -1, -1):
            # connect MLPs
            if not model.cfg.attn_only: 
                # this MLP writed to all future residual stream things
                cur_mlp_name = f"blocks.{layer_idx}.hook_mlp_out"
                cur_mlp_slice = TorchIndex([None])
                cur_mlp = TLACDCInterpNode(
                    name=cur_mlp_name,
                    index=cur_mlp_slice,
                    incoming_edge_type=EdgeType.DIRECT_COMPUTATION,
                )
                correspondence.add_node(cur_mlp)
                for residual_stream_node in downstream_residual_nodes:
                    correspondence.add_edge(
                        parent_node=cur_mlp,
                        child_node=residual_stream_node,
                        edge=Edge(edge_type=EdgeType.ADDITION),
                        safe=False,
                    )

                cur_mlp_input_name = f"blocks.{layer_idx}.hook_resid_mid"
                cur_mlp_input_slice = TorchIndex([None])
                cur_mlp_input = TLACDCInterpNode(
                    name=cur_mlp_input_name,
                    index=cur_mlp_input_slice,
                    incoming_edge_type=EdgeType.ADDITION,
                )
                correspondence.add_node(cur_mlp_input)
                correspondence.add_edge(
                    parent_node=cur_mlp_input,
                    child_node=cur_mlp,
                    edge=Edge(edge_type=EdgeType.DIRECT_COMPUTATION),
                    safe=False,
                )

                downstream_residual_nodes.append(cur_mlp_input)

            # connect attention heads
            for head_idx in range(model.cfg.n_heads - 1, -1, -1):
                # this head writes to all future residual stream things
                cur_head_name = f"blocks.{layer_idx}.attn.hook_result"
                cur_head_slice = TorchIndex([None, None, head_idx])
                cur_head = TLACDCInterpNode(
                    name=cur_head_name,
                    index=cur_head_slice,
                    incoming_edge_type=EdgeType.PLACEHOLDER,
                )
                correspondence.add_node(cur_head)
                for residual_stream_node in downstream_residual_nodes:
                    correspondence.add_edge(
                        parent_node=cur_head,
                        child_node=residual_stream_node,
                        edge=Edge(edge_type=EdgeType.ADDITION),
                        safe=False,
                    )

                # TODO maybe this needs be moved out of this block??? IDK
                for letter in "qkv":
                    hook_letter_name = f"blocks.{layer_idx}.attn.hook_{letter}"
                    hook_letter_slice = TorchIndex([None, None, head_idx])
                    hook_letter_node = TLACDCInterpNode(name=hook_letter_name, index=hook_letter_slice, incoming_edge_type=EdgeType.DIRECT_COMPUTATION)
                    correspondence.add_node(hook_letter_node)

                    hook_letter_input_name = f"blocks.{layer_idx}.hook_{letter}_input"
                    hook_letter_input_slice = TorchIndex([None, None, head_idx])
                    hook_letter_input_node = TLACDCInterpNode(
                        name=hook_letter_input_name, index=hook_letter_input_slice, incoming_edge_type=EdgeType.ADDITION
                    )
                    correspondence.add_node(hook_letter_input_node)

                    correspondence.add_edge(
                        parent_node = hook_letter_node,
                        child_node = cur_head,
                        edge = Edge(edge_type=EdgeType.PLACEHOLDER),
                        safe = False,
                    )

                    correspondence.add_edge(
                        parent_node=hook_letter_input_node,
                        child_node=hook_letter_node,
                        edge=Edge(edge_type=EdgeType.DIRECT_COMPUTATION),
                        safe=False,
                    )

                    new_downstream_residual_nodes.append(hook_letter_input_node)
            downstream_residual_nodes.extend(new_downstream_residual_nodes)

        # add the embedding node

        embedding_node = TLACDCInterpNode(
            name="blocks.0.hook_resid_pre",
            index=TorchIndex([None]),
            incoming_edge_type=EdgeType.PLACEHOLDER, # TODO maybe add some NoneType or something???
        )
        correspondence.add_node(embedding_node)
        for node in downstream_residual_nodes:
            correspondence.add_edge(
                parent_node=embedding_node,
                child_node=node,
                edge=Edge(edge_type=EdgeType.ADDITION),
                safe=False,
            )
    
        return correspondence

    def count_no_edges(self) -> int:
        cnt = 0
        for edge in self.all_edges().values():
            if edge.present and edge.edge_type != EdgeType.PLACEHOLDER:
                cnt += 1
        return cnt
