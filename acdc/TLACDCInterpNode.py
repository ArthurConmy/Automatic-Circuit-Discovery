from acdc.TLACDCEdge import (
    TorchIndex,
    Edge, 
    EdgeType,
)  # these introduce several important classes !!!
from typing import List, Dict, Optional, Tuple, Union, Set, Callable, TypeVar, Iterable, Any

class TLACDCInterpNode:
    """Represents one node in the computational graph, similar to ACDCInterpNode from the rust_circuit code
    
    But WARNING this has nodes closer to the input tokens as *parents* of nodes closer to the output tokens, the opposite of the rust_circuit code
    
    Params:
        name: name of the node
        index: the index of the tensor that this node represents
        mode: how we deal with this node when we bump into it as a parent of another node. Addition: it's summed to make up the child. Direct_computation: it's the sole node used to compute the child. Off: it's not the parent of a child ever."""
        
    def __init__(self, name: str, index: TorchIndex, incoming_edge_type: EdgeType):
        
        self.name = name
        self.index = index
        
        self.parents: List["TLACDCInterpNode"] = []
        self.children: List["TLACDCInterpNode"] = []

        self.incoming_edge_type = incoming_edge_type

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

# ------------------ 
# some munging utils
# ------------------

def parse_interpnode(s: str) -> TLACDCInterpNode:
    try:
        name, idx = s.split("[")
        name = name.replace("hook_resid_mid", "hook_mlp_in")
        try:
            idx = int(idx[-3:-1])
        except:
            try: 
                idx = int(idx[-2])
            except:
                idx = None
        return TLACDCInterpNode(name, TorchIndex([None, None, idx]) if idx is not None else TorchIndex([None]), EdgeType.ADDITION)

    except Exception as e: 
        print(s, e)
        raise e

    return TLACDCInterpNode(name, TorchIndex([None, None, idx]), EdgeType.ADDITION)

def heads_to_nodes_to_mask(heads: List[Tuple[int, int]], return_dict=False):
    nodes_to_mask_strings = [
        f"blocks.{layer_idx}{'.attn' if not inputting else ''}.hook_{letter}{'_input' if inputting else ''}[COL, COL, {head_idx}]"
        # for layer_idx in range(model.cfg.n_layers)
        # for head_idx in range(model.cfg.n_heads)
        for layer_idx, head_idx in heads
        for letter in ["q", "k", "v"]
        for inputting in [True, False]
    ]
    nodes_to_mask_strings.extend([
        f"blocks.{layer_idx}.attn.hook_result[COL, COL, {head_idx}]"
        for layer_idx, head_idx in heads
    ])

    if return_dict:
        return {s: parse_interpnode(s) for s in nodes_to_mask_strings}

    else:
        return [parse_interpnode(s) for s in nodes_to_mask_strings]
