import numpy as np
import torch
from collections import defaultdict
from enum import Enum
from typing import Optional


class EdgeType(Enum):
    """TODO Arthur explain this more clearly and use GPT-4 for clarity/coherence. Ping Arthur if you want a better explanation and this isn't done!!!
    Property of edges in the computational graph - either 
    
    ADDITION: the child (hook_name, index) is a sum of the parent (hook_name, index)s
    DIRECT_COMPUTATION The *single* child is a function of and only of the parent (e.g the value hooked by hook_q is a function of what hook_q_input saves).
    PLACEHOLDER generally like 2. but where there are generally multiple parents. Here in ACDC we just include these edges by default when we find them. Explained below?
    
    Q: Why do we do this?

    A: We need something inside TransformerLens to represent the edges of a computational graph.
    The object we choose is pairs (hook_name, index). For example the output of Layer 11 Heads is a hook (blocks.11.attn.hook_result) and to sepcify the 3rd head we add the index [:, :, 3]. Then we can build a computational graph on these! 

    However, when we do ACDC there turn out to be two conflicting things "removing edges" wants to do: 
    i) for things in the residual stream, we want to remove the sum of the effects from previous hooks 
    ii) for things that are not linear we want to *recompute* e.g the result inside the hook 
    blocks.11.attn.hook_result from a corrupted Q and normal K and V

    The easiest way I thought of of reconciling these different cases, while also having a connected computational graph, is to have three types of edges: addition for the residual case, direct computation for easy cases where we can just replace hook_q with a cached value when we e.g cut it off from hook_q_input, and placeholder to make the graph connected (when hook_result is connected to hook_q and hook_k and hook_v)"""

    ADDITION = 0
    DIRECT_COMPUTATION = 1
    PLACEHOLDER = 2

    def __eq__(self, other):
        # TODO WTF? Why do I need this?? To busy to look into now, check the commit where we add this later
        return self.value == other.value

class Edge:
    def __init__(
        self,
        edge_type: EdgeType,
        present: bool = True,
        effect_size: Optional[float] = None,
        device: Optional[str] = None,
        sp: Optional[Literal["edge", "node"]] = False,
    ):
        self.edge_type = edge_type
        self.present = present
        self.effect_size = effect_size

        self.sp = sp
        if self.sp is not None:
            """Ripped from subnetwork_probing/transformer_lens/transformer_lens/hook_points.py"""
            # With edit as no requires_grad set...?
            self.mask_score = torch.nn.Parameter(torch.tensor([1.0], requires_grad=True, device=device).clone())
            self.beta = (
                2 / 3
            )  # TODO: make this hyperaparams globally set and synced somehow
            self.gamma = -0.1
            self.zeta = 1.1
            self.mask_p = 0.9
            self.init_weights()
            self.sampled = False

    def init_weights(self):
        """Ripped from subnetwork_probing/transformer_lens/transformer_lens/hook_points.py"""

        assert self.sp is not None
        p = (self.mask_p - self.gamma) / (self.zeta - self.gamma)
        torch.nn.init.constant_(self.mask_score, val=np.log(p / (1 - p)))

    def sample_mask(self) -> None:
        """Ripped from subnetwork_probing/transformer_lens/transformer_lens/hook_points.py"""

        assert self.sp is not None
        assert not self.sampled
        uniform_sample = (
            torch.zeros_like(self.mask_score).uniform_().clamp(0.0001, 0.9999)
        )
        s = torch.sigmoid(
            (uniform_sample.log() - (1 - uniform_sample).log() + self.mask_score)
            / self.beta
        )
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        mask = s_bar.clamp(min=0.0, max=1.0)
        self.mask = mask

    def __repr__(self) -> str:
        return f"Edge({self.edge_type}, {self.present})"

class TorchIndex:
    """There is not a clean bijection between things we 
    want in the computational graph, and things that are hooked
    (e.g hook_result covers all heads in a layer)
    
    `TorchIndex`s are essentially indices that say which part of the tensor is being affected. 
    
    EXAMPLES: Initialise [:, :, 3] with TorchIndex([None, None, 3]) and [:] with TorchIndex([None])    

    Also we want to be able to call e.g `my_dictionary[my_torch_index]` hence the hashable tuple stuff"""

    def __init__(
        self, 
        list_of_things_in_tuple,
    ):
        # check correct types
        for arg in list_of_things_in_tuple:
            if type(arg) in [type(None), int]:
                continue
            else:
                assert isinstance(arg, list)
                assert all([type(x) == int for x in arg])

        # make an object that can be indexed into a tensor
        self.as_index = tuple([slice(None) if x is None else x for x in list_of_things_in_tuple])

        # make an object that can be hashed (so used as a dictionary key)
        self.hashable_tuple = tuple(list_of_things_in_tuple)

    def __hash__(self):
        return hash(self.hashable_tuple)

    def __eq__(self, other):
        return self.hashable_tuple == other.hashable_tuple

    # some graphics things

    def __repr__(self, graphviz_index=False) -> str:
        ret = "["
        for idx, x in enumerate(self.hashable_tuple):
            if idx > 0:
                ret += ", "
            if x is None:
                ret += ":" if not graphviz_index else "COLON"
            elif type(x) == int:
                ret += str(x)
            else:
                raise NotImplementedError(x)
        ret += "]"
        return ret

    def graphviz_index(self) -> str:
        return self.__repr__(graphviz_index=True)
