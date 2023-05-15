import wandb
from functools import partial
from copy import deepcopy
import warnings
import collections
import random
from collections import defaultdict
from enum import Enum
from typing import Any, Literal, Dict, Tuple, Union, List, Optional, Callable, TypeVar, Generic, Iterable, Set, Type, cast, Sequence, Mapping, overload
import torch
import time
import torch.nn.functional as F
from acdc.HookedTransformer import HookedTransformer
from collections import OrderedDict

TorchIndexHashableTuple = Tuple[Union[None, slice], ...]

def cleanup():
    import gc
    gc.collect()
    torch.cuda.empty_cache()

def shuffle_tensor(tens, seed=42):
    """Shuffle tensor along first dimension"""
    torch.random.manual_seed(seed)
    return tens[torch.randperm(tens.shape[0])]

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
    ):
        self.edge_type = edge_type
        self.present = present
        self.effect_size = effect_size

    def __repr__(self) -> str:
        return f"Edge({self.edge_type}, {self.present})"

# TODO attrs.frozen???
class TorchIndex:
    """There is not a clean bijection between things we 
    want in the computational graph, and things that are hooked
    (e.g hook_result covers all heads in a layer)
    
    `HookReference`s are essentially indices that say which part of the tensor is being affected. 
    
    E.g (slice(None), slice(None), 3) means index [:, :, 3]
    
    Also we want to be able to go my_dictionary[my_torch_index] hence the hashable tuple stuff
    
    EXAMPLES: Initialise [:, :, 3] with TorchIndex([None, None, 3]) and [:] with TorchIndex([None])"""

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

    def __eq__(self, other):
        return self.hashable_tuple == other.hashable_tuple

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

    # @classmethod
    # def from_index(cls, hashable_tuples: tuple) -> "TorchIndex":
    #     assert isinstance(index, tuple), type(index)
    #     assert all([i==slice(None) or isinstance(i, int) for i in index]), f"{index=} does not have support: in future ACDC may have spicier indexing"
    #     return cls([None if i==slice(None) else i for i in index])

def make_nd_dict(end_type, n = 3) -> Any:
    """Make biiig default dicts : ) : )"""

    if n not in [3, 4]:
        raise NotImplementedError("Only implemented for 3/4")
        
    if n == 3:
        return OrderedDefaultdict(lambda: defaultdict(lambda: defaultdict(end_type)))

    if n == 4:
        return OrderedDefaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(end_type))))

def ct():
    return time.ctime().replace(" ", "_").replace(":", "_").replace("__", "_")

def kl_divergence(
    logits: torch.Tensor,
    base_model_logprobs: torch.Tensor,
    mask_repeat_candidates: Optional[torch.Tensor] = None,
    last_seq_element_only: bool = True,
    base_model_probs_last_seq_element_only: bool = False,
    return_one_element: bool = True,
    end_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Note: we want base_model_probs_last_seq_element_only to remain False by default, because when the Docstring
    # circuit uses this, it already takes the last position before passing it in.

    if last_seq_element_only and end_positions is not None:
        raise ValueError("Can't have both of these true...")

    if last_seq_element_only:
        logits = logits[:, -1, :]
    if end_positions is not None:
        logits = logits[torch.arange(len(logits)), end_positions]

    if base_model_probs_last_seq_element_only:
        base_model_logprobs = base_model_logprobs[:, -1, :]

    logprobs = F.log_softmax(logits, dim=-1)
    kl_div = F.kl_div(logprobs, base_model_logprobs, log_target=True, reduction="none").sum(dim=-1)

    if mask_repeat_candidates is not None:
        assert kl_div.shape == mask_repeat_candidates.shape, (kl_div.shape, mask_repeat_candidates.shape)
        denom = mask_repeat_candidates.long().sum()
        kl_div = kl_div * mask_repeat_candidates
        if return_one_element:
            answer = kl_div.sum() / denom
        else:
            answer = kl_div
    else:
        if return_one_element:
            answer = kl_div.mean()
        else:
            answer = kl_div

    return answer


def negative_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask_repeat_candidates: Optional[torch.Tensor] = None,
    baseline: Union[float, torch.Tensor] = 0.0,
    last_seq_element_only: bool = True,
    return_one_element: bool=True,
) -> torch.Tensor:
    logprobs = F.log_softmax(logits, dim=-1)

    if last_seq_element_only:
        logprobs = logprobs[:, -1, :]

    # Subtract a baseline for each element -- which could be 0 or the NLL of the base_model_logprobs
    nll_all = (
        F.nll_loss(logprobs.view(-1, logprobs.size(-1)), labels.view(-1), reduction="none").view(logprobs.size()[:-1])
        - baseline
    )

    if mask_repeat_candidates is not None:
        assert nll_all.shape == mask_repeat_candidates.shape, (
            nll_all.shape,
            mask_repeat_candidates.shape,
        )
        denom = mask_repeat_candidates.long().sum()
        nll_all = nll_all * mask_repeat_candidates
        answer = nll_all
        if return_one_element:
            answer = nll_all.sum() / denom

    else:
        if return_one_element:
            answer = nll_all.mean()

    return answer


class MatchNLLMetric:
    def __init__(
        self,
        labels: torch.Tensor,
        base_model_logprobs: torch.Tensor,
        mask_repeat_candidates: Optional[torch.Tensor] = None,
        last_seq_element_only: bool = True,
        return_one_element: bool = True,
    ):
        self.labels = labels
        self.mask_repeat_candidates = mask_repeat_candidates
        self.last_seq_element_only = last_seq_element_only

        logprobs = base_model_logprobs
        if last_seq_element_only:
            assert logprobs.ndim == 2
        else:
            assert logprobs.ndim == 3

        self.base_nll_unreduced = F.nll_loss(
            logprobs.view(-1, logprobs.size(-1)), labels.view(-1), reduction="none"
        ).view(logprobs.size()[:-1])
        if mask_repeat_candidates is not None:
            assert self.base_nll_unreduced.shape == mask_repeat_candidates.shape

        self.return_one_element = return_one_element

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        return negative_log_probs(
            logits,
            self.labels,
            mask_repeat_candidates=self.mask_repeat_candidates,
            baseline=self.base_nll_unreduced,
            last_seq_element_only=self.last_seq_element_only,
            return_one_element=self.return_one_element,
        )

# ----------------------------------
# Random helpers for scraping
# ----------------------------------

import re
import ast

def extract_info(string):
    """Thanks GPT-4 for writing all this..."""

    # Regex patterns
    parent_pattern = r"cur_parent=TLACDCInterpNode\((.*?), \[(.*?)\]\)"
    current_pattern = r"self.current_node=TLACDCInterpNode\((.*?), \[(.*?)\]\)"

    # Extract parent info
    parent_match = re.search(parent_pattern, string)
    parent_name = parent_match.group(1) if parent_match else None
    parent_list_str = parent_match.group(2) if parent_match else None
    parent_list = None
    if parent_list_str:
        parent_list_items = parent_list_str.split(", ")
        parent_list = [ast.literal_eval(item if item != "COL" else "None") for item in parent_list_items]

    # Extract current node info
    current_match = re.search(current_pattern, string)
    current_name = current_match.group(1) if current_match else None
    current_list_str = current_match.group(2) if current_match else None
    current_list = None
    if current_list_str:
        current_list_items = current_list_str.split(", ")
        current_list = [ast.literal_eval(item if item != "COL" else "None") for item in current_list_items]

    return parent_name, parent_list, current_name, current_list

if __name__ == "__main__":
    string = "Node: cur_parent=TLACDCInterpNode(blocks.3.attn.hook_result, ['COL', 'COL', 1]) (self.current_node=TLACDCInterpNode(blocks.3.hook_resid_post, ['COL']))"
    parent_name, parent_list, current_name, current_list = extract_info(string)

    print(f"Parent Name: {parent_name}\nParent List: {parent_list}\nCurrent Name: {current_name}\nCurrent List: {current_list}")

# ----------------------------------
# Precision and recall etc metrics
# ----------------------------------

def get_stat(ground_truth, recovered, mode, verbose=False):
    assert mode in ["true positive", "false positive", "false negative"]
    assert set(ground_truth.all_edges().keys()) == set(recovered.all_edges().keys()), "There is a mismatch between the keys we're comparing here"

    ground_truth_all_edges = ground_truth.all_edges()
    recovered_all_edges = recovered.all_edges()

    cnt = 0
    for tupl, edge in ground_truth_all_edges.items():
        if edge.edge_type == EdgeType.PLACEHOLDER:
            continue
        if mode == "false positive": 
            if recovered_all_edges[tupl].present and not edge.present:
                cnt += 1
                if verbose:
                    print(tupl)
        elif mode == "negative":
            if not recovered_all_edges[tupl].present and edge.present:
                cnt += 1
        elif mode == "true positive":
            if recovered_all_edges[tupl].present and edge.present:
                cnt += 1

    return cnt

def false_positive_rate(ground_truth, recovered, verbose=False):
    return get_stat(ground_truth, recovered, mode="false positive", verbose=verbose)

def false_negative_rate(ground_truth, recovered, verbose=False):
    return get_stat(ground_truth, recovered, mode="false negative", verbose=verbose)

def true_positive_stat(ground_truth, recovered, verbose=False):
    return get_stat(ground_truth, recovered, mode="true positive", verbose=verbose)
