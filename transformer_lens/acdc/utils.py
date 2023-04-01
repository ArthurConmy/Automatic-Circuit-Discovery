from functools import partial
from copy import deepcopy
import warnings
import collections
import random
from collections import defaultdict
from enum import Enum
from typing import Any, Literal, Dict, Tuple, Union, List, Optional, Callable, TypeVar, Generic, Iterable, Set, Type, cast, Sequence, Mapping, overload
import torch
from transformer_lens.HookedTransformer import HookedTransformer
from collections import OrderedDict

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
        self.graph: Dict[str, List[TLACDCInterpNode]] = OrderedDefaultdict(list) # need to put in `list`???

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


class TLACDCExperiment:
    """Manages an ACDC experiment, including the computational graph, the model, the data etc.
    Based off of ACDCExperiment from rust_circuit code"""

    def __init__(
        self,
        model: HookedTransformer,
        ds: torch.Tensor,
        ref_ds: Optional[torch.Tensor],
        template_corr: TLACDCCorrespondence,
        threshold: float,
        metric: Callable[[torch.Tensor, torch.Tensor], float], # dataset and logits to metric
        second_metric: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None,
        verbose: bool = False,
        parallel_hypotheses: int = 1,
        using_wandb: bool = False,
        remove_redundant: bool = False, # TODO implement
        monotone_metric: Literal[
            "off", "maximize", "minimize"
        ] = "minimize",  # if this is set to "maximize" or "minimize", then the metric will be maximized or minimized, respectively instead of us trying to keep the metric roughly the same. We do KL divergence by default
        first_cache_cpu: bool = True,
        second_cache_cpu: bool = True,
        zero_ablation: bool = False, # use zero rather than 
    ):
        self.model = model
        self.zero_ablation = zero_ablation
        self.verbose = verbose

        self.template_corr = template_corr
        self.topologically_sort_corr()
        self.corr = deepcopy(template_corr)

        self.ds = ds
        self.ref_ds = ref_ds
        self.first_cache_cpu = first_cache_cpu
        self.second_cache_cpu = second_cache_cpu

        self.setup_second_cache()
        
        self.using_wandb = using_wandb
        self.threshold = threshold
        assert self.ref_ds is not None or self.zero_ablation, "If you're doing random ablation, you need a ref ds"

        if self.using_wandb:
            self.metrics_to_plot = {}
            self.metrics_to_plot["new_metrics"] = []
            self.metrics_to_plot["list_of_parents_evaluated"] = []
            self.metrics_to_plot["list_of_children_evaluated"] = []
            self.metrics_to_plot["list_of_nodes_evaluated"] = []
            self.metrics_to_plot["evaluated_metrics"] = []
            self.metrics_to_plot["current_metrics"] = []
            self.metrics_to_plot["results"] = []
            self.metrics_to_plot["acdc_step"] = 0
            self.metrics_to_plot["num_edges"] = []

    def topologically_sort_corr(self):
        """Topologically sort the template corr"""
        for hook in self.model.hook_dict.values():
            assert len(hook.fwd_hooks) == 0, "Don't load the model with hooks *then* call this"

        new_graph = OrderedDict()
        cache=OrderedDict() # what if?
        self.model.cache_all(cache)
        self.model(torch.arange(5))

        print(self.template_corr.graph.keys())

        for hook_name in cache:
            print(hook_name)            
            if hook_name in self.template_corr.graph:
                new_graph[hook_name] = self.template_corr.graph[hook_name]

        self.template_corr.graph = new_graph

    def sender_hook(self, z, hook, verbose=False, cache="first", device=None):
        """General, to cover online and corrupt caching"""

        if device == "cpu": # maaaybe saves memory??
            tens = z.cpu()
        else:
            tens = z.clone()
            if device is not None:
                tens = tens.to(device)

        if cache == "second":
            hook.global_cache.second_cache[hook.name] = tens
        elif cache == "first":
            hook.global_cache.cache[hook.name] = tens
        else:
            raise ValueError(f"Unknown cache type {cache}")

        if verbose:
            print(f"Saved {hook.name} with norm {z.norm().item()}")

        return z

    def receiver_hook(self, z, hook, verbose=False):
        for receiver_node in self.corr.graph[hook.name]:
            direct_computation_nodes = []
            for sender_node in receiver_node.parents:

                # implement skipping edges post-hoc

                if verbose:
                    print(
                        hook.name, receiver_node.index, sender_node.name, sender_node.index.as_index
                    )
                    print("-------")
                    if sender_node.mode == "addition":
                        print(
                            hook.global_cache.cache[sender_node.name].shape,
                            sender_node.index.as_index,
                        )
                
                if sender_node.mode == "addition": # TODO turn into ENUM
                    z[receiver_node.index.as_index] -= hook.global_cache.cache[
                        sender_node.name
                    ][sender_node.index.as_index].to(z.device)
                    z[receiver_node.index.as_index] += hook.global_cache.second_cache[
                        sender_node.name
                    ][sender_node.index.as_index].to(z.device)

                elif sender_node.mode == "direct_computation":
                    direct_computation_nodes.append(sender_node)
                    assert len(direct_computation_nodes) == 1, f"Found multiple direct computation nodes {direct_computation_nodes}"

                    z[receiver_node.index.as_index] = hook.global_cache.second_cache[receiver_node.name][receiver_node.index.as_index].to(z.device)

                else: 
                    assert sender_node.mode == "off", f"Unknown edge type {sender_node.mode}"

        return z

    def add_sender_hooks(self, reset=True, cache="first"):
        if reset:
            self.model.reset_hooks()
        device = {
            "first": "cpu" if self.first_cache_cpu else None,
            "second": "cpu" if self.second_cache_cpu else None,
        }[cache]
        for node in self.corr.nodes():
            if node.mode != "addition":
                continue
            if len(self.model.hook_dict[node.name].fwd_hooks) > 0:
                continue
            self.model.add_hook(
                name=node.name, 
                hook=partial(self.sender_hook, verbose=self.verbose, cache=cache, device=device),
            )

        # ugh sort of ugly, more nodes need to be added to above thing
        for node in self.corr.nodes():
            if node.mode == "direct_computation":
                for child in node.children:
                    if len(self.model.hook_dict[child.name].fwd_hooks) > 0:
                        continue
                    print(child.name)
                    self.model.add_hook(
                        name=child.name,
                        hook=partial(self.sender_hook, verbose=self.verbose, cache="second", device=device),
                    )

    def setup_second_cache(self):
        self.add_sender_hooks(cache="second")
        corrupt_stuff = self.model(self.ref_ds)

        if self.zero_ablation:
            for name in self.model.global_cache.second_cache:
                self.model.global_cache.second_cache[name] = torch.zeros_like(
                    self.model.global_cache.second_cache[name]
                )
                torch.cuda.empty_cache()

        if self.second_cache_cpu:
            self.model.global_cache.to("cpu", which_caches="second")

        self.model.reset_hooks()

    def setup_model_hooks(self):
        self.add_sender_hooks(cache="first")

        receiver_node_names = list(set([node.name for node in self.corr.nodes()]))
        for receiver_name in receiver_node_names: # TODO could remove the nodes that don't have any parents...
            self.model.add_hook(
                name=receiver_name,
                hook=partial(self.receiver_hook, verbose=self.verbose),
            )

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