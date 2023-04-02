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
from transformer_lens.HookedTransformer import HookedTransformer
from transformer_lens.acdc.graphics import (
    log_metrics_to_wandb,
)
from collections import OrderedDict

def next_key(ordered_dict: OrderedDict, current_key):
    key_iterator = iter(ordered_dict)
    next((key for key in key_iterator if key == current_key), None)
    return next(key_iterator, None)

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

    def __eq__(self, other):
        # TODO WTF? Why do I need this?? To busy to look into now, check the commit where we add this later
        return self.value == other.value

class Edge:
    def __init__(
        self,
        edge_type: EdgeType,
        present: bool = True,
    ):
        self.edge_type = edge_type
        self.present = present

    def __repr__(self) -> str:
        return f"Edge({self.edge_type}, {self.present})"

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

    def __eq__(self, other):
        return self.hashable_tuple == other.hashable_tuple

    def __repr__(self) -> str:
        return f"TorchIndex({self.hashable_tuple})"

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

class TLACDCCorrespondence:
    """Stores the full computational graph, similar to ACDCCorrespondence from the rust_circuit code"""
        
    def __init__(self):
        self.graph: OrderedDict[str, OrderedDict[TorchIndex, TLACDCInterpNode]] = OrderedDefaultdict(OrderedDict) # TODO rename "nodes?"
 
        # TODO implement this
        self.edges: OrderedDict[str, OrderedDict[TorchIndex, OrderedDict[str, OrderedDict[TorchIndex, Optional[Edge]]]]] = make_nd_dict(end_type=None, n=4) # TODO need n=4 thing?

        # this maps (child_name, child_index) to parent_node
        # TODO maybe a further level of nesting: str, TorchIndex for the parent too ???

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


class TLACDCExperiment:
    """Manages an ACDC experiment, including the computational graph, the model, the data etc.
    Based off of ACDCExperiment from rust_circuit code"""

    def __init__(
        self,
        model: HookedTransformer,
        ds: torch.Tensor,
        ref_ds: Optional[torch.Tensor],
        corr: TLACDCCorrespondence,
        metric: Callable[[torch.Tensor, torch.Tensor], float], # dataset and logits to metric
        second_metric: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None,
        verbose: bool = False,
        hook_verbose: bool = False,
        parallel_hypotheses: int = 1, # lol
        remove_redundant: bool = False, # TODO implement
        monotone_metric: Literal[
            "off", "maximize", "minimize"
        ] = "minimize",  # if this is set to "maximize" or "minimize", then the metric will be maximized or minimized, respectively instead of us trying to keep the metric roughly the same. We do KL divergence by default
        first_cache_cpu: bool = True,
        second_cache_cpu: bool = True,
        zero_ablation: bool = False, # use zero rather than 
        config: Optional[Dict] = None,
        wandb_notes: str = "",
        skip_edges = "yes",
    ):
        self.model = model
        self.zero_ablation = zero_ablation
        self.verbose = verbose
        self.hook_verbose = hook_verbose
        self.skip_edges = skip_edges
        if skip_edges != "yes":
            raise NotImplementedError() # TODO

        self.corr = corr
        self.reverse_topologically_sort_corr()
        self.current_node = self.corr.nodes()[0]
        print(f"{self.current_node=}")

        self.ds = ds
        self.ref_ds = ref_ds
        self.first_cache_cpu = first_cache_cpu
        self.second_cache_cpu = second_cache_cpu
        self.setup_second_cache()
        if self.second_cache_cpu:
            self.model.global_cache.to("cpu", which_caches="second")
        self.setup_model_hooks()

        if config is not None and config["USING_WANDB"]:
            self.using_wandb = config["USING_WANDB"]

            wandb.init(
                entity=config["WANDB_ENTITY_NAME"],
                project=config["WANDB_PROJECT_NAME"], 
                name=config["WANDB_RUN_NAME"],
                notes=wandb_notes,
            )

        self.metric = metric
        self.second_metric = second_metric
        self.second_metric = second_metric
        self.update_cur_metric()

        self.threshold = config["THRESHOLD"]
        assert self.ref_ds is not None or self.zero_ablation, "If you're doing random ablation, you need a ref ds"

        self.parallel_hypotheses = parallel_hypotheses
        if self.parallel_hypotheses != 1:
            raise NotImplementedError("Parallel hypotheses not implemented yet") # TODO?

        if self.using_wandb:
            # TODO?
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

    def update_cur_metric(self, initial=False):
        self.cur_metric = self.metric(self.model(self.ds))

        if initial:
            assert abs(self.cur_metric) < 1e-5, f"Metric {self.cur_metric=} is not zero"

        if self.using_wandb:
            wandb.log({"cur_metric": self.cur_metric})

    def reverse_topologically_sort_corr(self):
        """Topologically sort the template corr"""
        for hook in self.model.hook_dict.values():
            assert len(hook.fwd_hooks) == 0, "Don't load the model with hooks *then* call this"

        new_graph = OrderedDict()
        cache=OrderedDict() # what if?
        self.model.cache_all(cache)
        self.model(torch.arange(5))

        if self.verbose:
            print(self.corr.graph.keys())

        cache_keys = list(cache.keys())
        cache_keys.reverse()

        for hook_name in cache_keys:
            print(hook_name)            
            if hook_name in self.corr.graph:
                new_graph[hook_name] = self.corr.graph[hook_name]

        self.corr.graph = new_graph

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
        receiver_node_name = hook.name
        for receiver_node_index in self.corr.edges[hook.name]:
            direct_computation_nodes = []
            for sender_node_name in self.corr.edges[hook.name][receiver_node_index]:
                for sender_node_index in self.corr.edges[hook.name][receiver_node_index][sender_node_name]:

                    edge = self.corr.edges[hook.name][receiver_node_index][sender_node_name][sender_node_index] # TODO maybe less crazy nested indexes ... just make local variables each time?

                    if edge.present:
                        continue # don't do patching stuff, if it wastes time

                    if verbose:
                        print(
                            hook.name, receiver_node_index, sender_node_name, sender_node_index,
                        )
                        print("-------")
                        if edge.edge_type == EdgeType.ADDITION:
                            print(
                                hook.global_cache.cache[sender_node_name].shape,
                                sender_node_index,
                            )
                    
                    if edge.edge_type == EdgeType.ADDITION:
                        z[receiver_node_index.as_index] -= hook.global_cache.cache[
                            sender_node_name
                        ][sender_node_index.as_index].to(z.device)
                        z[receiver_node_index.as_index] += hook.global_cache.second_cache[
                            sender_node_name
                        ][sender_node_index.as_index].to(z.device)

                    elif edge.edge_type == EdgeType.DIRECT_COMPUTATION:
                        direct_computation_nodes.append(self.corr.graph[sender_node_name][sender_node_index])
                        assert len(direct_computation_nodes) == 1, f"Found multiple direct computation nodes {direct_computation_nodes}"

                        z[receiver_node_index.as_index] = hook.global_cache.second_cache[receiver_node_name][receiver_node_index.as_index].to(z.device)

                    else: 
                        print(edge)
                        raise ValueError(f"Unknown edge type {edge.edge_type}")

        return z

    def add_sender_hooks(self, reset=True, cache="first"):    
        if self.verbose:
            print("Adding sender hooks...")
        if reset:
            self.model.reset_hooks()
        device = {
            "first": "cpu" if self.first_cache_cpu else None,
            "second": "cpu" if self.second_cache_cpu else None,
        }[cache]
        for big_tuple, edge in self.corr.all_edges().items():
            if edge.edge_type == EdgeType.DIRECT_COMPUTATION:
                node = self.corr.graph[big_tuple[0]][big_tuple[1]]
            elif edge.edge_type == EdgeType.ADDITION:
                node = self.corr.graph[big_tuple[2]][big_tuple[3]]
            else:
                print(edge.edge_type.value, EdgeType.ADDITION.value, edge.edge_type.value == EdgeType.ADDITION.value, type(edge.edge_type.value), type(EdgeType.ADDITION.value))
                raise ValueError(f"{str(big_tuple)} {str(edge)} failed")

            print(big_tuple, "worked!")

            if len(self.model.hook_dict[node.name].fwd_hooks) > 0:
                continue
            self.model.add_hook(
                name=node.name, 
                hook=partial(self.sender_hook, verbose=self.hook_verbose, cache=cache, device=device),
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
                hook=partial(self.receiver_hook, verbose=self.hook_verbose),
            )

    def step(self, early_stop=False):
        """
        TOIMPLEMENT: (prioritise these)
        - Incoming effect sizes on the nodes
        - Dynamic threshold?
        - Node stats
        - Parallelism
        - Not just monotone metric
        - remove_redundant
        """

        if self.current_node is None:
            return

        warnings.warn("Implement incoming effect sizes on the nodes")
        start_step_time = time.time()

        initial_metric = self.metric(self.model(self.ds)) # toks_int_values))
        assert isinstance(initial_metric, float), f"Initial metric is a {type(initial_metric)} not a float"

        cur_metric = initial_metric
        if self.verbose:
            print("New metric:", cur_metric)

        for sender_name in self.corr.edges[self.current_node.name][self.current_node.index]:
            for sender_index in self.corr.edges[self.current_node.name][self.current_node.index][sender_name]:
                edge = self.corr.edges[self.current_node.name][self.current_node.index][sender_name][sender_index]
                cur_parent = self.corr.graph[sender_name][sender_index]

                if self.verbose:
                    print(f"\nNode: {cur_parent=} ({self.current_node=})\n")

                edge.present = False

                if early_stop: # for debugging the effects of one and only one forward pass WITH a corrupted edge
                    return self.model(self.ds)

                evaluated_metric = self.metric(self.model(self.ds))

                if self.verbose:
                    print(
                        "Metric after removing connection to",
                        sender_name,
                        sender_index,
                        "is",
                        evaluated_metric,
                        "(and current metric " + str(cur_metric) + ")",
                    )

                result = evaluated_metric - cur_metric

                if self.verbose:
                    print("Result is", result, end="")

                if result < self.threshold:
                    print("...so removing connection")
                    cur_metric = evaluated_metric

                else:
                    if self.verbose:
                        print("...so keeping connection")
                    edge.present = True

                if self.using_wandb:
                    log_metrics_to_wandb(
                        self,
                        current_metric = cur_metric,
                        parent_name = str(self.corr.graph[sender_name][sender_index]),
                        child_name = str(self.current_node),
                        result = result,
                    )
                    wandb.log({"num_edges": self.count_no_edges()})

            cur_metric = self.metric(self.model(self.ds))

        # increment the current node
        self.increment_current_node()

    def current_node_connected(self):
        not_presents_exist = False

        for child_name, rest1 in self.corr.edges.items():
            for child_index, rest2 in rest1.items():
                if self.current_node.name in rest2 and self.current_node.index in rest2[self.current_node.name]:
                    if rest2[self.current_node.name][self.current_node.index].present:
                        return True
                    else:
                        not_presents_exist = True

                # if child_name == self.current_node.name and child_index == self.current_node.index and len(rest2) == 0: # [self.current_node.name][self.current_node.index]:
                #     return True
        
        return not not_presents_exist

    def find_next_node(self) -> Optional[TLACDCInterpNode]:
        next_index = next_key(self.corr.graph[self.current_node.name], self.current_node.index)
        if next_index is not None:
            return self.corr.graph[self.current_node.name][next_index]

        next_name = next_key(self.corr.graph, self.current_node.name)

        if next_name is not None:
            return list(self.corr.graph[next_name].values())[0]
            
        warnings.warn("Finished iterating")
        return None

    def increment_current_node(self) -> None:
        self.current_node = self.find_next_node()
        print("We moved to ", self.current_node)

        if self.current_node is not None and not self.current_node_connected():
            print("But it's bad")
            self.increment_current_node()

    def count_no_edges(self):

        # TODO if this is slow (check) find a faster way

        cnt = 0

        for edge in self.corr.all_edges().values():
            if edge.present:
                cnt += 1

        return cnt

def make_nd_dict(end_type, n = 3) -> Any:
    """Make biiig default dicts : ) : )"""

    if n not in [3, 4]:
        raise NotImplementedError("Only implemented for 3/4")
        
    if n == 3:
        return OrderedDefaultdict(lambda: defaultdict(lambda: defaultdict(end_type)))

    if n == 4:
        return OrderedDefaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(end_type))))

def ct():
    return time.ctime().replace(" ", "_").replace(":", "_")