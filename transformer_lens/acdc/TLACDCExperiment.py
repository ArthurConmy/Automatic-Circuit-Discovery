from typing import Callable, Optional, Literal, List, Dict, Any, Tuple, Union, Set, Iterable, TypeVar, Type
import random
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from transformer_lens.acdc.TLACDCInterpNode import TLACDCInterpNode
from transformer_lens.acdc.TLACDCCorrespondence import TLACDCCorrespondence
from transformer_lens.HookedTransformer import HookedTransformer
from transformer_lens.acdc.graphics import log_metrics_to_wandb
import warnings
import wandb
from transformer_lens.acdc.utils import TorchIndex, Edge, EdgeType
from collections import OrderedDict
from functools import partial
import time

def next_key(ordered_dict: OrderedDict, current_key):
    key_iterator = iter(ordered_dict)
    next((key for key in key_iterator if key == current_key), None)
    return next(key_iterator, None)

class TLACDCExperiment:
    """Manages an ACDC experiment, including the computational graph, the model, the data etc.
    Based off of ACDCExperiment from rust_circuit code"""

    def __init__(
        self,
        model: HookedTransformer,
        ds: torch.Tensor,
        ref_ds: Optional[torch.Tensor],
        threshold: float,
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
        using_wandb: bool = False,
        wandb_entity_name: str = "",
        wandb_project_name: str = "",
        wandb_run_name: str = "",
        wandb_notes: str = "",
        skip_edges = "yes",
        indices_mode: Literal["normal", "reverse", "shuffle"] = "normal", # last minute we found that this helps quite a lot with zero ablation case
        names_mode: Literal["normal", "reverse", "shuffle"] = "normal",
    ):
        """Initialize the ACDC experiment"""

        self.indices_mode = indices_mode
        self.names_mode = names_mode

        self.model = model
        self.zero_ablation = zero_ablation
        self.verbose = verbose
        self.hook_verbose = hook_verbose
        self.skip_edges = skip_edges
        if skip_edges != "yes":
            raise NotImplementedError() # TODO if edge counts are slow...

        self.corr = TLACDCCorrespondence.setup_from_model(self.model)
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

        self.using_wandb = using_wandb
        if using_wandb:
            wandb.init(
                entity=wandb_entity_name,
                project=wandb_project_name,
                name=wandb_run_name,
                notes=wandb_notes,
            )

        self.metric = metric
        self.second_metric = second_metric
        self.second_metric = second_metric
        self.update_cur_metric()

        self.threshold = threshold
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
            names = list(self.model.global_cache.second_cache.keys())
            assert len(names)>0, "No second cache names found"
            print("WE NAAMING")
            for name in names:
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
        if self.current_node is None:
            return

        warnings.warn("Implement incoming effect sizes on the nodes")
        start_step_time = time.time()

        initial_metric = self.metric(self.model(self.ds)) # toks_int_values))
        assert isinstance(initial_metric, float), f"Initial metric is a {type(initial_metric)} not a float"

        cur_metric = initial_metric
        if self.verbose:
            print("New metric:", cur_metric)

        sender_names_list = list(self.corr.edges[self.current_node.name][self.current_node.index])
        
        if self.names_mode == "random":
            random.shuffle(sender_names_list)
        elif self.names_mode == "reverse":
            sender_names_list = list(reversed(sender_names_list))

        for sender_name in sender_names_list:
            sender_indices_list = list(self.corr.edges[self.current_node.name][self.current_node.index][sender_name])

            if self.indices_mode == "random":
                random.shuffle(sender_indices_list)
            elif self.indices_mode == "reverse":
                sender_indices_list = list(reversed(sender_indices_list))

            for sender_index in sender_indices_list:
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
                edge.effect_size = result

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

            for parent_node_name in self.corr.edges[self.current_node.name][self.current_node.index]:
                for parent_node_index in self.corr.edges[self.current_node.name][self.current_node.index][parent_node_name]:
                    self.corr.edges[self.current_node.name][self.current_node.index][parent_node_name][parent_node_index].present = False

            self.increment_current_node()

    def count_no_edges(self):

        # TODO if this is slow (check) find a faster way

        cnt = 0

        for edge in self.corr.all_edges().values():
            if edge.present:
                cnt += 1

        return cnt