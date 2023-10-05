from argparse import Namespace
import os
import pickle
import gc
import tempfile
from typing import Callable, Optional, Literal, List, Dict, Any, Tuple, Union, Set, Iterable, TypeVar, Type
import random
from dataclasses import dataclass
import torch
from acdc.acdc_graphics import show
from torch import nn
from torch.nn import functional as F
from acdc.TLACDCInterpNode import TLACDCInterpNode
from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from transformer_lens.HookedTransformer import HookedTransformer
from acdc.global_cache import GlobalCache
from acdc.acdc_graphics import log_metrics_to_wandb
import warnings
import wandb
from acdc.acdc_utils import extract_info, shuffle_tensor
from acdc.TLACDCEdge import (
    TorchIndex,
    Edge, 
    EdgeType,
)  # these introduce several important classes !!!
from collections import OrderedDict
from functools import partial
import time
from acdc.acdc_utils import next_key

# some types that will help
TorchIndexHashableTuple = Tuple[Union[None, slice], ...]
Subgraph = Dict[Tuple[str, TorchIndexHashableTuple, str, TorchIndexHashableTuple], bool] # an alias for loading and saving from WANDB (primarily)

T = TypeVar("T")

class TLACDCExperiment:
    """Manages an ACDC experiment, including the computational graph, the model, the data etc.

    Note: we always minimize a metric! Pass the negative version of your metric if you wish to maximize a metric.

    The *important method* is the .step() method which processes one node (and all the connections into that node)
    It's also helpful to understand what's going on in the def sender_hook(...) and def receiver_hook(...) methods - these are attached to the model to do path patching
    (see https://github.com/redwoodresearch/Easy-Transformer for a gentler introduction to path patching)

    You could also read __init__ for what's going on in this class. 

    A lot of the other methods are not important apart from specific evals and things.

    Based off of ACDCExperiment from old rust_circuit code"""

    def __init__(
        self,
        model: HookedTransformer,
        ds: torch.Tensor,
        ref_ds: Optional[torch.Tensor],
        threshold: float,
        metric: Callable[[torch.Tensor], torch.Tensor],
        second_metric: Optional[Callable[[torch.Tensor], float]] = None,
        verbose: bool = False,
        hook_verbose: bool = False,
        parallel_hypotheses: int = 1, # lol
        remove_redundant: bool = False, 
        online_cache_cpu: bool = True,
        corrupted_cache_cpu: bool = True,
        zero_ablation: bool = False, # use zero rather than
        abs_value_threshold: bool = False,
        show_full_index = False,
        using_wandb: bool = False,
        wandb_entity_name: str = "",
        wandb_project_name: str = "",
        wandb_run_name: str = "",
        wandb_group_name: str = "",
        wandb_notes: str = "",
        wandb_dir: Optional[str]=None,
        wandb_mode: str="online",
        use_pos_embed: bool = False,
        skip_edges = "no",
        add_sender_hooks: bool = True,
        add_receiver_hooks: bool = False,
        indices_mode: Literal["normal", "reverse", "shuffle"] = "reverse", # we get best performance with reverse I think
        names_mode: Literal["normal", "reverse", "shuffle"] = "normal",
        wandb_config: Optional[Namespace] = None,
        early_exit: bool = False,
        positions: Optional[List[int]] = None, # if None, do not split by position. TODO change the syntax here...
    ):
        """Initialize the ACDC experiment"""

        if zero_ablation and remove_redundant:
            raise ValueError("It's not possible to do zero ablation and remove redundant paths.\
            remove_redundant removes `dead` paths that don't go all the way back to the input.\
            When we do corrupted ablation, removing the dead paths has no effect on forward passes.\
            However, a dead edge in the zero ablation cases outputs a non-zero output! (That is usually computed from zero inputs)")

        model.reset_hooks()

        self.remove_redundant = remove_redundant
        self.indices_mode = indices_mode
        self.names_mode = names_mode
        self.use_pos_embed = use_pos_embed
        self.show_full_index = show_full_index
        
        self.positions = positions if positions is not None else [None]
        if self.positions != [None]:
            raise NotImplementedError("Splitting by position not implemented yet")

        self.model = model
        self.verify_model_setup()
        self.zero_ablation = zero_ablation
        self.abs_value_threshold = abs_value_threshold
        self.verbose = verbose
        self.step_idx = 0
        self.hook_verbose = hook_verbose
        self.skip_edges = skip_edges

        if skip_edges != "no":
            warnings.warn("Never skipping edges, for now")
            skip_edges = "no"

        self.corr = TLACDCCorrespondence.setup_from_model(self.model, use_pos_embed=use_pos_embed)

        if early_exit: 
            return
            
        self.reverse_topologically_sort_corr()
        self.current_node = self.corr.first_node()
        print(f"{self.current_node=}")
        self.corr = self.corr

        self.ds = ds
        self.ref_ds = ref_ds
        self.online_cache_cpu = online_cache_cpu
        self.corrupted_cache_cpu = corrupted_cache_cpu

        if zero_ablation:
            if self.ref_ds is None:
                self.ref_ds = self.ds.clone()
            else:
                warnings.warn("We shall overwrite the ref_ds with zeros.")
        self.global_cache = GlobalCache(
            device=("cpu" if self.online_cache_cpu else "cuda", "cpu" if self.corrupted_cache_cpu else "cuda"),
        )

        self.setup_corrupted_cache()
        if self.corrupted_cache_cpu:
            self.global_cache.to("cpu", which_caches="corrupted")

        self.setup_model_hooks(
            add_sender_hooks=add_sender_hooks,
            add_receiver_hooks=add_receiver_hooks,
        )

        self.using_wandb = using_wandb
        if using_wandb:
            wandb.init(
                entity=wandb_entity_name,
                group=wandb_group_name,
                project=wandb_project_name,
                name=wandb_run_name,
                notes=wandb_notes,
                dir=wandb_dir,
                mode=wandb_mode,
                config=wandb_config,
            )

        self.metric = lambda x: metric(x).item()
        self.second_metric = second_metric
        self.update_cur_metric(recalc_metric=True, recalc_edges=True)

        self.threshold = threshold
        assert self.ref_ds is not None or self.zero_ablation, "If you're doing random ablation, you need a ref ds"

        self.parallel_hypotheses = parallel_hypotheses
        if self.parallel_hypotheses != 1:
            raise NotImplementedError("Parallel hypotheses not implemented yet")

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
            self.metrics_to_plot["times"] = []
            self.metrics_to_plot["times_diff"] = []

    def verify_model_setup(self):
        if not self.model.cfg.attn_only and "use_hook_mlp_in" in self.model.cfg.to_dict():
            assert self.model.cfg.use_hook_mlp_in, "Need to be able to see hook MLP inputs"
        assert self.model.cfg.use_attn_result, "Need to be able to see split by head outputs"
        assert self.model.cfg.use_split_qkv_input, "Need to be able to see split by head QKV inputs"

    def update_cur_metric(self, recalc_metric=True, recalc_edges=True, initial=False):
        if recalc_metric:
            logits = self.model(self.ds)
            self.cur_metric = self.metric(logits)
            if self.second_metric is not None:
                self.cur_second_metric = self.second_metric(logits)

        if recalc_edges:
            self.cur_edges = self.count_no_edges()

        if initial:
            assert abs(self.cur_metric) < 1e-5, f"Metric {self.cur_metric=} is not zero"

        if self.using_wandb:
            wandb_return_dict = {
                "cur_metric": self.cur_metric,
                "num_edges": self.cur_edges,
            }
            if self.second_metric is not None:
                wandb_return_dict["second_cur_metric"] = self.cur_second_metric
            wandb.log(wandb_return_dict)

    def reverse_topologically_sort_corr(self):
        """Topologically sort the template corr"""
        for hook in self.model.hook_dict.values():
            assert len(hook.fwd_hooks) == 0, "Don't load the model with hooks *then* call this"

        new_graph = OrderedDict()
        cache=OrderedDict()
        self.model.cache_all(cache)
        self.model(torch.arange(min(10, self.model.cfg.d_vocab)).unsqueeze(0)) # Some random forward pass so that we can see all the hook names
        self.model.reset_hooks()

        if self.verbose:
            print(self.corr.graph.keys())

        cache_keys = list(cache.keys())
        cache_keys.reverse()

        for hook_name in cache_keys:
            print(hook_name)            
            if hook_name in self.corr.graph:
                new_graph[hook_name] = self.corr.graph[hook_name]

        self.corr.graph = new_graph

    def sender_hook(self, z, hook, verbose=False, cache="online", device=None):
        """Hook that saves activations of a HookPoint to a cache
        Supports cache="corrupted" to save corrupted activations
        
        And cache="online" to save activations 'online' throughout a forward pass"""

        if device == "cpu":
            tens = z.cpu()
        else:
            tens = z
            if device is not None:
                tens = tens.to(device)

        if cache == "corrupted":
            self.global_cache.corrupted_cache[hook.name] = tens
        elif cache == "online":
            self.global_cache.online_cache[hook.name] = tens
        else:
            raise ValueError(f"Unknown cache type {cache}")

        if verbose:
            print(f"Saved {hook.name} with norm {z.norm().item()}")

        return z

    def receiver_hook(self, hook_point_input, hook, verbose=False):
        """Hook that manages nodes *receiving* input from other nodes

        Note that we add **one receiver_hook per HookPoint**
        So all the computation for a given HookPoint, for example all computations involving query inputs to layer 1 is managed by one receiver_hook 
        (because all layer 1 query inputs are contained in the `blocks.1.hook_q_input` hook)"""
        
        incoming_edge_types = [self.corr.graph[hook.name][receiver_index].incoming_edge_type for receiver_index in list(self.corr.edges[hook.name].keys())]

        if verbose:
            print("In receiver hook", hook.name)

        if EdgeType.DIRECT_COMPUTATION in incoming_edge_types:

            old_z = hook_point_input.clone()
            hook_point_input[:] = self.global_cache.corrupted_cache[hook.name].to(hook_point_input.device) # It is crucial to use [:] to not use same tensor

            if verbose:
                print("Overwrote to sec cache")

            assert incoming_edge_types == [EdgeType.DIRECT_COMPUTATION for _ in incoming_edge_types], f"All incoming edges should be the same type not {incoming_edge_types}"

            for receiver_index in self.corr.edges[hook.name]:
                list_of_senders = list(self.corr.edges[hook.name][receiver_index].keys())
                assert len(list_of_senders) <= 1, "This is a direct computation, so there should only be one sender node" # TODO maybe implement expect-to-be-1 ???
                if len(list_of_senders) == 0:
                    continue
                sender_node = list_of_senders[0]

                sender_indices = list(self.corr.edges[hook.name][receiver_index][sender_node].keys())
                assert len(sender_indices) <= 1, "This is a direct computation, so there should only be one sender index"
                if len(sender_indices) == 0:
                    continue
                sender_index = sender_indices[0]

                edge = self.corr.edges[hook.name][receiver_index][sender_node][sender_index]

                if edge.present:
                    if verbose:
                        print(f"Overwrote {receiver_index} with norm {old_z[receiver_index.as_index].norm().item()}")

                    hook_point_input[receiver_index.as_index] = old_z[receiver_index.as_index].to(hook_point_input.device)
    
            return hook_point_input

        assert incoming_edge_types == [EdgeType.ADDITION for _ in incoming_edge_types], f"All incoming edges should be the same type, not {incoming_edge_types}"

        # corrupted_cache (and thus z) contains the residual stream for the corrupted data
        # That is, the sum of all heads and MLPs and biases from previous layers
        hook_point_input[:] = self.global_cache.corrupted_cache[hook.name].to(hook_point_input.device) # It is crucial to use [:] to not use same tensor

        # We will now edit the input activations to this component 
        # This is one of the key reasons ACDC is slow, so the implementation is for performance
        # 
        # In general we will be looking at very few input edges.
        # So it was a design decision to compute the inputs to model components by the two step process.
        # i) set input to corrupted activation
        # ii) add back to residual_stream_in the (hopefully small number of) clean activations, by firstly subtracting their corrupted activation, and then adding back the clean activations
        
        for receiver_node_index in self.corr.edges[hook.name]:
            for sender_node_name in self.corr.edges[hook.name][receiver_node_index]:
                for sender_node_index in self.corr.edges[hook.name][receiver_node_index][sender_node_name]:

                    edge = self.corr.edges[hook.name][receiver_node_index][sender_node_name][sender_node_index] # TODO maybe less crazy nested indexes ... just make local variables each time?

                    if not edge.present:
                        continue # don't do patching stuff, if it wastes time

                    if verbose:
                        print(
                            hook.name, receiver_node_index, sender_node_name, sender_node_index,
                        )
                        print("-------")
                        if edge.edge_type == EdgeType.ADDITION:
                            print(
                                self.global_cache.online_cache[sender_node_name].shape,
                                sender_node_index,
                            )
                    
                    if edge.edge_type == EdgeType.ADDITION:
                        # Add the effect of the new head (from the current forward pass)
                        hook_point_input[receiver_node_index.as_index] += self.global_cache.online_cache[
                            sender_node_name
                        ][sender_node_index.as_index].to(hook_point_input.device)
                        # Remove the effect of this head (from the corrupted data)
                        hook_point_input[receiver_node_index.as_index] -= self.global_cache.corrupted_cache[
                            sender_node_name
                        ][sender_node_index.as_index].to(hook_point_input.device)

                    else: 
                        raise ValueError(f"Unknown edge type {edge.edge_type} ... {edge}")

        return hook_point_input

    def add_all_sender_hooks(self, reset=True, cache="online", skip_direct_computation=False, add_all_hooks=False, sender_and_receiver_both_ok=False):
        """We use add_sender_hook for lazily adding *some* sender hooks

        :param sender_and_receiver_both_ok: The sender hooks had some checks that we're not adding both adding sender
        and receiver hooks to the same HookPoint. Usually this is a sign something has gone wrong, but in the case where
        we're adding all the hooks (for test loss calculation) it's not wrong

        """

        if self.verbose:
            print("Adding sender hooks...")
        if reset:
            self.model.reset_hooks()
        device = {
            "online": "cpu" if self.online_cache_cpu else None,
            "corrupted": "cpu" if self.corrupted_cache_cpu else None,
        }[cache]

        for big_tuple, edge in self.corr.all_edges().items():
            nodes = []

            if edge.edge_type == EdgeType.DIRECT_COMPUTATION:
                if not skip_direct_computation:
                    nodes.append(self.corr.graph[big_tuple[2]][big_tuple[3]])
                    if add_all_hooks:
                        nodes.append(self.corr.graph[big_tuple[0]][big_tuple[1]])
            elif edge.edge_type == EdgeType.ADDITION:
                nodes.append(self.corr.graph[big_tuple[2]][big_tuple[3]])
                if add_all_hooks:
                    nodes.append(self.corr.graph[big_tuple[0]][big_tuple[1]])
            elif edge.edge_type != EdgeType.PLACEHOLDER:
                print(edge.edge_type.value, EdgeType.ADDITION.value, edge.edge_type.value == EdgeType.ADDITION.value, type(edge.edge_type.value), type(EdgeType.ADDITION.value))
                raise ValueError(f"{str(big_tuple)} {str(edge)} failed")

            for node in nodes:
                fwd_hooks = self.model.hook_dict[node.name].fwd_hooks
                if len(fwd_hooks) > 0 and not sender_and_receiver_both_ok:
                    resolved_hooks_dicts = [fwd_hook.hook.hooks_dict_ref() for fwd_hook in fwd_hooks]
                    assert all([resolved_hooks_dict == resolved_hooks_dicts[0] for resolved_hooks_dict in resolved_hooks_dicts]), f"{resolved_hooks_dicts}\nUnexpected behavior: different hook dict for different hooks on the same HookPoint?! https://github.com/neelnanda-io/TransformerLens/issues/297"
                    for fwd_hook in resolved_hooks_dicts[0].values():
                        if isinstance(fwd_hook, partial):
                            print("Hello!") # TODO remove
                        hook_func_name = fwd_hook.__wrapped__.__name__ if isinstance(fwd_hook, partial) else fwd_hook.__name__
                        assert "sender_hook" in hook_func_name, f"You should only add sender hooks to {node.name}, and this: {hook_func_name} doesn't look like a sender hook"
                    continue

                self.model.add_hook( # TODO is this slow part??? Speed up???
                    name=node.name, 
                    hook=partial(self.sender_hook, verbose=self.hook_verbose, cache=cache, device=device),
                )

    def setup_corrupted_cache(self):
        if self.verbose:
            print("Adding sender hooks...")

        self.model.reset_hooks()

        if self.zero_ablation:
            # to calculate the inputs to each model component, 
            # we need zero out all the outputs into the residual stream

            # all hooknames that output into the residual stream
            hook_name_substrings = ["hook_result", "mlp_out"]
            if self.use_pos_embed:
                hook_name_substrings.extend(["hook_pos_embed", "hook_embed"])
            else:
                hook_name_substrings.append("blocks.0.hook_resid_pre")

            # add hooks to zero out all these hook points
            hook_name_bool_function = lambda hook_name: any([hook_name_substring in hook_name for hook_name_substring in hook_name_substrings])
            self.model.add_hook(
                name = hook_name_bool_function,
                hook = lambda z, hook: torch.zeros_like(z),
            )
            # we now add the saving hooks AFTER we've zeroed out activations

        if self.use_pos_embed and not self.zero_ablation:    
            def scramble_positions(z, hook):
                z[:] = shuffle_tensor(z[0], seed=49)
            self.model.add_hook(
                "hook_pos_embed",
                scramble_positions,
            )
        self.model.cache_all(self.global_cache.corrupted_cache)
        corrupt_stuff = self.model(self.ref_ds)

        if self.verbose:
            print("Done corrupting things")

        if self.corrupted_cache_cpu:
            self.global_cache.to("cpu", which_caches="second")

        self.model.reset_hooks()

    def setup_model_hooks(
        self, 
        add_sender_hooks=False,
        add_receiver_hooks=False,
        doing_acdc_runs=True,
    ):
        if add_receiver_hooks:
            if doing_acdc_runs:
                warnings.warn("Deprecating adding receiver hooks before launching into ACDC runs, this may be totally broke. Ignore this warning if you are not doing ACDC runs")

            receiver_node_names = list(set([node.name for node in self.corr.nodes() if node.incoming_edge_type != EdgeType.PLACEHOLDER]))
            for receiver_name in receiver_node_names: # TODO could remove the nodes that don't have any parents...
                self.model.add_hook(
                    name=receiver_name,
                    hook=partial(self.receiver_hook, verbose=self.hook_verbose),
                )

        if add_sender_hooks: # bug fixed; crucial to add sender hooks AFTER the receivers
            self.add_all_sender_hooks(cache="online", skip_direct_computation=False, add_all_hooks=True, reset=False, sender_and_receiver_both_ok=True)


    def save_edges(self, fname):
        """Stefan's idea for fast saving!
        TODO pickling of the whole experiment work"""

        edges_list = []
        for t, e in self.corr.all_edges().items():
            if e.present and e.edge_type != EdgeType.PLACEHOLDER:
                edges_list.append((t, e.effect_size))
        
        with open(fname, "wb") as f:
            pickle.dump(edges_list, f)

    def add_sender_hook(self, node, override=False):
        if not override and len(self.model.hook_dict[node.name].fwd_hooks) > 0:
            fwd_hooks = self.model.hook_dict[node.name].fwd_hooks
            if len(fwd_hooks) > 0:
                resolved_hooks_dicts = [fwd_hook.hook.hooks_dict_ref() for fwd_hook in fwd_hooks]
                assert all([resolved_hooks_dict == resolved_hooks_dicts[0] for resolved_hooks_dict in resolved_hooks_dicts]), f"{resolved_hooks_dicts}\nUnexpected behavior: different hook dict for different hooks on the same HookPoint?! https://github.com/neelnanda-io/TransformerLens/issues/297"
                for fwd_hook in resolved_hooks_dicts[0].values():
                    hook_func_name = fwd_hook.__wrapped__.__name__ if isinstance(fwd_hook, partial) else fwd_hook.__name__
                    assert "sender_hook" in hook_func_name, f"You should only add sender hooks to {node.name}, and this: {hook_func_name} doesn't look like a sender hook"
            return False # already added, move on

        self.model.add_hook(
            name=node.name, 
            hook=partial(self.sender_hook, verbose=self.hook_verbose, cache="online", device="cpu" if self.online_cache_cpu else None),
        )

        return True

    def add_receiver_hook(self, node, override=False, prepend=False):
        if not override and len(self.model.hook_dict[node.name].fwd_hooks) > 0: # repeating code from add_sender_hooks
            fwd_hooks = self.model.hook_dict[node.name].fwd_hooks
            if len(fwd_hooks) > 0:
                resolved_hooks_dicts = [fwd_hook.hook.hooks_dict_ref() for fwd_hook in fwd_hooks]
                assert all([resolved_hooks_dict == resolved_hooks_dicts[0] for resolved_hooks_dict in resolved_hooks_dicts]), f"{resolved_hooks_dicts}\nUnexpected behavior: different hook dict for different hooks on the same HookPoint?! https://github.com/neelnanda-io/TransformerLens/issues/297"
                for fwd_hook in resolved_hooks_dicts[0].values():
                    hook_func_name = fwd_hook.__wrapped__.__name__ if isinstance(fwd_hook, partial) else fwd_hook.__name__
                    assert "receiver_hook" in hook_func_name, f"You should only add receiver hooks to {node.name}, and this: {hook_func_name} doesn't look like a receiver hook"
            return False # already added, move on

        self.model.add_hook(
            name=node.name,
            hook=partial(self.receiver_hook, verbose=self.hook_verbose),
            prepend=prepend,
        )

        return True


    def step(self, early_stop=False, testing=False):
        if self.current_node is None:
            return

        start_step_time = time.time()
        self.step_idx += 1

        self.update_cur_metric(recalc_metric=True, recalc_edges=True)
        initial_metric = self.cur_metric

        cur_metric = initial_metric
        if self.verbose:
            print("New metric:", cur_metric)

        if self.current_node.incoming_edge_type.value != EdgeType.PLACEHOLDER.value:
            added_receiver_hook = self.add_receiver_hook(self.current_node, override=True, prepend=True)

        if self.current_node.incoming_edge_type.value == EdgeType.DIRECT_COMPUTATION.value:
            # basically, because these nodes are the only ones that act as both receivers and senders
            added_sender_hook = self.add_sender_hook(self.current_node, override=True)

        is_this_node_used = False
        if self.current_node.name in ["blocks.0.hook_resid_pre", "hook_pos_embed", "hook_embed"]:
            is_this_node_used = True

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

                if edge.edge_type == EdgeType.PLACEHOLDER:
                    is_this_node_used = True
                    continue # include by default

                if self.verbose:
                    print(f"\nNode: {cur_parent=} ({self.current_node=})\n")

                edge.present = False

                if edge.edge_type == EdgeType.ADDITION:
                    added_sender_hook = self.add_sender_hook(
                        cur_parent,
                    )
                else:
                    added_sender_hook = False
                
                old_metric = self.cur_metric
                if self.second_metric is not None:
                    old_second_metric = self.cur_second_metric

                self.update_cur_metric(recalc_edges=False) # warning: gives fast evaluation, though edge count is wrong
                evaluated_metric = self.cur_metric # self.metric(self.model(self.ds)) # OK, don't calculate second metric?

                if early_stop: # for debugging the effects of one and only one forward pass WITH a corrupted edge
                    return

                if self.verbose:
                    print(
                        "Metric after removing connection to",
                        sender_name,
                        sender_index,
                        "is",
                        evaluated_metric,
                        "(and current metric " + str(old_metric) + ")",
                    )

                result = evaluated_metric - old_metric
                edge.effect_size = result

                if self.verbose:
                    print("Result is", result, end="")

                if self.abs_value_threshold:
                    result = abs(result)

                if result < self.threshold:
                    if self.verbose:
                        print("...so removing connection")
                    self.corr.remove_edge(
                        self.current_node.name, self.current_node.index, sender_name, sender_index
                    )
                    
                else: # include this edge in the graph
                    self.cur_metric = old_metric
                    if self.second_metric is not None:
                        self.cur_second_metric = old_second_metric
                    is_this_node_used = True
                    if self.verbose:
                        print("...so keeping connection")
                    edge.present = True

                self.update_cur_metric(recalc_edges=False, recalc_metric=False) # So we log current state to wandb

                if self.using_wandb:
                    log_metrics_to_wandb(
                        self,
                        current_metric = self.cur_metric,
                        parent_name = str(self.corr.graph[sender_name][sender_index]),
                        child_name = str(self.current_node),
                        result = result,
                        times = time.time(),
                    )

            self.update_cur_metric(recalc_metric=True, recalc_edges=True)
            if testing:
                break

        # TODO find an efficient way to do remove hooks sensibly

        if not is_this_node_used and self.remove_redundant:
            if self.verbose:
                print("Removing redundant node", self.current_node)
            self.remove_redundant_node(self.current_node)

        if is_this_node_used and self.current_node.incoming_edge_type.value != EdgeType.PLACEHOLDER.value:
            fname = f"ims/img_new_{self.step_idx}.png"
            show(
                self.corr,
                fname=fname,
                show_full_index=self.show_full_index,
            )
            if self.using_wandb:
                try:
                    wandb.log(
                        {"acdc_graph": wandb.Image(fname),}
                    )
                except Exception as e:
                    pass # Usually a race condition when running many jobs. It's fine to not log an image.

        # increment the current node
        self.increment_current_node()
        self.update_cur_metric(recalc_metric=True, recalc_edges=True) # so we log the correct state...

    def remove_redundant_node(self, node, safe=True, allow_fails=True):
        if safe:
            for parent_name in self.corr.edges[node.name][node.index]:
                for parent_index in self.corr.edges[node.name][node.index][parent_name]:
                    if self.corr.edges[node.name][node.index][parent_name][parent_index].present:
                        raise Exception(f"You should not be removing a node that is still used by another node {node} {(parent_name, parent_index)}")

        bfs = [node]
        bfs_idx = 0
        
        while bfs_idx < len(bfs):
            cur_node = bfs[bfs_idx]
            bfs_idx += 1

            children = self.corr.graph[cur_node.name][cur_node.index].children

            for child_node in children:
                if self.corr.edges[child_node.name][child_node.index][cur_node.name][cur_node.index].edge_type.value == EdgeType.PLACEHOLDER.value:
                    # TODO be a bit more permissive, this can include all things when we have dropped an attention head...
                    continue

                try:
                    self.corr.remove_edge(
                        child_node.name, child_node.index, cur_node.name, cur_node.index
                    )
                except KeyError as e:
                    print("Got an error", e)
                    if allow_fails:
                        continue
                    else:
                        raise e

                remove_this = True
                for parent_of_child_name in self.corr.edges[child_node.name][child_node.index]:
                    for parent_of_child_index in self.corr.edges[child_node.name][child_node.index][parent_of_child_name]:
                        if self.corr.edges[child_node.name][child_node.index][parent_of_child_name][parent_of_child_index].present:
                            remove_this = False
                            break
                    if not remove_this:
                        break
                
                if remove_this and child_node not in bfs:
                    bfs.append(child_node)

    def current_node_connected(self):
        for child_name, rest1 in self.corr.edges.items(): # TODO: use parent and children abstractions
            for child_index, rest2 in rest1.items():
                if self.current_node.name in rest2 and self.current_node.index in rest2[self.current_node.name]:
                    if rest2[self.current_node.name][self.current_node.index].present:
                        return True

        # if this is NOT connected, then remove all incoming edges, too

        self.update_cur_metric(recalc_metric=True, recalc_edges=True)
        old_metric = self.cur_metric

        parent_names = list(self.corr.edges[self.current_node.name][self.current_node.index].keys())

        for parent_name in parent_names:

            try:
                rest1 = self.corr.edges[self.current_node.name][self.current_node.index][parent_name]
            except KeyError:
                continue

            parent_indices = list(rest1.keys())

            for parent_index in parent_indices:
                try:
                    edge = self.corr.edges[self.current_node.name][self.current_node.index][parent_name][parent_index]
                except KeyError:
                    continue
                
                edge.present = False

                self.corr.remove_edge(
                    self.current_node.name, 
                    self.current_node.index, 
                    parent_name, 
                    parent_index,
                )

        self.update_cur_metric(recalc_edges=True)
        assert abs(self.cur_metric - old_metric) < 3e-3, ("Removing all incoming edges should not change the metric ... you may want to see *which* remooval in the above loop mattered, too", self.cur_metric, old_metric, self.current_node) # TODO this seems to fail quite regularly

        return False

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
        while True:
            self.current_node = self.find_next_node()
            print("We moved to ", self.current_node)

            if self.current_node is None or self.current_node_connected() or self.current_node.name in ["blocks.0.hook_resid_pre", "hook_pos_embed", "hook_embed"]:
                break

            print("But it's bad")

    def count_no_edges(self, verbose=False) -> int:
        cnt = self.corr.count_no_edges(verbose=verbose)
        if self.verbose:
            print("No edge", cnt)
        return cnt

    def reload_hooks(self):
        old_corr = self.corr
        self.corr = TLACDCCorrespondence.setup_from_model(self.model)

    def save_subgraph(self, fpath: Optional[str]=None, return_it=False) -> None:
        """Saves the subgraph as a Dictionary of all the edges, so it can be reloaded (or return that)"""

        ret = OrderedDict()
        for tupl, edge in self.corr.all_edges().items():
            receiver_name, receiver_torch_index, sender_name, sender_torch_index = tupl
            receiver_index, sender_index = receiver_torch_index.hashable_tuple, sender_torch_index.hashable_tuple
            ret[
                (receiver_name, receiver_index, sender_name, sender_index)
            ] = edge.present

        if fpath is not None:
            assert fpath.endswith(".pth")
            torch.save(ret, fpath)

        if return_it:
            return ret

    def load_subgraph(self, subgraph: Subgraph):

        # assert formatting is correct
        set_of_edges = set()
        for tupl, edge in self.corr.all_edges().items():
            receiver_name, receiver_torch_index, sender_name, sender_torch_index = tupl
            receiver_index, sender_index = receiver_torch_index.hashable_tuple, sender_torch_index.hashable_tuple
            set_of_edges.add((receiver_name, receiver_index, sender_name, sender_index))
        assert set(subgraph.keys()) == set_of_edges, f"Ensure that the dictionary includes exactly the correct keys... e.g missing {list( set(set_of_edges) - set(subgraph.keys()) )[:1]} and has excess stuff { list(set(subgraph.keys()) - set_of_edges)[:1] }"

        print("Editing all edges...")
        for (receiver_name, receiver_index, sender_name, sender_index), is_present in subgraph.items():
            receiver_torch_index, sender_torch_index = TorchIndex(receiver_index), TorchIndex(sender_index)
            edge = self.corr.edges[receiver_name][receiver_torch_index][sender_name][sender_torch_index]
            edge.present = is_present
        print("Done!")

    def load_from_wandb_run(
        self,
        log_text: str,
    ) -> None:
        """Set the current subgraph equal to one from a wandb run"""

        # initialize by marking no edges as present
        for _, edge in self.corr.all_edges().items():
            edge.present = False

        found_at_least_one_readable_line = False
        lines = log_text.split("\n")

        for i, line in enumerate(lines):
            removing_connection = "...so removing connection" in line
            keeping_connection = "...so keeping connection" in line

            if removing_connection or keeping_connection:
                # check that the readable line is well readable

                found_at_least_one_readable_line = True
                for back_index in range(6):
                    previous_line = lines[i-back_index] # magic number because of the formatting of the log lines
                    if previous_line.startswith("Node: "):
                        break
                else:
                    raise ValueError("Didn't find corresponding node line")
                parent_name, parent_list, current_name, current_list = extract_info(previous_line)
                parent_torch_index, current_torch_index = TorchIndex(parent_list), TorchIndex(current_list)
                self.corr.edges[current_name][current_torch_index][parent_name][parent_torch_index].present = keeping_connection

        assert found_at_least_one_readable_line, f"No readable lines found in the log file. Is this formatted correctly ??? {lines=}"
        
    def remove_all_non_attention_connections(self):
        # remove all connection except the MLP connections
        includes_attention = [ # substrings of hook names that imply they're related to attention
            "attn",
            "hook_q",
            "hook_k",
            "hook_v",
        ]

        for receiver_name in self.corr.edges:
            for receiver_index in self.corr.edges[receiver_name]:
                for sender_name in self.corr.edges[receiver_name][receiver_index]:
                    for receiever_index in self.corr.edges[receiver_name][receiver_index][sender_name]:
                        related_to_attention = False

                        for substr in includes_attention:
                            if substr in sender_name or substr in receiver_name:
                                related_to_attention = True
                                break

                        if not related_to_attention:
                            continue

                        edge = self.corr.edges[receiver_name][receiver_index][sender_name][receiever_index]
                        edge.present = False

    def add_back_head(self, layer_idx, head_idx):

        raise NotImplementedError("This is wrong do not use")

        all_edges = self.corr.all_edges()
        for (tupl, edge) in all_edges.items():
            for hook_name, hook_idx in [(tupl[0], tupl[1]), (tupl[2], tupl[3])]:
                if hook_name.startswith(f"blocks.{layer_idx}") and len(hook_idx.hashable_tuple) >= 3 and int(hook_idx.hashable_tuple[2]) == head_idx:
                    assert edge.present == False or edge.edge_type == EdgeType.PLACEHOLDER, (tupl, edge, hook_name, hook_idx)
                    edge.present = True
                    break # don't double remove


    def call_metric_with_corr(self, corr: TLACDCCorrespondence, metric_fn: Callable[[torch.Tensor], T], data: torch.Tensor) -> T:
        """Call a function ``metric_fn`` with a new correspondence ``corr``.

        Remember to call ``self.setup_cache()`` with the desired ``ref_ds`` before this function.
        """

        old_exp_corr = self.corr
        try:
            self.corr = corr
            self.model.reset_hooks()
            self.setup_model_hooks(
                add_sender_hooks=True,
                add_receiver_hooks=True,
                doing_acdc_runs=False,
            )
            out = metric_fn(self.model(data))
        finally:
            self.corr = old_exp_corr
        return out
