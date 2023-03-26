#%%

import IPython
if IPython.get_ipython() is not None:
    IPython.get_ipython().run_line_magic("load_ext", "autoreload")  # type: ignore
    IPython.get_ipython().run_line_magic("autoreload", "2")  # type: ignore

from copy import deepcopy
from typing import List, Tuple, Dict, Any, Optional, Union, Callable, TypeVar, Iterable, Set
import wandb
import IPython
import torch
# from easy_transformer.ioi_dataset import IOIDataset  # type: ignore
from tqdm import tqdm
import random
from functools import *
import json
import pathlib
import warnings
import time
import networkx as nx
import os
import torch
import graphviz
from enum import Enum
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
pio.renderers.default = "colab"
device = "cuda" if torch.cuda.is_available() else "cpu"
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens.HookedTransformer import (
    HookedTransformer,
)
from transformer_lens.utils import make_nd_dict, TorchIndex # these introduce several important classes !!!
from collections import defaultdict, deque, OrderedDict
from transformer_lens.induction.utils import kl_divergence, toks_int_values, toks_int_values_other, good_induction_candidates, validation_data, build_colorscheme, show

#rust%%

tl_model = HookedTransformer.from_pretrained(
    "redwood_attn_2l",
    use_global_cache=True,
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
)
tl_model.set_use_attn_result(True)
tl_model.set_use_split_qkv_input(True)

# %%

base_model_logits = tl_model(toks_int_values)
base_model_probs = F.softmax(base_model_logits, dim=-1)

#%%

metric = partial(kl_divergence, base_model_probs=base_model_probs)

#%%

full_graph = make_nd_dict(end_type=bool, n=4) # TODO really global variable
all_senders_names = set() # get receivers by .keys()

residual_stream_items: Dict[str, List[TorchIndex]] = defaultdict(list)
residual_stream_items[f"blocks.{tl_model.cfg.n_layers-1}.hook_resid_post"] = [TorchIndex([None])] # TODO the position version 

def add_edge(
    receiver_name: str,
    receiver_slice: TorchIndex,
    sender_name: str,
    sender_slice: TorchIndex,
):
    full_graph[receiver_name][receiver_slice][sender_name][sender_slice] = True
    all_senders_names.add(sender_name)

for layer_idx in range(tl_model.cfg.n_layers-1, -1, -1):
    # connect MLPs
    if not tl_model.cfg.attn_only:
        raise NotImplementedError() # TODO

    # connect attention heads
    for head_idx in range(tl_model.cfg.n_heads-1, -1, -1):
        # this head writes to all future residual stream things
        cur_head_name = f"blocks.{layer_idx}.attn.hook_result"
        cur_head_slice = TorchIndex([None, None, head_idx])
        for receiver_name in residual_stream_items:
            for receiver_slice_tuple in residual_stream_items[receiver_name]:
                add_edge(receiver_name, receiver_slice_tuple, cur_head_name, cur_head_slice)

    for head_idx in range(tl_model.cfg.n_heads-1, -1, -1):        
        for letter in "qkv": # plausibly we could prototype even slower, by using just input to each head...
            # eventually add the positional shit heeere
            letter_hook_name = f"blocks.{layer_idx}.hook_{letter}_input"       
            letter_tuple_slice = TorchIndex([None, None, head_idx])
            residual_stream_items[letter_hook_name].append(letter_tuple_slice)

# add the embedding node
for receiver_name in residual_stream_items:
    for receiver_slice_tuple in residual_stream_items[receiver_name]:
        add_edge(receiver_name, receiver_slice_tuple, "blocks.0.hook_resid_pre", TorchIndex([None]))

# %%

if False:
    current_graph = deepcopy(full_graph)
    nodes = OrderedDict() # used as an OrderedSet replacement
    for node_name in full_graph.keys():
        for node_slice in full_graph[node_name].keys():
            nodes[(node_name, node_slice)] = None

#%%

# add the saving hooks

def sender_hook(z, hook, verbose=False, cache="first"):
    """General, to cover online and corrupt caching"""

    if cache == "second":
        hook.global_cache.second_cache[hook.name] = z.clone()
    elif cache == "first":
        hook.global_cache.cache[hook.name] = z.clone()
    else:
        raise ValueError(f"Unknown cache type {cache}")

    if verbose:
        print(f"Saved {hook.name} with norm {z.norm().item()}")

    return z

tl_model.reset_hooks()
for name in all_senders_names:
    tl_model.add_hook(name=name, hook=partial(sender_hook, verbose=False, cache="second"))
corrupt_stuff = tl_model(toks_int_values_other)
tl_model.reset_hooks()

#%%

b_O = tl_model.blocks[0].attn.b_O

def receiver_hook(z, hook, verbose=True):
    for receiver_slice_tuple in full_graph[hook.name].keys():
        for sender_name in full_graph[hook.name][receiver_slice_tuple].keys():
            for sender_slice_tuple in full_graph[hook.name][receiver_slice_tuple][sender_name]:
                if full_graph[hook.name][receiver_slice_tuple][sender_name][sender_slice_tuple]:
                    # do "actually" include this edge
                    continue

                if verbose:
                    print(hook.name, receiver_slice_tuple, sender_name, sender_slice_tuple)
                    print("-------")
                    # print(receiver_slice_tuple.as_index, sender_slice_tuple.as_index)
                    print(hook.global_cache.cache[sender_name].shape, sender_slice_tuple.as_index)


                z[receiver_slice_tuple.as_index] # this checks where the indexing is bugged
                z[receiver_slice_tuple.as_index] -= hook.global_cache.cache[sender_name][sender_slice_tuple.as_index]
                z[receiver_slice_tuple.as_index] += hook.global_cache.second_cache[sender_name][sender_slice_tuple.as_index]

    # assert torch.allclose(z, torch.zeros_like(z)) or torch.allclose(z, b_O)
    return z

# add both sender and receiver hooks
for receiver_name in full_graph.keys():
    tl_model.add_hook(
        name=receiver_name,
        hook=receiver_hook,
    )
for sender_name in all_senders_names:
    tl_model.add_hook(
        name=sender_name,
        hook=partial(sender_hook, cache="first"),
    )

#%%

ans = tl_model(toks_int_values) # torch.arange(5) # note that biases mean not EVERYTHING is zero ablated
new_metric = metric(ans)
assert abs(new_metric) < 1e-5, f"Metric {new_metric} is not zero"

#%%

show(full_graph, "test.png")

#%%

def step(
    graph,
    threshold,
    receiver_name,
    receiver_slice_tuple,
    verbose=True,
    using_wandb=False,
    remove_redundant=False,
    early_stop=False,
):
    """
    TOIMPLEMENT: (prioritise these)
    - Incoming effect sizes on the nodes
    - Dynamic threshold?
    - Node stats
    - Parallelism
    - Not just monotone metric
    - wandb
    - remove_redundant
    """
    warnings.warn("Implement incoming effect sizes on the nodes")
    start_step_time = time.time()

    initial_metric = metric(tl_model(toks_int_values))
    assert isinstance(initial_metric, float), "Make this float"
    cur_metric = initial_metric
    if verbose:
        print("New metric:", cur_metric)

    for sender_name in graph[receiver_name][receiver_slice_tuple]:
        print(f"\nNode name: {sender_name}\n")

        for sender_slice_tuple in graph[receiver_name][receiver_slice_tuple][sender_name]:
            graph[receiver_name][receiver_slice_tuple][sender_name][sender_slice_tuple] = False

            if early_stop:
                return tl_model(toks_int_values)

            evaluated_metric = metric(tl_model(toks_int_values))

            if verbose:
                print(
                    "Metric after removing connection to",
                    sender_name,
                    sender_slice_tuple,
                    "is",
                    evaluated_metric,
                    "(and current metric " + str(cur_metric) + ")",
                )

            result = evaluated_metric - cur_metric

            if verbose:
                print("Result is", result, end="")

            if result < threshold:
                print("...so removing connection")
                cur_metric = evaluated_metric

            else:
                if verbose:
                    print("...so keeping connection")
                graph[receiver_name][receiver_slice_tuple][sender_name][sender_slice_tuple] = True

            if using_wandb:
                raise NotImplementedError("WANDB not implemented yet")
    
    if all(not graph[receiver_name][receiver_slice_tuple][sender_name][sender_slice_tuple] for sender_name in graph[receiver_name] for sender_slice_tuple in graph[receiver_name][receiver_slice_tuple][sender_name]):
        # removed all connections
        print(
            f"Warning: we added {receiver_name} at {receiver_slice_tuple} earlier, but we just removed all its child connections. So we are{(' ' if remove_redundant else ' not ')}removing it and its redundant descendants now (remove_redundant=remove_redundant)"
        )

    # TODO: this is annoying
    if using_wandb:
        raise NotImplementedError()
    

#%%

old = tl_model(toks_int_values)

#%%

# Let's do some ACDC !!!

for receiver_name in full_graph.keys():
    for receiver_slice_tuple in full_graph[receiver_name]:

        print("Currently at ", receiver_name, "and the tuple is", receiver_slice_tuple)

        new = step(
            graph=full_graph,
            threshold=0.0,
            receiver_name=receiver_name,
            receiver_slice_tuple=receiver_slice_tuple,
            verbose=True,
            early_stop=False,
        )

        assert False


#%%

def zero_hook(z, hook):
    return torch.zeros_like(z)

def run_in_normal_way(tl_model):
    tl_model.reset_hooks()
    for hook_name in ['blocks.0.attn.hook_result', 'blocks.0.hook_resid_pre', 'blocks.1.attn.hook_result']:
        tl_model.add_hook(hook_name, zero_hook)
    answer = tl_model(torch.arange(5))
    tl_model.reset_hooks()
    return answer

b2 = run_in_normal_way(tl_model)
assert torch.allclose(b, b2)

# %%

for step_idx in range(int(1e6)):
    node_name, node_slice = list(full_graph.keys())[0]
    full_graph.pop((node_name, node_slice))

    for sender_name in full_graph[node_name][node_slice]:
        for sender_slice in full_graph[node_name][node_slice][sender_name]:
            # try and remove this
            full_graph[node_name][node_slice][sender_name].remove(sender_slice)

