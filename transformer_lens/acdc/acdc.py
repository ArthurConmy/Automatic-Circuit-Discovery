# %%

import IPython

if IPython.get_ipython() is not None:
    IPython.get_ipython().run_line_magic("load_ext", "autoreload")  # type: ignore
    IPython.get_ipython().run_line_magic("autoreload", "2")  # type: ignore

from copy import deepcopy
from typing import (
    List,
    Tuple,
    Dict,
    Any,
    Optional,
    Union,
    Callable,
    TypeVar,
    Iterable,
    Set,
)
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
from transformer_lens.acdc.utils import (
    make_nd_dict,
    TLACDCInterpNode,
    TLACDCCorrespondence,
    TorchIndex,
    Edge,
    EdgeType,
)  # these introduce several important classes !!!
from collections import defaultdict, deque, OrderedDict
from transformer_lens.acdc.induction.utils import (
    kl_divergence,
    toks_int_values,
    toks_int_values_other,
    good_induction_candidates,
    validation_data,
)
from transformer_lens.acdc.graphics import (
    build_colorscheme,
    show,
)
from transformer_lens.acdc.utils import count_no_edges
import argparse

#%%

parser = argparse.ArgumentParser()
parser.add_argument('--first-cache-cpu', type=bool, required=False, default=True, help='Value for FIRST_CACHE_CPU')
parser.add_argument('--second-cache-cpu', type=bool, required=False, default=True, help='Value for SECOND_CACHE_CPU')
# parser.add_argument('--num-examples', type=int, required=False, default=40, help='Value for NUM_EXAMPLES') # TODO integrate with utils files
# parser.add_argument('--seq-len', type=int, required=False, default=300, help='Value for SEQ_LEN')
parser.add_argument('--using-wandb', type=bool, required=False, default=True, help='Value for USING_WANDB')
parser.add_argument('--threshold', type=float, required=False, default=1.0, help='Value for THRESHOLD')
parser.add_argument('--zero-ablation', action='store_true', help='A flag without a value')

if IPython.get_ipython() is not None: # heheh get around this failing in notebooks
    args = parser.parse_args("".split())
    # args = parser.parse_args("--zero-ablation".split())
else:
    args = parser.parse_args()

FIRST_CACHE_CPU = args.first_cache_cpu
SECOND_CACHE_CPU = args.second_cache_cpu
# NUM_EXAMPLES = args.num_examples
# SEQ_LEN = args.seq_len
USING_WANDB = args.using_wandb
THRESHOLD = args.threshold

print(f"{args.zero_ablation=}")

if args.zero_ablation:
    ZERO_ABLATION = True
else:
    ZERO_ABLATION = False
#%%

tl_model = HookedTransformer.from_pretrained(
    "redwood_attn_2l",
    use_global_cache=True,
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
)
tl_model.set_use_attn_result(True)
tl_model.set_use_split_qkv_input(True)

#%%

base_model_logits = tl_model(toks_int_values)
base_model_probs = F.softmax(base_model_logits, dim=-1)

# %%

raw_metric = partial(kl_divergence, base_model_probs=base_model_probs, using_wandb=False)
metric = partial(kl_divergence, base_model_probs=base_model_probs, using_wandb=USING_WANDB)

#%%

if False: # a test of the zero ablation stuff
    def zer(z, hook, head):
        z[:, :, head] = 0.0

    for layer in range(0, 2):
        for head in range(0, 8):
            if (layer, head) not in [(0, 0), (1, 5), (1, 6)]:
                tl_model.add_hook(
                    name=f"blocks.{layer}.attn.hook_result",
                    hook=partial(zer, head=head),
                )
    
    zer_logits = tl_model(toks_int_values)
    zer_probs = F.softmax(zer_logits, dim=-1)
    zer_metric = raw_metric(zer_probs)
    print(zer_metric, "is the result...")
    tl_model.reset_hooks()
# %%

correspondence = TLACDCCorrespondence()

upstream_residual_nodes: List[TLACDCInterpNode] = []
logits_node = TLACDCInterpNode(
    name=f"blocks.{tl_model.cfg.n_layers-1}.hook_resid_post",
    index=TorchIndex([None]),
    mode="addition",
)
upstream_residual_nodes.append(logits_node)

for layer_idx in range(tl_model.cfg.n_layers - 1, -1, -1):
    # connect MLPs
    if not tl_model.cfg.attn_only:
        raise NotImplementedError()  # TODO

    # connect attention heads
    for head_idx in range(tl_model.cfg.n_heads - 1, -1, -1):
        # this head writes to all future residual stream things
        cur_head_name = f"blocks.{layer_idx}.attn.hook_result"
        cur_head_slice = TorchIndex([None, None, head_idx])
        cur_head = TLACDCInterpNode(
            name=cur_head_name,
            index=cur_head_slice,
            mode="addition",
        )
        for residual_stream_node in upstream_residual_nodes:
            correspondence.add_edge(
                parent_node=residual_stream_node,
                child_node=cur_head,
            )

        # TODO maybe this needs be moved out of this block??? IDK
        hook_letter_inputs = {}
        for letter in "qkv":
            hook_letter_name = f"blocks.{layer_idx}.attn.hook_{letter}"
            hook_letter_slice = TorchIndex([None, None, head_idx])
            hook_letter_node = TLACDCInterpNode(name=hook_letter_name, index=hook_letter_slice, mode="off")
            
            hook_letter_input_name = f"blocks.{layer_idx}.hook_{letter}_input"
            hook_letter_input_slice = TorchIndex([None, None, head_idx])
            hook_letter_input_node = TLACDCInterpNode(
                name=hook_letter_input_name, index=hook_letter_input_slice, mode="direct_computation"
            )

            correspondence.add_edge(
                parent_node=hook_letter_input_node,
                child_node=hook_letter_node,
            )

            # Surely this doesn't need be added anywhere else?
            upstream_residual_nodes.append(hook_letter_input_node)

# add the embedding node

embedding_node = TLACDCInterpNode(
    name="blocks.0.hook_resid_pre",
    index=TorchIndex([None]),
    mode="addition",
)
for node in upstream_residual_nodes:
    correspondence.add_edge(
        parent_node=node,
        child_node=embedding_node,
    )

# %%

# add the saving hooks

def sender_hook(z, hook, verbose=False, cache="first", device=None):
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
tl_model.reset_hooks()
for node in correspondence.nodes():
    if node.mode != "addition":
        continue
    if len(tl_model.hook_dict[node.name].fwd_hooks) > 0:
        continue
    tl_model.add_hook(
        name=node.name, 
        hook=partial(sender_hook, verbose=False, cache="second", device="cpu" if FIRST_CACHE_CPU else None),
    )

corrupt_stuff = tl_model(toks_int_values_other)

if ZERO_ABLATION:
    for name in tl_model.global_cache.second_cache:
        tl_model.global_cache.second_cache[name] = torch.zeros_like(
            tl_model.global_cache.second_cache[name]
        )
        torch.cuda.empty_cache()

tl_model.reset_hooks()

#%%

if SECOND_CACHE_CPU:
    tl_model.global_cache.to("cpu", which_caches="second")

# %%

b_O = tl_model.blocks[0].attn.b_O # lolol

def receiver_hook(z, hook, verbose=False):
    
    receiver_name = hook.name
    found_direct_computation = False

    for receiver_node in correspondence.graph[receiver_name]:
        for sender_node in receiver_node.parents:

            # implement skipping edges post-hoc
            if verbose:
                # TODO delete this eventually, but useful scrappy debugging
                print(
                    hook.name, receiver_node.index, sender_node.name, sender_node.index.index
                )
                print("-------")
                print(
                    hook.global_cache.cache[sender_name].shape,
                    sender_node.index.index.as_index,
                )
            
            if sender_node.mode == "addition": # TODO turn into ENUM
                z[receiver_node.index.as_index] -= hook.global_cache.cache[
                    sender_name
                ][sender_node.index.index.as_index].to(z.device)
                z[receiver_node.index.as_index] += hook.global_cache.second_cache[
                    sender_name
                ][sender_node.index.index.as_index].to(z.device)

            elif sender_node.mode == "direct_computation":
                assert not found_direct_computation, "Found multiple direct computation nodes"
                found_direct_computation = True

                z[receiver_node.index.as_index] = hook.global_cache.second_cache[receiver_name][receiver_node.index.as_index].to(z.device)

            else: 
                assert sender_node.mode == "off", f"Unknown edge type {sender_node.mode}"

    return z

# add both sender and receiver hooks

tl_model.reset_hooks()
sender_node_names = list(set([node.name for node in correspondence.nodes() if node.mode == "addition"]))
for sender_name in sender_node_names:
    tl_model.add_hook(
        name=sender_name,
        hook=partial(sender_hook, cache="first"),
    )

receiver_node_names = list(set([node.name for node in correspondence.nodes()]))
for receiver_name in receiver_node_names: # TODO could remove the nodes that don't have any parents...
    tl_model.add_hook(
        name=receiver_name,
        hook=receiver_hook,
    )

# %%

# TODO: why are the sender hooks not storing anything???

ans = tl_model(
    toks_int_values
)  # torch.arange(5) # note that biases mean not EVERYTHING is zero ablated
new_metric = raw_metric(ans)
assert abs(new_metric) < 1e-5, f"Metric {new_metric} is not zero"

# %%

show(full_graph, "test.png")

# %%

def step(
    graph: FullGraphT,
    threshold,
    receiver_name,
    receiver_node.index,
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
    - remove_redundant
    - just record the metrics that we actually stick to
    """

    warnings.warn("Implement incoming effect sizes on the nodes")
    start_step_time = time.time()

    initial_metric = metric(tl_model(toks_int_values))
    assert isinstance(initial_metric, float), "Make this float"
    cur_metric = initial_metric
    if verbose:
        print("New metric:", cur_metric)

    for sender_name in graph[receiver_name][receiver_node.index]:
        print(f"\nNode name: {sender_name}\n")

        for sender_node.index.index in graph[receiver_name][receiver_node.index][sender_name]:
            graph[receiver_name][receiver_node.index][sender_name][sender_node.index.index].present = False

            if early_stop: # for debugging the effects of one and only one forward pass WITH a corrupted edge
                return tl_model(toks_int_values)

            evaluated_metric = metric(tl_model(toks_int_values))

            if verbose:
                print(
                    "Metric after removing connection to",
                    sender_name,
                    sender_node.index.index,
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
                graph[receiver_name][receiver_node.index][sender_name][
                    sender_node.index.index
                ].present = True

            if using_wandb:
                wandb.log({"num_edges": count_no_edges(graph)})

    cur_metric = metric(tl_model(toks_int_values))

    if all(
        not graph[receiver_name][receiver_node.index][sender_name][sender_node.index.index].present
        for sender_name in graph[receiver_name][receiver_node.index].keys()
        for sender_node.index.index in graph[receiver_name][receiver_node.index][sender_name]
    ):
        # removed all connections
        print(
            f"Warning: we added {receiver_name} at {receiver_node.index} earlier, but we just removed all its child connections. So we are{(' ' if remove_redundant else ' not ')}removing it and its redundant descendants now (remove_redundant={remove_redundant})"
        )

# %%

old = tl_model(toks_int_values)
print(count_no_edges(full_graph), "is the full number of edges")

# %%

# Let's do some ACDC !!!

import wandb
import random

file_content = ""
with open(__file__, "r") as f:
    file_content = f.read()

threshold = THRESHOLD

#%%

if USING_WANDB:
    wandb.init(
        entity="remix_school-of-rock", 
        project="tl_induction_proper", 
        name="arthurs_example_threshold_" + str(threshold) + "_" + ("_zero" if ZERO_ABLATION else "") + str(random.randint(0, 1000000)),
        notes=file_content,
    )

for receiver_name in full_graph.keys():
    for receiver_node.index in full_graph[receiver_name]:
        print("Currently at ", receiver_name, "and the tuple is", receiver_node.index)

        # TODO c'mon... you must be able to implement a speedup where you check if this position doesn't matter at all...

        step(
            graph=full_graph,
            threshold=threshold,
            receiver_name=receiver_name,
            receiver_node.index=receiver_node.index,
            verbose=True,
            early_stop=False,
            using_wandb=USING_WANDB,
        )

        # TODO implement not backtracking where it doesn't matter

print("Done!!!")
final_edges = count_no_edges(full_graph)
print(final_edges) # ... I think that this is the number of edges!

if USING_WANDB:
    wandb.log({"num_edges": final_edges})
    wandb.finish()

#%%

show(full_graph, f"ims/{threshold}.png")


for receiver_name in full_graph.keys():
    for receiver_node.index in full_graph[receiver_name]:
        for sender_name in full_graph[receiver_name][receiver_node.index].keys():
            for sender_node.index.index in full_graph[receiver_name][receiver_node.index][sender_name]:
                if full_graph[receiver_name][receiver_node.index][sender_name][sender_node.index.index].present:
                    print("Connection from", sender_name, sender_node.index.index, "to", receiver_name, receiver_node.index)

# %%

if False:
    """A good way to sanity check our wild hooking stuff is to check zero ablating everything is the same as zero ablating everything in the normal way. For now this is not implemented"""

    def zero_hook(z, hook):
        return torch.zeros_like(z)

    def run_in_normal_way(tl_model):
        tl_model.reset_hooks()
        for hook_name in [
            "blocks.0.attn.hook_result",
            "blocks.0.hook_resid_pre",
            "blocks.1.attn.hook_result",
        ]:
            tl_model.add_hook(hook_name, zero_hook)
        answer = tl_model(torch.arange(5))
        tl_model.reset_hooks()
        return answer


    b2 = run_in_normal_way(tl_model)
    assert torch.allclose(b, b2)  # TODO fix this, I dunno if we have settings for the
