#%%

"""Currently a notebook so that I can develop the 16 Heads tests fast"""

from IPython import get_ipython
if get_ipython() is not None:
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')

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
import pickle
import wandb
import IPython
import torch
from pathlib import Path
from tqdm import tqdm
import random
from functools import partial
import json
import pathlib
import warnings
import time
import networkx as nx
import os
import torch
import huggingface_hub
import graphviz
from enum import Enum
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from tqdm import tqdm
import yaml
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
pio.renderers.default = "colab"
from acdc.hook_points import HookedRootModule, HookPoint
from acdc.graphics import show
from acdc.HookedTransformer import (
    HookedTransformer,
)
from acdc.tracr.utils import get_tracr_data, get_tracr_model_input_and_tl_model
from acdc.docstring.utils import get_all_docstring_things, get_docstring_model
from acdc.acdc_utils import (
    make_nd_dict,
    shuffle_tensor,
    cleanup,
    ct,
    TorchIndex,
    Edge,
    EdgeType,
)  # these introduce several important classes !!!
from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from acdc.TLACDCInterpNode import TLACDCInterpNode
from acdc.TLACDCExperiment import TLACDCExperiment
from collections import defaultdict, deque, OrderedDict
from acdc.acdc_utils import (
    kl_divergence,
)
from acdc.ioi.utils import (
    get_ioi_data,
    get_gpt2_small,
)
from acdc.induction.utils import (
    get_all_induction_things,
    get_model,
    get_validation_data,
    get_good_induction_candidates,
    get_mask_repeat_candidates,
)
from acdc.greaterthan.utils import get_all_greaterthan_things
from acdc.graphics import (
    build_colorscheme,
    show,
)
import argparse

#%%

parser = argparse.ArgumentParser(description="Used to launch ACDC runs. Only task and threshold are required")
parser.add_argument('--task', type=str, required=True, choices=['ioi', 'docstring', 'induction', 'tracr', 'greaterthan'], help='Choose a task from the available options: ioi, docstring, induction, tracr (no guarentee I implement all...)')
parser.add_argument('--zero-ablation', action='store_true', help='Use zero ablation')
parser.add_argument('--wandb-entity-name', type=str, required=False, default="remix_school-of-rock", help='Value for WANDB_ENTITY_NAME')
parser.add_argument('--wandb-group-name', type=str, required=False, default="default", help='Value for WANDB_GROUP_NAME')
parser.add_argument('--wandb-project-name', type=str, required=False, default="acdc", help='Value for WANDB_PROJECT_NAME')
parser.add_argument('--wandb-run-name', type=str, required=False, default=None, help='Value for WANDB_RUN_NAME')
parser.add_argument('--device', type=str, default="cuda")

# for now, force the args to be the same as the ones in the notebook, later make this a CLI tool
if get_ipython() is not None: # heheh get around this failing in notebooks
    args = parser.parse_args("--task docstring --wandb-run-name docstring_sixteen_heads".split())
else:
    args = parser.parse_args()

#%%

TASK = args.task
ZERO_ABLATION = True if args.zero_ablation else False
WANDB_ENTITY_NAME = args.wandb_entity_name
WANDB_PROJECT_NAME = args.wandb_project_name
WANDB_RUN_NAME = args.wandb_run_name
WANDB_GROUP_NAME = args.wandb_group_name
DEVICE = args.device
DO_CHECKING_RECALC = False # testing only

#%%

"""Mostly copied from acdc/main.py"""

if TASK == "ioi":
    num_examples = 100 
    tl_model = get_gpt2_small(device=DEVICE, sixteen_heads=True)
    toks_int_values, toks_int_values_other, metric = get_ioi_data(tl_model, num_examples, kl_return_one_element=False)
    assert len(toks_int_values) == len(toks_int_values_other) == num_examples, (len(toks_int_values), len(toks_int_values_other), num_examples)
    seq_len = toks_int_values.shape[1]
    model_getter = get_gpt2_small

if TASK == "greaterthan":
    num_examples = 100
    tl_model, toks_int_values, prompts, metric = get_all_greaterthan_things(num_examples=num_examples, device=DEVICE, sixteen_heads=True, return_tensor=True, return_one_element=False)
    toks_int_values_other = toks_int_values.clone()
    toks_int_values_other[:, 7] = 486 # replace with 01
    seq_len = toks_int_values.shape[1]
    model_getter = get_gpt2_small

elif TASK == "docstring":
    num_examples = 50
    seq_len = 41
    docstring_things = get_all_docstring_things(
        num_examples=num_examples, 
        seq_len=seq_len,
        device=DEVICE,
        metric_name="kl_div", 
        correct_incorrect_wandb=True,
        sixteen_heads=True,
        return_one_element=False,
    )
    tl_model, toks_int_values, toks_int_values_other = docstring_things.tl_model, docstring_things.validation_data, docstring_things.validation_patch_data

    metric = docstring_things.validation_metric # we take this as metric, because it splits

    test_metric_fns = docstring_things.test_metrics
    test_metric_data = docstring_things.test_data

    model_getter = get_docstring_model

elif TASK == "induction":
    raise NotImplementedError("Induction has same sentences with multiple places we take loss / KL divergence; fiddlier implementation")

# note to self: turn of split_qkv for less OOM

else:
    raise NotImplementedError("TODO")

# %%

assert not tl_model.global_cache.sixteen_heads_config.forward_pass_enabled

with torch.no_grad():
    _, corrupted_cache = tl_model.run_with_cache(
        toks_int_values_other,
    )
corrupted_cache.to("cpu")
tl_model.zero_grad()
tl_model.global_cache.second_cache = corrupted_cache

#%%
# [markdown]
# <h1>Try a demo backwards pass of the model</h1>

tl_model.global_cache.sixteen_heads_config.forward_pass_enabled = True
clean_cache = tl_model.add_caching_hooks(
    # toks_int_values,
    incl_bwd=True,
)
clean_logits = tl_model(toks_int_values)
metric_result = metric(clean_logits)
assert list(metric_result.shape) == [num_examples], metric_result.shape
metric_result = metric_result.sum() / len(metric_result)
metric_result.backward(retain_graph=True)

#%%

keys = []
for layer_idx in range(tl_model.cfg.n_layers):
    for head_idx in range(tl_model.cfg.n_heads):
        keys.append((layer_idx, head_idx))

results = {
    (layer_idx, head_idx): torch.zeros(size=(num_examples,))
    for layer_idx, head_idx in keys
}

# %%

kls = {
    (layer_idx, head_idx): torch.zeros(size=(num_examples,))
    for layer_idx, head_idx in results.keys()
}

from tqdm import tqdm

for i in tqdm(range(num_examples)):
    tl_model.zero_grad()
    tl_model.reset_hooks()
    clean_cache = tl_model.add_caching_hooks(names_filter=lambda name: "hook_result" in name, incl_bwd=True)
    clean_logits = tl_model(toks_int_values)
    kl_result = metric(clean_logits)[i]
    kl_result.backward(retain_graph=True)

    for layer_idx in range(tl_model.cfg.n_layers):
        fwd_hook_name = f"blocks.{layer_idx}.attn.hook_result"

        for head_idx in range(tl_model.cfg.n_heads):
            g = (
                tl_model.hook_dict[fwd_hook_name]
                .xi.grad[0, 0, head_idx, 0]
                .norm()
                .item()
            )
            kls[(layer_idx, head_idx)][i] = g

    tl_model.zero_grad()
    tl_model.reset_hooks()
    del clean_cache
    del clean_logits
    import gc; gc.collect()
    torch.cuda.empty_cache()

for k in kls:
    kls[k].to("cpu")

#%%

if DO_CHECKING_RECALC:
    for i in tqdm(range(num_examples)):
        tl_model.zero_grad()
        tl_model.reset_hooks()

        clean_cache = tl_model.add_caching_hooks(incl_bwd=True)
        clean_logits = tl_model(toks_int_values)
        kl_result = metric(clean_logits)[i]
        kl_result.backward(retain_graph=True)

        for layer_idx in range(tl_model.cfg.n_layers):
            fwd_hook_name = f"blocks.{layer_idx}.attn.hook_result"
            bwd_hook_name = f"blocks.{layer_idx}.attn.hook_result_grad"

            cur_results = torch.abs( # TODO implement abs and not abs???
                torch.einsum(
                    "bshd,bshd->bh",
                    clean_cache[bwd_hook_name], # gradient
                    clean_cache[fwd_hook_name]- (0.0 if ZERO_ABLATION else corrupted_cache[fwd_hook_name].to(DEVICE)),
                )
            )

            for head_idx in range(tl_model.cfg.n_heads):
                results_entry = cur_results[i, head_idx].item()
                results[(layer_idx, head_idx)][i] = results_entry

        del clean_cache
        del clean_logits
        tl_model.reset_hooks()
        tl_model.zero_grad()
        torch.cuda.empty_cache()
        import gc; gc.collect()

    for k in results:
        results[k].to("cpu")

    for k in results:
        print(k, results[k].norm().item(), kls[k].norm().item())  # should all be close!!!
        if k[1] == None: continue
        assert torch.allclose(results[k], kls[k]) # oh lol we forgot the MLPs ... and then later I remove these as I don't think HISP is using them

#%%

kl_dict = deepcopy(kls)
scores_list = torch.zeros(size=(tl_model.cfg.n_layers, tl_model.cfg.n_heads))
mask_list = []
for layer_idx in range(tl_model.cfg.n_layers):
    for head_idx in range(tl_model.cfg.n_heads):
        score = kl_dict[(layer_idx, head_idx)].sum()
        scores_list[layer_idx, head_idx] = score

# normalize by L2 of the layers
l2_norms = scores_list.norm(dim=1).unsqueeze(-1)
scores_list = scores_list / l2_norms

all_heads = []
for layer_idx in range(tl_model.cfg.n_layers):
    for head_idx in range(tl_model.cfg.n_heads):
        all_heads.append((layer_idx, head_idx))

# sort both lists by scores
sorted_indices = sorted(all_heads, key=lambda x: scores_list[x], reverse=True)

#%%

# reload in a TLModel that is more memory hungry 
# but does not use backwards pass things

del tl_model
tl_model = model_getter(device=DEVICE, sixteen_heads=False)

#%%

with open(__file__, "r") as f:
    notes = f.read()
exp = TLACDCExperiment(
    model=tl_model,
    threshold=100_000,
    using_wandb=True, # for now
    wandb_entity_name=WANDB_ENTITY_NAME,
    wandb_project_name=WANDB_PROJECT_NAME,
    wandb_run_name=WANDB_RUN_NAME,
    wandb_group_name=WANDB_GROUP_NAME,
    wandb_notes=notes,
    zero_ablation=ZERO_ABLATION,
    ds=toks_int_values,
    ref_ds=toks_int_values_other,
    metric=metric,
    second_metric=None,
    verbose=True,
    hook_verbose=False,
    add_sender_hooks=True,
    add_receiver_hooks=False,
    remove_redundant=False,
)
exp.setup_model_hooks( # so we have complete control over connections
    add_sender_hooks=True,
    add_receiver_hooks=True,
    doing_acdc_runs=False,
)

# %%

max_edges = exp.count_no_edges()

# remove all connection except the MLP connections
includes_attention = [ # substrings of hook names that imply they're relatred to attention
    "attn",
    "hook_q",
    "hook_k",
    "hook_v",
]

for receiver_name in exp.corr.edges:
    for receiver_index in exp.corr.edges[receiver_name]:
        for sender_name in exp.corr.edges[receiver_name][receiver_index]:
            for receiever_index in exp.corr.edges[receiver_name][receiver_index][sender_name]:
                related_to_attention = False

                for substr in includes_attention:
                    if substr in sender_name or substr in receiver_name:
                        related_to_attention = True
                        break

                if not related_to_attention:
                    continue

                edge = exp.corr.edges[receiver_name][receiver_index][sender_name][receiever_index]
                edge.present = False

# %%

edges_to_metric = {}

for nodes_present in tqdm(range(len(sorted_indices) + 1)):

    # measure performance
    clean_logits = tl_model(toks_int_values)
    metric_result = metric(clean_logits).mean()
    cur_edges = exp.count_no_edges()
    edges_to_metric[cur_edges] = metric_result.item()

    # then add next node
    layer_idx, head_idx = sorted_indices[nodes_present]

    for (tupl, edge) in exp.corr.all_edges().items():
        for hook_name, hook_idx in [(tupl[0], tupl[1]), (tupl[2], tupl[3])]:
            if hook_name.startswith(f"blocks.{layer_idx}") and len(hook_idx.hashable_tuple) >= 3 and int(hook_idx.hashable_tuple[2]) == head_idx:
                edge.present = True

    wandb.log(
        {
            "layer_idx": layer_idx,
            "head_idx": head_idx,
            "nodes": nodes_present,
            "metric": metric_result.item(),
            "edges": cur_edges,
        }
    )

# %%

wandb.finish()

#%%