# %% [markdown]
# This notebook / script shows several use cases of ACDC
# 
# (The code relies on our modification of the TransformerLens codebase, 
# mainly giving all HookPoints access to a global cache)

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
from pathlib import Path

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
from acdc.HookedTransformer import (
    HookedTransformer,
)
from acdc.tracr.utils import get_tracr_data, get_tracr_model_input_and_tl_model
from acdc.docstring.utils import get_all_docstring_things
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
    get_ioi_gpt2_small,
)
from acdc.induction.utils import (
    get_all_induction_things,
    get_model,
    get_validation_data,
    get_good_induction_candidates,
    get_mask_repeat_candidates,
)
from acdc.graphics import (
    build_colorscheme,
    show,
)
import argparse
torch.autograd.set_grad_enabled(False)

#%%

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, required=True)
parser.add_argument('--threshold', type=float, required=True, help='Value for THRESHOLD')
parser.add_argument('--first-cache-cpu', type=bool, required=False, default=True, help='Value for FIRST_CACHE_CPU')
parser.add_argument('--second-cache-cpu', type=bool, required=False, default=True, help='Value for SECOND_CACHE_CPU')
parser.add_argument('--zero-ablation', action='store_true', help='A flag without a value')
parser.add_argument('--using-wandb', action='store_true', help='A flag without a value')
parser.add_argument('--wandb-entity-name', type=str, required=False, default="remix_school-of-rock", help='Value for WANDB_ENTITY_NAME')
parser.add_argument('--wandb-group-name', type=str, required=False, default="default", help='Value for WANDB_GROUP_NAME')
parser.add_argument('--wandb-project-name', type=str, required=False, default="acdc", help='Value for WANDB_PROJECT_NAME')
parser.add_argument('--wandb-run-name', type=str, required=False, default=None, help='Value for WANDB_RUN_NAME')
parser.add_argument('--indices-mode', type=str, default="normal")
parser.add_argument('--names-mode', type=str, default="normal")
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--reset-network', type=int, default=0, help="Whether to reset the network we're operating on before running interp on it")

# for now, force the args to be the same as the ones in the notebook, later make this a CLI tool
if False or IPython.get_ipython() is not None: # heheh get around this failing in notebooks
    # args = parser.parse_args("--threshold 1.733333 --zero-ablation".split())
    # args = parser.parse_args("--threshold 0.001 --using-wandb".split())
    args = parser.parse_args("--task docstring --using-wandb --threshold 0.075".split()) # TODO figure out why this is such high edge count...
else:
    args = parser.parse_args()

TASK = args.task
FIRST_CACHE_CPU = args.first_cache_cpu
SECOND_CACHE_CPU = args.second_cache_cpu
THRESHOLD = args.threshold # only used if >= 0.0
ZERO_ABLATION = True if args.zero_ablation else False
USING_WANDB = True if args.using_wandb else False
WANDB_ENTITY_NAME = args.wandb_entity_name
WANDB_PROJECT_NAME = args.wandb_project_name
WANDB_RUN_NAME = args.wandb_run_name
WANDB_GROUP_NAME = args.wandb_group_name
INDICES_MODE = args.indices_mode
NAMES_MODE = args.names_mode
DEVICE = args.device
RESET_NETWORK = args.reset_network

#%% [markdown]
# Setup

second_metric = None # some tasks only have one metric

if TASK == "ioi":
    num_examples = 100
    tl_model = get_ioi_gpt2_small()
    toks_int_values, toks_int_values_other, metric = get_ioi_data(tl_model, num_examples)

elif TASK in ["tracr-reverse", "tracr-proportion"]: # do tracr
    tracr_task = TASK.split("-")[-1] # "reverse"
   
    # this implementation doesn't ablate the position embeddings (which the plots in the paper do do), so results are different. See the rust_circuit implemntation if this need be checked
    # also there's no splitting by neuron yet TODO
   
    _, tl_model = get_tracr_model_input_and_tl_model(task=TASK)
    toks_int_values, toks_int_values_other, metric = get_tracr_data(tl_model, task=TASK)

elif TASK == "induction":
    num_examples = 50
    seq_len = 300
    # TODO initialize the `tl_model` with the right model
    induction_things = get_all_induction_things(num_examples=num_examples, seq_len=seq_len, device=DEVICE)
    tl_model, toks_int_values, toks_int_values_other = induction_things.tl_model, induction_things.validation_data, induction_things.validation_patch_data

    validation_metric = induction_things.validation_metric
    metric = lambda x: validation_metric(x).item()

    test_metric_fn = induction_things.test_metric
    test_metric_data = induction_things.test_data
    test_metric = lambda model: test_metric_fn(model(test_metric_data)).item()

elif TASK == "docstring":
    num_examples = 50
    seq_len = 41
    tl_model, toks_int_values, toks_int_values_other, metric, second_metric = get_all_docstring_things(num_examples=num_examples, seq_len=seq_len, device=DEVICE, metric_name="docstring_metric", correct_incorrect_wandb=True)

else:
    raise ValueError(f"Unknown task {TASK}")

if RESET_NETWORK:
    base_dir = Path(__file__).parent.parent / "subnetwork-probing/" / "data" / "induction"
    reset_state_dict = torch.load(base_dir / "random_model.pt")
    for layer_i in range(2):
        for qkv in ["q", "k", "v"]:
            # Delete subnetwork probing masks
            del reset_state_dict[f"blocks.{layer_i}.attn.hook_{qkv}.mask_scores"]

    tl_model.load_state_dict(reset_state_dict, strict=True)
    del reset_state_dict

#%%

with open(__file__, "r") as f:
    notes = f.read()

tl_model.global_cache.clear()
tl_model.reset_hooks()

if WANDB_RUN_NAME is None or IPython.get_ipython() is not None:
    WANDB_RUN_NAME = f"{ct()}{'_randomindices' if INDICES_MODE=='random' else ''}_{THRESHOLD}{'_zero' if ZERO_ABLATION else ''}"
else:
    assert False # I want named runs, always

exp = TLACDCExperiment(
    model=tl_model,
    threshold=THRESHOLD,
    using_wandb=USING_WANDB,
    wandb_entity_name=WANDB_ENTITY_NAME,
    wandb_project_name=WANDB_PROJECT_NAME,
    wandb_run_name=WANDB_RUN_NAME,
    wandb_group_name=WANDB_GROUP_NAME,
    wandb_notes=notes,
    zero_ablation=ZERO_ABLATION,
    ds=toks_int_values,
    ref_ds=toks_int_values_other,
    metric=metric,
    second_metric=second_metric,
    verbose=True,
    indices_mode=INDICES_MODE,
    names_mode=NAMES_MODE,
    second_cache_cpu=SECOND_CACHE_CPU,
    first_cache_cpu=FIRST_CACHE_CPU,
    add_sender_hooks=False, # attempting to be efficient...
    add_receiver_hooks=False,
)

# %%

if False: # Stefan snippet
    print("KL div:", exp.metric(exp.model(exp.ds)), "no_edges", exp.count_no_edges())

    receiver_name = "blocks.0.hook_q_input"
    receiver_index = TorchIndex([None, None, 3])
    receiver_node = exp.corr.graph[receiver_name][receiver_index]

    sender_name = "blocks.0.hook_resid_pre"
    sender_index = TorchIndex([None])
    sender_node = exp.corr.graph[sender_name][sender_index]

    exp.add_sender_hook(sender_node)
    exp.add_receiver_hook(receiver_node)

    exp.corr.edges[receiver_name][receiver_index][sender_name][sender_index].present = False

    print("KL div:", exp.metric(exp.model(exp.ds)), "no_edges", exp.count_no_edges())

#%%

for i in range(100_000):
    exp.step()
    show(
        exp.corr,
        f"ims/img_new_{i+1}.png",
        show_full_index=False, # hopefully works
    )
    print(i, "-" * 50)
    print(exp.count_no_edges())

    if i==0:
        exp.save_edges("edges.pkl")

    if TASK == "induction":
        with torch.no_grad():
            test_metric_value = test_metric(exp.model)
            print("test metric:", test_metric_value)
        if USING_WANDB:
            wandb.log(dict(test_specific_metric=test_metric_value))

    if exp.current_node is None:
        break

exp.save_edges("another_final_edges.pkl")

#%%

print("Ful circ metric:", exp.metric(exp.model(exp.ds)), f"#edges={exp.count_no_edges()}")

# Indices to save writing lots
COL = TorchIndex([None])
H0 = TorchIndex([None, None, 0])
H4 = TorchIndex([None, None, 4])
H5 = TorchIndex([None, None, 5])
H6 = TorchIndex([None, None, 6])

def remove_edge(receiver_name, receiver_index, sender_name, sender_index):
    sender_node = exp.corr.graph[sender_name][sender_index]
    receiver_node = exp.corr.graph[receiver_name][receiver_index]
    edge = exp.corr.edges[receiver_name][receiver_index][sender_name][sender_index]
    edge_type_print = "ADDITION" if edge.edge_type.value == EdgeType.ADDITION.value else "PLACEHOLDER" if edge.edge_type.value == EdgeType.PLACEHOLDER.value else "DIRECT_COMPUTATION" if edge.edge_type.value == EdgeType.DIRECT_COMPUTATION.value else "UNKNOWN"
    print(f"Removing edge {receiver_name} {receiver_index} <- {sender_name} {sender_index} with type {edge_type_print}")
    if edge.edge_type.value == EdgeType.DIRECT_COMPUTATION.value:
        exp.add_receiver_hook(receiver_node)
    if edge.edge_type.value == EdgeType.ADDITION.value:
        exp.add_sender_hook(sender_node)
        exp.add_receiver_hook(receiver_node)
    if edge.edge_type.value == EdgeType.PLACEHOLDER.value:
        pass
    edge.present = False

#%%

if False:
    edges_to_keep = []
    for L3H in [H0, H6]:
        edges_to_keep.append(("blocks.3.hook_resid_post", COL, "blocks.3.attn.hook_result", L3H))
        edges_to_keep.append(("blocks.3.attn.hook_q", L3H, "blocks.3.hook_q_input", L3H))
        edges_to_keep.append(("blocks.3.hook_q_input", L3H, "blocks.1.attn.hook_result", H4))
        edges_to_keep.append(("blocks.1.attn.hook_v", H4, "blocks.1.hook_v_input", H4))
        edges_to_keep.append(("blocks.1.hook_v_input", H4, "blocks.0.hook_resid_pre", COL))
        edges_to_keep.append(("blocks.1.hook_v_input", H4, "blocks.0.attn.hook_result", H5))
        edges_to_keep.append(("blocks.0.attn.hook_v", H5, "blocks.0.hook_v_input", H5))
        edges_to_keep.append(("blocks.0.hook_v_input", H5, "blocks.0.hook_resid_pre", COL))
        edges_to_keep.append(("blocks.3.attn.hook_v", L3H, "blocks.3.hook_v_input", L3H))
        edges_to_keep.append(("blocks.3.hook_v_input", L3H, "blocks.0.hook_resid_pre", COL))
        edges_to_keep.append(("blocks.3.hook_v_input", L3H, "blocks.0.attn.hook_result", H5))
        edges_to_keep.append(("blocks.0.attn.hook_v", H5, "blocks.0.hook_v_input", H5))
        edges_to_keep.append(("blocks.0.hook_v_input", H5, "blocks.0.hook_resid_pre", COL))
        edges_to_keep.append(("blocks.3.attn.hook_k", L3H, "blocks.3.hook_k_input", L3H))
        edges_to_keep.append(("blocks.3.hook_k_input", L3H, "blocks.2.attn.hook_result", H0))
        edges_to_keep.append(("blocks.2.attn.hook_q", H0, "blocks.2.hook_q_input", H0))
        edges_to_keep.append(("blocks.2.hook_q_input", H0, "blocks.0.hook_resid_pre", COL))
        edges_to_keep.append(("blocks.2.hook_q_input", H0, "blocks.0.attn.hook_result", H5))
        edges_to_keep.append(("blocks.0.attn.hook_v", H5, "blocks.0.hook_v_input", H5))
        edges_to_keep.append(("blocks.0.hook_v_input", H5, "blocks.0.hook_resid_pre", COL))
        edges_to_keep.append(("blocks.2.attn.hook_v", H0, "blocks.2.hook_v_input", H0))
        edges_to_keep.append(("blocks.2.hook_v_input", H0, "blocks.1.attn.hook_result", H4))
        edges_to_keep.append(("blocks.1.attn.hook_v", H4, "blocks.1.hook_v_input", H4))
        edges_to_keep.append(("blocks.1.hook_v_input", H4, "blocks.0.hook_resid_pre", COL))


#%%

if True:
    import pickle
    with open("another_final_edges.pkl", "rb") as f:  # Use "rb" instead of "r"
        final_edges = pickle.load(f)
    edges_to_keep = []
    for e in final_edges:
        edges_to_keep.append(e[0])

# %%

exp.model.reset_hooks() # essential, I would guess
exp.setup_second_cache()

#... then recall 

for t in exp.corr.all_edges():
    if t not in edges_to_keep:
        remove_edge(t[0], t[1], t[2], t[3])
    else:
        edge = exp.corr.edges[t[0]][t[1]][t[2]][t[3]]
        edge.effect_size = 1
        print("Keeping", t)

print("Docstring circuit metric:", exp.metric(exp.model(exp.ds)), f"#edges={ exp.count_no_edges()}")

# %%
