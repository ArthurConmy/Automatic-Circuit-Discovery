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
import pickle
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
from acdc.graphics import show
from acdc.HookedTransformer import (
    HookedTransformer,
)
from acdc.tracr.utils import get_all_tracr_things, get_tracr_data, get_tracr_model_input_and_tl_model
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
    get_all_ioi_things,
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
torch.autograd.set_grad_enabled(False)

#%%

parser = argparse.ArgumentParser(description="Used to launch ACDC runs. Only task and threshold are required")
parser.add_argument('--task', type=str, required=True, choices=['ioi', 'docstring', 'induction', 'tracr', 'greaterthan'], help='Choose a task from the available options: ioi, docstring, induction, tracr (WIPs)')
parser.add_argument('--threshold', type=float, required=True, help='Value for THRESHOLD')
parser.add_argument('--first-cache-cpu', type=bool, required=False, default=True, help='Value for FIRST_CACHE_CPU')
parser.add_argument('--second-cache-cpu', type=bool, required=False, default=True, help='Value for SECOND_CACHE_CPU')
parser.add_argument('--zero-ablation', action='store_true', help='Use zero ablation')
parser.add_argument('--using-wandb', action='store_true', help='Use wandb')
parser.add_argument('--wandb-entity-name', type=str, required=False, default="remix_school-of-rock", help='Value for WANDB_ENTITY_NAME')
parser.add_argument('--wandb-group-name', type=str, required=False, default="default", help='Value for WANDB_GROUP_NAME')
parser.add_argument('--wandb-project-name', type=str, required=False, default="acdc", help='Value for WANDB_PROJECT_NAME')
parser.add_argument('--wandb-run-name', type=str, required=False, default=None, help='Value for WANDB_RUN_NAME')
parser.add_argument("--wandb-dir", type=str, default="/tmp/wandb")
parser.add_argument("--wandb-mode", type=str, default="online")
parser.add_argument('--indices-mode', type=str, default="normal")
parser.add_argument('--names-mode', type=str, default="normal")
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--reset-network', type=int, default=0, help="Whether to reset the network we're operating on before running interp on it")
parser.add_argument('--metric', type=str, default="kl_div", help="Which metric to use for the experiment")
parser.add_argument('--torch-num-threads', type=int, default=0, help="How many threads to use for torch (0=all)")
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument("--max-num-epochs",type=int, default=100_000)
parser.add_argument('--single-step', action='store_true', help='Use single step, mostly for testing')

# for now, force the args to be the same as the ones in the notebook, later make this a CLI tool
if IPython.get_ipython() is not None: # heheh get around this failing in notebooks
    # args = parser.parse_args("--threshold 1.733333 --zero-ablation".split())
    # args = parser.parse_args("--threshold 0.001 --using-wandb".split())
    args = parser.parse_args("--task docstring --using-wandb --threshold 0.005 --wandb-project-name acdc --indices-mode reverse --first-cache-cpu False --second-cache-cpu False".split()) # TODO figure out why this is such high edge count...
else:
    args = parser.parse_args()


if args.torch_num_threads > 0:
    torch.set_num_threads(args.torch_num_threads)
torch.manual_seed(args.seed)

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
SINGLE_STEP = True if args.single_step else False

#%% [markdown]
# Setup

second_metric = None # some tasks only have one metric
use_pos_embed = False

if TASK == "ioi":
    num_examples = 100
    things = get_all_ioi_things(num_examples=num_examples, device=DEVICE, metric_name=args.metric)
elif TASK == "tracr-reverse":
    num_examples = 6
    things = get_all_tracr_things(task="reverse", metric_name=args.metric, num_examples=num_examples, device=DEVICE)
elif TASK == "tracr-proportion":
    num_examples = 50
    things = get_all_tracr_things(task="proportion", metric_name=args.metric, num_examples=num_examples, device=DEVICE)
elif TASK == "induction":
    num_examples = 50
    seq_len = 300
    # TODO initialize the `tl_model` with the right model
    things = get_all_induction_things(num_examples=num_examples, seq_len=seq_len, device=DEVICE, metric=args.metric)

elif TASK == "docstring":
    num_examples = 50
    seq_len = 41
    things = get_all_docstring_things(num_examples=num_examples, seq_len=seq_len, device=DEVICE,
                                                metric_name=args.metric, correct_incorrect_wandb=True)
elif TASK == "greaterthan":
    num_examples = 100
    things = get_all_greaterthan_things(num_examples=num_examples, metric_name=args.metric, device=DEVICE)
else:
    raise ValueError(f"Unknown task {TASK}")


validation_metric = things.validation_metric
def metric(x):
    return validation_metric(x).item()

toks_int_values = things.validation_data
toks_int_values_other = things.validation_patch_data

if RESET_NETWORK:
    raise NotImplementedError("TODO")
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
    assert WANDB_RUN_NAME is not None, "I want named runs, always"

tl_model.reset_hooks()
exp = TLACDCExperiment(
    model=tl_model,
    threshold=THRESHOLD,
    using_wandb=USING_WANDB,
    wandb_entity_name=WANDB_ENTITY_NAME,
    wandb_project_name=WANDB_PROJECT_NAME,
    wandb_run_name=WANDB_RUN_NAME,
    wandb_group_name=WANDB_GROUP_NAME,
    wandb_notes=notes,
    wandb_dir=args.wandb_dir,
    wandb_mode=args.wandb_mode,
    wandb_config=args,
    zero_ablation=ZERO_ABLATION,
    ds=toks_int_values,
    ref_ds=toks_int_values_other,
    metric=metric,
    second_metric=second_metric,
    verbose=True,
    indices_mode=INDICES_MODE,
    names_mode=NAMES_MODE,
    second_cache_cpu=SECOND_CACHE_CPU,
    hook_verbose=False,
    first_cache_cpu=FIRST_CACHE_CPU,
    add_sender_hooks=True,
    use_pos_embed=use_pos_embed,
    add_receiver_hooks=False,
    remove_redundant=False,
    show_full_index=use_pos_embed,
)

# exp.load_from_wandb_run("remix_school-of-rock", "acdc", "c4bixuq5")
# # ^that line of code can scrape from a WANDB run
# print(exp.count_no_edges()) # should be 6 for that run

#%%

for i in range(args.max_num_epochs):

    exp.step() # TODO why aren't we while looping to completion ???

    show(
        exp.corr,
        f"ims/img_new_{i+1}.png",
        show_full_index=use_pos_embed,
    )
    print(i, "-" * 50)
    print(exp.count_no_edges())

    if i==0:
        exp.save_edges("edges.pkl")

    if TASK in ["docstring", "induction"]:
        with torch.no_grad():
            test_metric_values = {}
            for k, fn in test_metric_fns.items():
                test_metric_values["test_"+k] = fn(exp.model(test_metric_data)).item()
        if USING_WANDB:
            wandb.log(test_metric_values)

    if exp.current_node is None or SINGLE_STEP:
        break

exp.save_edges("another_final_edges.pkl") 

if USING_WANDB:
    edges_fname = f"edges.pth"
    exp.save_edges(edges_fname)
    artifact = wandb.Artifact(edges_fname, type='dataset')
    artifact.add_file(edges_fname)
    wandb.log_artifact(artifact)
    os.remove(edges_fname)
    wandb.finish()


#%%
