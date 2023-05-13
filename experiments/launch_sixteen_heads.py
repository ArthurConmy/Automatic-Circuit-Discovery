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

# TODO what we need
# similar to main.py setup of metrics
# but these need be adapted so that we have KLs for each datapoint
# then some looping
# probably a WANDB run is easiest SAVE THE ARTIFACT

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
    args = parser.parse_args("--task ioi --wandb-run-name test_16_heads".split())
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

#%%

"""Mostly copied from acdc/main.py"""

if TASK == "ioi":
    num_examples = 100 
    tl_model = get_gpt2_small(device=DEVICE, sixteen_heads=True)
    toks_int_values, toks_int_values_other, metric = get_ioi_data(tl_model, num_examples, kl_return_one_element=False)
    assert len(toks_int_values) == len(toks_int_values_other) == num_examples, (len(toks_int_values), len(toks_int_values_other), num_examples)
    seq_len = toks_int_values.shape[1]
elif TASK == "induction":
    raise NotImplementedError("Induction has same sentences with multiple places we take loss / KL divergence; fiddlier implementation")

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
    if not tl_model.cfg.attn_only:
        keys.append((layer_idx, None)) # MLP

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

    # TODO implement MLP

    tl_model.zero_grad()
    tl_model.reset_hooks()
    del clean_cache
    del clean_logits
    import gc; gc.collect()
    torch.cuda.empty_cache()

for k in kls:
    kls[k].to("cpu")

#%%

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

        cur_mlp_result = torch.abs(
            torch.einsum(
                "bsd,bsd->b", # TODO check
                clean_cache[f"blocks.{layer_idx}.hook_mlp_out_grad"],
                clean_cache[f"blocks.{layer_idx}.hook_mlp_out"] - (0.0 if ZERO_ABLATION else corrupted_cache[f"blocks.{layer_idx}.hook_mlp_out"]),
            )
        )

        results[(layer_idx, None)][i] = cur_mlp_result[i].item()
        # tl_model = get_gpt2_small(device=DEVICE, sixteen_heads=True)

    del clean_cache
    del clean_logits
    tl_model.reset_hooks()
    tl_model.zero_grad()
    torch.cuda.empty_cache()
    import gc; gc.collect()

for k in results:
    results[k].to("cpu")

# %%
