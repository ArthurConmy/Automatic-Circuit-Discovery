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
device = "cuda" if torch.cuda.is_available() else "cpu" # TODO check CPU support
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens.HookedTransformer import (
    HookedTransformer,
)
from transformer_lens.acdc.utils import (
    make_nd_dict,
    ct,
    TLACDCInterpNode,
    TLACDCCorrespondence,
    TLACDCExperiment,
    TorchIndex,
    Edge,
    EdgeType,
)  # these introduce several important classes !!!
from collections import defaultdict, deque, OrderedDict
from transformer_lens.acdc.induction.utils import (
    kl_divergence,
)
from transformer_lens.acdc.graphics import (
    build_colorscheme,
    show,
)
import argparse

#%%

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=False, help="Path to YAML config file", default="../../configs_acdc/base_config.yaml")
parser.add_argument('--first-cache-cpu', type=bool, required=False, default=True, help='Value for FIRST_CACHE_CPU')
parser.add_argument('--second-cache-cpu', type=bool, required=False, default=True, help='Value for SECOND_CACHE_CPU') # TODO move these to the config file. ... or do YAML overrides
parser.add_argument('--threshold', type=float, required=False, default=-1.0, help='Value for THRESHOLD') # defaults to fake value
parser.add_argument('--zero-ablation', action='store_true', help='A flag without a value')

if IPython.get_ipython() is not None: # heheh get around this failing in notebooks
    args = parser.parse_args("--config ../../configs_acdc/base_config.yaml --zero-ablation".split())
else:
    args = parser.parse_args()

FIRST_CACHE_CPU = args.first_cache_cpu
SECOND_CACHE_CPU = args.second_cache_cpu
THRESHOLD = args.threshold # only used if >= 0.0

with open(args.config, 'r') as yaml_file:
    yaml_config = yaml.safe_load(yaml_file)

USING_WANDB = yaml_config['USING_WANDB']

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

# %%

SEQ_LEN = 300
NUM_EXAMPLES = 40

# get the dataset from HF
validation_fname = huggingface_hub.hf_hub_download(
    repo_id="ArthurConmy/redwood_attn_2l", filename="validation_data.pt"
)
validation_data = torch.load(validation_fname)

good_induction_candidates_fname = huggingface_hub.hf_hub_download(
    repo_id="ArthurConmy/redwood_attn_2l", filename="good_induction_candidates.pt"
)
good_induction_candidates = torch.load(good_induction_candidates_fname)

mask_repeat_candidates_fname = huggingface_hub.hf_hub_download(
    repo_id="ArthurConmy/redwood_attn_2l", filename="mask_repeat_candidates.pkl"
)
mask_repeat_candidates = torch.load(mask_repeat_candidates_fname)
mask_repeat_candidates.requires_grad = False
mask_repeat_candidates = mask_repeat_candidates[:NUM_EXAMPLES, :SEQ_LEN]


def shuffle_tensor(tens):
    """Shuffle tensor along first dimension"""
    torch.random.manual_seed(42)
    return tens[torch.randperm(tens.shape[0])]


toks_int_values = validation_data[:NUM_EXAMPLES, :SEQ_LEN].to(device).long()
toks_int_values_other = (
    shuffle_tensor(validation_data[:NUM_EXAMPLES, :SEQ_LEN]).to(device).long()
)
good_induction_candidates = mask_repeat_candidates[:NUM_EXAMPLES, :SEQ_LEN].to(device)
labels = validation_data[:NUM_EXAMPLES, 1 : SEQ_LEN + 1].to(device).long()

#%%

base_model_logits = tl_model(toks_int_values)
base_model_probs = F.softmax(base_model_logits, dim=-1)

#%%

raw_metric = partial(kl_divergence, base_model_probs=base_model_probs, mask_repeat_candidates=mask_repeat_candidates)
metric = partial(kl_divergence, base_model_probs=base_model_probs, mask_repeat_candidates=mask_repeat_candidates)

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

# %% [markdown]
# build the graph

correspondence = TLACDCCorrespondence()

downstream_residual_nodes: List[TLACDCInterpNode] = []
logits_node = TLACDCInterpNode(
    name=f"blocks.{tl_model.cfg.n_layers-1}.hook_resid_post",
    index=TorchIndex([None]),
)
downstream_residual_nodes.append(logits_node)
new_downstream_residual_nodes: List[TLACDCInterpNode] = []

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
        )
        for residual_stream_node in downstream_residual_nodes:
            correspondence.add_edge(
                parent_node=cur_head,
                child_node=residual_stream_node,
                edge=Edge(edge_type=EdgeType.ADDITION),
            )

        # TODO maybe this needs be moved out of this block??? IDK
        hook_letter_inputs = {}
        for letter in "qkv":
            hook_letter_name = f"blocks.{layer_idx}.attn.hook_{letter}"
            hook_letter_slice = TorchIndex([None, None, head_idx])
            hook_letter_node = TLACDCInterpNode(name=hook_letter_name, index=hook_letter_slice)
            
            hook_letter_input_name = f"blocks.{layer_idx}.hook_{letter}_input"
            hook_letter_input_slice = TorchIndex([None, None, head_idx])
            hook_letter_input_node = TLACDCInterpNode(
                name=hook_letter_input_name, index=hook_letter_input_slice,
            )

            correspondence.add_edge(
                parent_node=hook_letter_input_node,
                child_node=hook_letter_node,
                edge=Edge(edge_type=EdgeType.DIRECT_COMPUTATION),
            )

            new_downstream_residual_nodes.append(hook_letter_input_node)
    downstream_residual_nodes.extend(new_downstream_residual_nodes)

# add the embedding node

embedding_node = TLACDCInterpNode(
    name="blocks.0.hook_resid_pre",
    index=TorchIndex([None]),
)
for node in downstream_residual_nodes:
    correspondence.add_edge(
        parent_node=embedding_node,
        child_node=node,
        edge=Edge(edge_type=EdgeType.ADDITION),
    )
#%%

with open(__file__, "r") as f:
    notes = f.read()

config_overrides = {}
if THRESHOLD >= 0.0: # we actually added this
    config_overrides["THRESHOLD"] = THRESHOLD

config_overrides["WANDB_RUN_NAME"] = f"{ct()}_{THRESHOLD if THRESHOLD >= 0.0 else yaml_config['THRESHOLD']}"

for key, value in config_overrides.items():
    yaml_config[key] = value

tl_model.global_cache.clear()
tl_model.reset_hooks()
exp = TLACDCExperiment(
    model=tl_model,
    ds=toks_int_values,
    ref_ds=toks_int_values_other,
    corr=correspondence,
    metric=metric,
    verbose=True,
    wandb_notes=notes,
    config=yaml_config,
)

#%%

while exp.current_node is not None:
    exp.step()

# %%
