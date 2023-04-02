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
    shuffle_tensor,
    ct,
    TorchIndex,
    Edge,
    EdgeType,
)  # these introduce several important classes !!!

from transformer_lens.acdc.TLACDCCorrespondence import TLACDCCorrespondence
from transformer_lens.acdc.TLACDCInterpNode import TLACDCInterpNode
from transformer_lens.acdc.TLACDCExperiment import TLACDCExperiment

from collections import defaultdict, deque, OrderedDict
from transformer_lens.acdc.induction.utils import (
    kl_divergence,
    get_model,
    get_validation_data,
    get_good_induction_candidates,
    get_mask_repeat_candidates,
)
from transformer_lens.acdc.graphics import (
    build_colorscheme,
    show,
)
import argparse

#%%

parser = argparse.ArgumentParser()
parser.add_argument('--threshold', type=float, required=True, help='Value for THRESHOLD')
parser.add_argument('--first-cache-cpu', type=bool, required=False, default=True, help='Value for FIRST_CACHE_CPU')
parser.add_argument('--second-cache-cpu', type=bool, required=False, default=True, help='Value for SECOND_CACHE_CPU')
parser.add_argument('--zero-ablation', action='store_true', help='A flag without a value')
parser.add_argument('--using-wandb', action='store_true', help='A flag without a value')
parser.add_argument('--wandb-entity-name', type=str, required=False, default="remix_school-of-rock", help='Value for WANDB_ENTITY_NAME')
parser.add_argument('--wandb-project-name', type=str, required=False, default="acdc", help='Value for WANDB_PROJECT_NAME')
parser.add_argument('--wandb-run-name', type=str, required=False, default=None, help='Value for WANDB_RUN_NAME')

if IPython.get_ipython() is not None: # heheh get around this failing in notebooks
    args = parser.parse_args("--threshold 1.733333 --zero-ablation".split())
else:
    args = parser.parse_args()

FIRST_CACHE_CPU = args.first_cache_cpu
SECOND_CACHE_CPU = args.second_cache_cpu
THRESHOLD = args.threshold # only used if >= 0.0
ZERO_ABLATION = True if args.zero_ablation else False
USING_WANDB = True if args.using_wandb else False
WANDB_ENTITY_NAME = args.wandb_entity_name
WANDB_PROJECT_NAME = args.wandb_project_name
WANDB_RUN_NAME = args.wandb_run_name

#%% [markdown]
# Model

tl_model = get_model()

# %% [markdown]
# Data

NUM_EXAMPLES = 40
SEQ_LEN = 300
validation_data = get_validation_data()
mask_repeat_candidates = get_mask_repeat_candidates(NUM_EXAMPLES, SEQ_LEN)
toks_int_values = validation_data[:NUM_EXAMPLES, :SEQ_LEN].to(device).long()
toks_int_values_other = (
    shuffle_tensor(validation_data[:NUM_EXAMPLES, :SEQ_LEN]).to(device).long()
)

labels = validation_data[:NUM_EXAMPLES, 1 : SEQ_LEN + 1].to(device).long()

#%%

base_model_logits = tl_model(toks_int_values)
base_model_probs = F.softmax(base_model_logits, dim=-1)

#%%

raw_metric = partial(kl_divergence, base_model_probs=base_model_probs, mask_repeat_candidates=mask_repeat_candidates)
metric = partial(kl_divergence, base_model_probs=base_model_probs, mask_repeat_candidates=mask_repeat_candidates)

#%%

with open(__file__, "r") as f:
    notes = f.read()

tl_model.global_cache.clear()
tl_model.reset_hooks()

if WANDB_RUN_NAME is None:
    WANDB_RUN_NAME = f"{ct()}_{THRESHOLD}{'_zero' if ZERO_ABLATION else ''}_reversed"
else:
    assert False # I want named runs, always

exp = TLACDCExperiment(
    model=tl_model,
    threshold=THRESHOLD,
    using_wandb=USING_WANDB,
    wandb_entity_name=WANDB_ENTITY_NAME,
    wandb_project_name=WANDB_PROJECT_NAME,
    wandb_run_name=WANDB_RUN_NAME,
    wandb_notes=notes,
    zero_ablation=ZERO_ABLATION,
    ds=toks_int_values,
    ref_ds=toks_int_values_other,
    metric=metric,
    verbose=True,
)

#%%

while exp.current_node is not None:
    exp.step()

# %%

show(
    correspondence,
    "arthur_spice.png"
)
# %%
