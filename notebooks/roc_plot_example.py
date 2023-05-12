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

#%% [markdown]
# Setup

num_examples = 50
seq_len = 41
tl_model, toks_int_values, toks_int_values_other, metric, second_metric = get_all_docstring_things(num_examples=num_examples, seq_len=seq_len, device="cuda", metric_name="kl_divergence", correct_incorrect_wandb=False)

#%%

tl_model.global_cache.clear()

tl_model.reset_hooks()
exp = TLACDCExperiment(
    model=tl_model,
    threshold=100_000,
    using_wandb=False,
    zero_ablation=False,
    ds=toks_int_values,
    ref_ds=toks_int_values_other,
    metric=metric,
    second_metric=second_metric,
    verbose=True,
)

#%% [markdown]
# <h2> Arthur plays about with loading in graphs </h2>
# <h3> Not relevant for doing ACDC runs </h3>
# <p> Get Adria's docstring runs! </p>

from acdc.acdc_utils import false_positive_rate, false_negative_rate, true_positive_rate
api = wandb.Api()
runs = api.runs("remix_school-of-rock/acdc")
filtered_runs = []

for run in tqdm(runs):
    if "agarriga-docstring" in run.name:
        filtered_runs.append(run)

assert len(filtered_runs) > 150, len(filtered_runs)

#%% [markdown]
# <h2> The output log seems malformatted in about 6 cases. I don't know why and it seems OK to just skip these </h2> 

cnt = 0
corrs = []

for run in tqdm(filtered_runs):
    run_id = run.id
    print(len(corrs), "len_corr")
    try:
        exp.load_from_wandb_run("remix_school-of-rock", "acdc", run_id)
        print(exp.count_no_edges())
        corrs.append(deepcopy(exp.corr))
    except Exception as e:
        print(e)
        cnt+=1
        continue

# %%

edges_to_keep = []
COL = TorchIndex([None])
H0 = TorchIndex([None, None, 0])
H4 = TorchIndex([None, None, 4])
H5 = TorchIndex([None, None, 5])
H6 = TorchIndex([None, None, 6])
H = lambda i: TorchIndex([None, None, i])
for L3H in [H0, H6]:
    edges_to_keep.append(("blocks.3.hook_resid_post", COL, "blocks.3.attn.hook_result", L3H))
    edges_to_keep.append(("blocks.3.attn.hook_q", L3H, "blocks.3.hook_q_input", L3H))
    edges_to_keep.append(("blocks.3.hook_q_input", L3H, "blocks.1.attn.hook_result", H4))
    edges_to_keep.append(("blocks.3.attn.hook_v", L3H, "blocks.3.hook_v_input", L3H))
    edges_to_keep.append(("blocks.3.hook_v_input", L3H, "blocks.0.hook_resid_pre", COL))
    edges_to_keep.append(("blocks.3.hook_v_input", L3H, "blocks.0.attn.hook_result", H5))
    edges_to_keep.append(("blocks.3.attn.hook_k", L3H, "blocks.3.hook_k_input", L3H))
    edges_to_keep.append(("blocks.3.hook_k_input", L3H, "blocks.2.attn.hook_result", H0))
edges_to_keep.append(("blocks.2.attn.hook_q", H0, "blocks.2.hook_q_input", H0))
edges_to_keep.append(("blocks.2.hook_q_input", H0, "blocks.0.hook_resid_pre", COL))
edges_to_keep.append(("blocks.2.hook_q_input", H0, "blocks.0.attn.hook_result", H5))
edges_to_keep.append(("blocks.2.attn.hook_v", H0, "blocks.2.hook_v_input", H0))
edges_to_keep.append(("blocks.2.hook_v_input", H0, "blocks.1.attn.hook_result", H4))
edges_to_keep.append(("blocks.0.attn.hook_v", H5, "blocks.0.hook_v_input", H5))
edges_to_keep.append(("blocks.0.hook_v_input", H5, "blocks.0.hook_resid_pre", COL))
edges_to_keep.append(("blocks.1.attn.hook_v", H4, "blocks.1.hook_v_input", H4))
edges_to_keep.append(("blocks.1.hook_v_input", H4, "blocks.0.hook_resid_pre", COL)) 
print(len(edges_to_keep))

# format this into the dict thing... munging ugh
d = {(d[0], d[1].hashable_tuple, d[2], d[3].hashable_tuple): False for d in exp.corr.all_edges()}
for k in edges_to_keep:
    tupl = (k[0], k[1].hashable_tuple, k[2], k[3].hashable_tuple)
    assert tupl in d
    d[tupl] = True

# %%

exp2 = deepcopy(exp)

#%%

exp2.load_subgraph(d)

#%%

ground_truth = exp2.corr

# %%

false_negative_rate(exp2.corr, exp.corr)

# %%

circuit_size = exp.count_no_edges()
ground_truth_size = exp2.count_no_edges()

#%%

points = []
for corr in tqdm(corrs):
    circuit_size = corr.count_no_edges()
    if circuit_size == 0:
        continue
    points.append((false_positive_rate(ground_truth, corr)/circuit_size, true_positive_rate(ground_truth, corr)/ground_truth_size))
    print(points[-1])
    if points[-1][0] > 1:
        print(false_positive_rate(ground_truth, corr, verbose=True))
        assert False

#%%

def discard_non_pareto_optimal(points):
    ret = [(0.0, 0.0), (1.0, 1.0)]
    for x, y in points:
        for x1, y1 in points:
            if x1 <= x and y1 >= y and (x1, y1) != (x, y):
                break
        else:
            ret.append((x, y))
    return ret

processed_points = discard_non_pareto_optimal(points)
# sort by x
processed_points = sorted(processed_points, key=lambda x: x[0])

#%%

def get_roc_figure(all_points, names): # =""):
    """Points are (false positive rate, true positive rate)"""
    roc_figure = go.Figure()
    # roc_figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random (assuming binary classification...)"))
    for points, name in zip(all_points, names):
        roc_figure.add_trace(
            go.Scatter(
                x=[p[0] for p in points],
                y=[p[1] for p in points],
                mode="lines",
                line=dict(shape='hv'),  # Adding this line will make the curve stepped.
                name=name,
            )
        )
    roc_figure.update_xaxes(title_text="False positive rate")
    roc_figure.update_yaxes(title_text="True positive rate")
    return roc_figure

fig = get_roc_figure([processed_points], ["ACDC"])
fig.show()

# %%
