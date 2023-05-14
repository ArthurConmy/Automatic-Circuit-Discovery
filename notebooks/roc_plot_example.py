# %% [markdown]
# Script of ROC Plots

import IPython

if IPython.get_ipython() is not None:
    IPython.get_ipython().run_line_magic("load_ext", "autoreload")  # type: ignore
    IPython.get_ipython().run_line_magic("autoreload", "2")  # type: ignore

from copy import deepcopy
from acdc.acdc_utils import false_positive_rate, false_negative_rate, true_positive_stat
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
from acdc.docstring.utils import get_all_docstring_things, get_docstring_model, get_docstring_subgraph_true_edges
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

parser = argparse.ArgumentParser(description="Used to control ROC plot scripts (for standardisation with other files...)")
parser.add_argument('--task', type=str, required=True, choices=['ioi', 'docstring', 'induction', 'tracr', 'greaterthan'], help='Choose a task from the available options: ioi, docstring, induction, tracr (WIPs except docstring!!!)')
parser.add_argument('--zero-ablation', action='store_true', help='Use zero ablation')
parser.add_argument("--ignore-sixteen-heads", action="store_true", help="Ignore the 16 heads stuff TODO")

# for now, force the args to be the same as the ones in the notebook, later make this a CLI tool
if IPython.get_ipython() is not None: # heheh get around this failing in notebooks
    args = parser.parse_args("--task docstring".split())
else:
    args = parser.parse_args()

TASK = args.task
ZERO_ABLATION = True if args.zero_ablation else False
SKIP_ACDC=False
SKIP_SP = False
SKIP_SIXTEEN_HEADS = True if args.ignore_sixteen_heads else False

#%% [markdown]
# Setup

if TASK == "docstring":
    num_examples = 50
    seq_len = 41
    all_docstring_things = get_all_docstring_things(num_examples=num_examples, seq_len=seq_len, device="cuda", metric_name="kl_div", correct_incorrect_wandb=False)
    toks_int_values = all_docstring_things.validation_data
    toks_int_values_other = all_docstring_things.validation_patch_data
    tl_model = all_docstring_things.tl_model
    metric = all_docstring_things.validation_metric
    second_metric = None
    get_true_edges = get_docstring_subgraph_true_edges

    ACDC_PROJECT_NAME = "remix_school-of-rock/acdc"
    ACDC_RUN_FILTER = lambda name: name.startswith("agarriga-docstring")

    SP_PROJECT_NAME = "remix_school-of-rock/induction-sp-replicate"
    def sp_run_filter(name):
        if not name.startswith("agarriga-sp-"): return False
        try: 
            return int(name.split("-")[-1])
        except:
            return False
        return 0 <= int(name.split("-")[-1]) <= 319
    SP_RUN_FILTER = lambda name: sp_run_filter(name)

else:
    raise NotImplementedError("TODO " + TASK)

#%% [markdown]
# Setup the experiment for wrapping functionality nicely

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
# Load the *canonical* circuit

d = {(d[0], d[1].hashable_tuple, d[2], d[3].hashable_tuple): False for d in exp.corr.all_edges()}
d_trues = get_true_edges()
for k in d_trues:
    d[k] = True
exp.load_subgraph(d)
canonical_circuit_subgraph = deepcopy(exp.corr)

#%% [markdown]
# <h2> Arthur plays about with loading in graphs </h2>
# <h3> Not relevant for doing ACDC runs </h3>
# <p> Get Adria's docstring runs! </p>

def get_acdc_runs(
    experiment,
    project_name: str = ACDC_PROJECT_NAME,
    run_name_filter: Callable[[str], bool] = ACDC_RUN_FILTER,
):
    api = wandb.Api()
    runs = api.runs(project_name)
    filtered_runs = []
    for run in tqdm(runs):
        if run_name_filter(run.name):
            filtered_runs.append(run)
    cnt = 0
    corrs = []
    args = project_name.split("/")
    for run in tqdm(filtered_runs):
        run_id = run.id
        try:
            experiment.load_from_wandb_run(*args, run_id)
            corrs.append(deepcopy(exp.corr))
        except Exception as e:
            print(e)
            cnt+=1
            continue
    return corrs

if "corrs" not in locals(): # this is slow, so run once
    corrs = get_acdc_runs(exp)

#%%

# do SP stuff

def get_sp_runs(
    experiment, 
    project_name: str = SP_PROJECT_NAME,
    run_name_filter: Callable[[str], bool] = SP_RUN_FILTER,
):
    api = wandb.Api()
    runs = api.runs(project_name)
    filtered_runs = []
    for run in tqdm(runs):
        if run_name_filter(run.name):
            filtered_runs.append(run)
    cnt = 0
    corrs = []

    args = project_name.split("/")

    last_plotly = None

    for run in tqdm(filtered_runs):
        for f in run.files():
            if "media/plotly" not in str(f):
                continue
            if last_plotly is None or int(f.name.split("_")[2]) > int(last_plotly.name.split("_")[2]):
                last_plotly = f

    last_plotly.download(replace=True, root="/tmp/")
    with open("/tmp/" + last_plotly.name, "r") as f:
        plotly = json.load(f)
    return plotly

plotlies = get_sp_runs(exp)

#%%

# reload in the circuit
exp.load_subgraph(d)
ground_truth = exp.corr
ground_truth_size = ground_truth.count_no_edges()

#%%

points = {}
if not SKIP_ACDC: points["ACDC"] = []    
if not SKIP_SP: points["SP"] = []
if not SKIP_SIXTEEN_HEADS: points["16H"] = []

#%%

# this is for ACDC
for corr in tqdm(corrs): 
    circuit_size = corr.count_no_edges()
    if circuit_size == 0:
        continue
    points["ACDC"].append((false_positive_rate(ground_truth, corr)/circuit_size, true_positive_stat(ground_truth, corr)/ground_truth_size))
    print(points["ACDC"][-1])
    if points["ACDC"][-1][0] > 1:
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

processed_points = {key: discard_non_pareto_optimal(points[key]) for key in points}

# sort by x
processed_points = {k: sorted(processed_points[k], key=lambda x: x[0]) for k in processed_points}

#%%

def get_roc_figure(all_points, names):
    """Points are (false positive rate, true positive rate)"""
    roc_figure = go.Figure()
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

fig = get_roc_figure(list(processed_points.values()), list(processed_points.keys()))
fig.show()

# %%
