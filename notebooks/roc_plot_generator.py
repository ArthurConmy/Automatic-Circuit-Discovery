# %% [markdown]
# Script of ROC Plots!!!

# You need to define

# # for everything
# loading in the experiment correctly (toks_int_values_other etc.)
# get_true_edges # function that return the edges of the *true* circuit (see example of docstring in this code)

# and then a bunch of paths and filters for wandb runs (again see docstring example)

# # for ACDC
# ACDC_PROJECT_NAME
# ACDC_RUN_FILTER 

# # for SP # filters are more annoying since some things are nested in groups
# SP_PROJECT_NAME
# SP_PRE_RUN_FILTER 
# SP_RUN_FILTER

# # for 16 heads # sixteen heads is just one run
# SIXTEEN_HEADS_PROJECT_NAME
# SIXTEEN_HEADS_RUN

import IPython

if IPython.get_ipython() is not None:
    IPython.get_ipython().run_line_magic("load_ext", "autoreload")  # type: ignore
    IPython.get_ipython().run_line_magic("autoreload", "2")  # type: ignore

from copy import deepcopy
from subnetwork_probing.train import correspondence_from_mask
from acdc.acdc_utils import false_positive_rate, false_negative_rate, true_positive_stat
import pandas as pd
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
from acdc.munging_utils import parse_interpnode
import pickle
import wandb
import IPython
from acdc.munging_utils import heads_to_nodes_to_mask
import torch

# from easy_transformer.ioi_dataset import IOIDataset  # type: ignore
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

from acdc.hook_points import HookedRootModule, HookPoint
from acdc.HookedTransformer import (
    HookedTransformer,
)
from acdc.tracr.utils import get_tracr_model_input_and_tl_model, get_tracr_proportion_edges, get_tracr_reverse_edges, get_all_tracr_things
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
    get_ioi_true_edges,
    get_gpt2_small,
)
from acdc.induction.utils import (
    get_all_induction_things,
    get_validation_data,
    get_good_induction_candidates,
    get_mask_repeat_candidates,
)
from acdc.graphics import (
    build_colorscheme,
    show,
)
from acdc.ioi.utils import (
    get_all_ioi_things,
    get_gpt2_small,
)
import argparse
from acdc.greaterthan.utils import get_all_greaterthan_things
from pathlib import Path

from notebooks.emacs_plotly_render import set_plotly_renderer
set_plotly_renderer("emacs")


def get_col(df, col): # dumb util
    non_null_entries = list(df.loc[df[col].notnull(), col])
    return non_null_entries 

torch.autograd.set_grad_enabled(False)

#%% [markdown]

parser = argparse.ArgumentParser(description="Used to control ROC plot scripts (for standardisation with other files...)")
parser.add_argument('--task', type=str, required=True, choices=['ioi', 'docstring', 'induction', 'tracr-reverse', 'tracr-proportion', 'greaterthan'], help='Choose a task from the available options: ioi, docstring, induction, tracr (WIPs)')
parser.add_argument("--mode", type=str, required=False, choices=["edges", "nodes"], help="Choose a mode from the available options: edges, nodes", default="edges") # TODO implement nodes
parser.add_argument('--zero-ablation', action='store_true', help='Use zero ablation')
parser.add_argument('--metric', type=str, default="kl_div", help="Which metric to use for the experiment")
parser.add_argument('--reset-network', type=int, default=0, help="Whether to reset the network we're operating on before running interp on it")
parser.add_argument("--skip-sixteen-heads", action="store_true", help="Skip the 16 heads stuff")
parser.add_argument("--skip-sp", action="store_true", help="Skip the SP stuff")
parser.add_argument("--testing", action="store_true", help="Use testing data instead of validation data")

if IPython.get_ipython() is not None:
    args = parser.parse_args("--task ioi --metric=kl_div".split())
else:
    args = parser.parse_args()


TASK = args.task
METRIC = args.metric
DEVICE = "cpu"
ZERO_ABLATION = True if args.zero_ablation else False
RESET_NETWORK = 1 if args.reset_network else 0
SKIP_ACDC = False
SKIP_SP = True if args.skip_sp else False
SKIP_SIXTEEN_HEADS = True if args.skip_sixteen_heads else False
TESTING = True if args.testing else False

# defaults
ACDC_PROJECT_NAME = "remix_school-of-rock/acdc"
ACDC_PRE_RUN_FILTER = {
    "state": "finished",
    "group": "acdc-spreadsheet2",
    "config.task": TASK,
    "config.metric": METRIC,
    "config.zero_ablation": ZERO_ABLATION,
    "config.reset_network": RESET_NETWORK,
}
ACDC_RUN_FILTER = None

# # for SP # filters are more annoying since some things are nested in groups
SP_PROJECT_NAME = "remix_school-of-rock/induction-sp-replicate"
SP_PRE_RUN_FILTER = {
    "state": "finished",
    "config.task": TASK,
    "config.loss_type": METRIC,
    "config.zero_ablation": int(ZERO_ABLATION),
    "config.reset_subject": RESET_NETWORK,
}
SP_RUN_FILTER = None

# # for 16 heads it's just one run but this way we just use the same code
SIXTEEN_HEADS_PROJECT_NAME = "remix_school-of-rock/acdc"
SIXTEEN_HEADS_PRE_RUN_FILTER = {
    "state": "finished",
    "group": "sixteen-heads",
    "config.task": TASK,
    "config.metric": METRIC,
    "config.zero_ablation": ZERO_ABLATION,
    "config.reset_network": RESET_NETWORK,
}
SIXTEEN_HEADS_RUN_FILTER = None

USE_POS_EMBED = False


ROOT = Path("/tmp/artifacts_for_plot")
ROOT.mkdir(exist_ok=True)

#%% [markdown]
# Setup
# substantial copy paste from main.py, with some new configs, directories...

if TASK == "docstring":
    num_examples = 50
    seq_len = 41
    things = get_all_docstring_things(num_examples=num_examples, seq_len=seq_len, device=DEVICE,
                                                metric_name=METRIC, correct_incorrect_wandb=False)
    get_true_edges = get_docstring_subgraph_true_edges
    SP_PRE_RUN_FILTER["group"] = "docstring3"

elif TASK in ["tracr-reverse", "tracr-proportion"]: # do tracr
    tracr_task = TASK.split("-")[-1] # "reverse"/"proportion"
    if tracr_task == "proportion":
        get_true_edges = get_tracr_proportion_edges
        num_examples = 50
    elif tracr_task == "reverse":
        get_true_edges = get_tracr_reverse_edges
        num_examples = 6
    else:
        raise NotImplementedError("not a tracr task")

    things = get_all_tracr_things(task=tracr_task, metric_name=METRIC, num_examples=num_examples, device=DEVICE)

    # # for propotion, 
    # tl_model(toks_int_values[:1])[0, :, 0] 
    # is the proportion at each space (including irrelevant first position

elif TASK == "ioi":
    num_examples = 100
    things = get_all_ioi_things(num_examples=num_examples, device=DEVICE, metric_name=METRIC)

    if METRIC == "kl_div":
        ACDC_PROJECT_NAME = "remix_school-of-rock/arthur_ioi_sweep"
        del ACDC_PRE_RUN_FILTER["config.reset_network"]
        ACDC_PRE_RUN_FILTER["group"] = "default"

    get_true_edges = partial(get_ioi_true_edges, model=things.tl_model)

elif TASK == "greaterthan":
    num_examples = 100
    things = get_all_greaterthan_things(num_examples=num_examples, metric_name=METRIC, device=DEVICE)
    get_true_edges = partial(get_greaterthan_true_edges, model=tl_model)

elif TASK == "induction":
    raise ValueError("There is no ground truth circuit for Induction!!!")
else:
    raise NotImplementedError("TODO " + TASK)

#%% [markdown]
# Setup the experiment for wrapping functionality nicely

things.tl_model.global_cache.clear()
things.tl_model.reset_hooks()
exp = TLACDCExperiment(
    model=things.tl_model,
    threshold=100_000,
    early_exit=True,
    using_wandb=False,
    zero_ablation=False,
    ds=things.validation_data,
    ref_ds=things.validation_patch_data,
    metric=things.validation_metric,
    second_metric=None,
    verbose=True,
    use_pos_embed=USE_POS_EMBED,
)

max_subgraph_size = exp.corr.count_no_edges()

#%% [markdown]
# Load the *canonical* circuit

d = {(d[0], d[1].hashable_tuple, d[2], d[3].hashable_tuple): False for d in exp.corr.all_edges()}
d_trues = get_true_edges()
for k in d_trues:
    d[k] = True
exp.load_subgraph(d)
canonical_circuit_subgraph = deepcopy(exp.corr)
canonical_circuit_subgraph_size = canonical_circuit_subgraph.count_no_edges()

#%%

def get_acdc_runs(
    experiment,
    project_name: str = ACDC_PROJECT_NAME,
    pre_run_filter: dict = ACDC_PRE_RUN_FILTER,
    run_filter: Optional[Callable[[Any], bool]] = ACDC_RUN_FILTER,
    clip: Optional[int] = None,
):
    if clip is None:
        clip = 100_000 # so we don't clip anything

    api = wandb.Api()
    runs = api.runs(project_name, filters=pre_run_filter)
    if run_filter is None:
        filtered_runs = runs[:clip]
    else:
        filtered_runs = list(filter(run_filter, tqdm(runs[:clip])))
    print(f"loading {len(filtered_runs)} runs")

    corrs = []
    for run in filtered_runs:
        # Try to find `edges.pth`
        edges_artifact = None
        for art in run.logged_artifacts():
            if "edges.pth" in art.name:
                edges_artifact = art
                break

        if edges_artifact is None:
            # We'll have to parse the run
            print(f"Edges.pth not found for run {run.name}, falling back to parsing")
            try:
                log_text = run.file("output.log").download(root=ROOT, replace=False, exist_ok=True).read()
                experiment.load_from_wandb_run(log_text)
                corrs.append(deepcopy(exp.corr))
            except wandb.CommError:
                print(f"Loading run {run.name} with state={runs.state} config={run.config} totally failed.")
                continue

        else:
            all_edges = exp.corr.all_edges()
            for edge in all_edges.values():
                edge.present = False

            this_root = ROOT / edges_artifact.name
            # Load the edges
            for f in edges_artifact.files():
                with f.download(root=this_root, replace=False, exist_ok=True) as fopen:
                    # Sadly f.download opens in text mode
                    with open(fopen.name, "rb") as fopenb:
                        edges_pth = pickle.load(fopenb)

            for t, _effect_size in edges_pth:
                all_edges[t].present = True

            corrs.append((deepcopy(exp.corr), run.config["threshold"]))
        print(f"Added run with threshold={run.config['threshold']}")
    return corrs

if not SKIP_ACDC: # this is slow, so run once
    acdc_corrs = get_acdc_runs(exp, clip = 1 if TESTING else None)

#%%

# Do SP stuff
def get_sp_corrs(
    experiment, 
    model = things.tl_model,
    project_name: str = SP_PROJECT_NAME,
    pre_run_filter: dict = SP_PRE_RUN_FILTER,
    run_filter: Optional[Callable[[Any], bool]] = SP_RUN_FILTER,
    clip: Optional[int] = None,
):
    if clip is None:
        clip = 100_000 # so we don't clip anything

    api = wandb.Api()
    runs = api.runs(project_name, filters=pre_run_filter)
    if run_filter is None:
        filtered_runs = runs[:clip]
    else:
        filtered_runs = list(filter(run_filter, tqdm(runs[:clip])))
    print(f"loading {len(filtered_runs)} runs")

    corrs = []
    for run in filtered_runs:
        nodes_to_mask_strings = run.summary["nodes_to_mask"]
        nodes_to_mask = [parse_interpnode(s) for s in nodes_to_mask_strings]
        corr = correspondence_from_mask(
            model = model,
            nodes_to_mask=nodes_to_mask,
        )
        corrs.append((corr, run.config["lambda_reg"]))
    return corrs

if not SKIP_SP: # this is slow, so run once
    sp_corrs = get_sp_corrs(exp, clip = 1 if TESTING else None) # clip for testing

#%%

def get_sixteen_heads_corrs(
    experiment = exp,
    project_name = SIXTEEN_HEADS_PROJECT_NAME,
    pre_run_filter = SIXTEEN_HEADS_PRE_RUN_FILTER,
    run_filter = SIXTEEN_HEADS_RUN_FILTER,
    model= things.tl_model,
):
    api = wandb.Api()
    runs = api.runs(project_name, filters=pre_run_filter)
    if run_filter is None:
        run = runs[0]
    else:
        run = None
        for r in runs:
            if run_filter(r):
                run = r
                break
        assert run is not None

    nodes_names_indices = run.summary["nodes_names_indices"]
    print(nodes_names_indices)

    corrs = []
    nodes_to_mask = []
    cum_score = 0.0
    for nodes, hook_name, idx, score in tqdm(nodes_names_indices[::3]):
        nodes_to_mask += list(map(parse_interpnode, nodes))
        corr = correspondence_from_mask(model=model, nodes_to_mask=nodes_to_mask, use_pos_embed=exp.use_pos_embed)
        cum_score += score
        corrs.append((corr, cum_score))
    return corrs

if "sixteen_heads_corrs" not in locals() and not SKIP_SIXTEEN_HEADS: # this is slow, so run once
    sixteen_heads_corrs = get_sixteen_heads_corrs()

#%%

methods = []

if not SKIP_ACDC: methods.append("ACDC") 
if not SKIP_SP: methods.append("SP")
if not SKIP_SIXTEEN_HEADS: methods.append("16H")

#%%

# get points from correspondence
def get_points(corrs_and_scores):
    points = []
    for corr, score in tqdm(sorted(corrs_and_scores, key=lambda x: x[1])):
        circuit_size = corr.count_no_edges()
        if circuit_size == 0:
            continue
        points.append((false_positive_rate(canonical_circuit_subgraph, corr)/(max_subgraph_size - canonical_circuit_subgraph_size),
                       true_positive_stat(canonical_circuit_subgraph, corr)/canonical_circuit_subgraph_size))
        # points.append((true_positive_stat(canonical_circuit_subgraph, corr)/circuit_size,
        #     true_positive_stat(canonical_circuit_subgraph, corr)/canonical_circuit_subgraph_size))
        print(points[-1])
        if points[-1][0] > 1:
            print(false_positive_rate(canonical_circuit_subgraph, corr, verbose=True))
            assert False
    return points

points = {}
#%%

if "ACDC" in methods:
    if "ACDC" not in points: points["ACDC"] = []
    points["ACDC"].extend(get_points(acdc_corrs))

#%%

if "SP" in methods:
    if "SP" not in points: points["SP"] = []
    points["SP"].extend(get_points(sp_corrs))

#%%

if "16H" in methods:
    if "16H" not in points: points["16H"] = []
    points["16H"].extend(get_points(sixteen_heads_corrs))

#%%

def discard_non_pareto_optimal(points):
    ret = [(0.0, 0.0), *points, (1.0, 1.0)]
    return ret

    for x, y in points:
        for x1, y1 in points:
            if x1 <= x and y1 >= y and (x1, y1) != (x, y):
                break
        else:
            ret.append((x, y))
    return ret

for method in methods:
    processed_points = discard_non_pareto_optimal(points[method]) #  for key in points[method]}
    # sort by x
    processed_points = sorted(processed_points, key=lambda x: x[0])  # for k in processed_points}
    points[method] = processed_points

#%%

def get_roc_figure(all_points, names): # TODO make the plots grey / black / yellow?
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
    roc_figure.update_xaxes(title_text="Precision")
    roc_figure.update_yaxes(title_text="True positive rate")
    return roc_figure

fig = get_roc_figure(list(points.values()), list(points.keys()))
fig.show()

#%%
