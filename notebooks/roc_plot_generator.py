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

from notebooks.emacs_plotly_render import set_plotly_renderer
# set_plotly_renderer("emacs")


def get_col(df, col): # dumb util
    non_null_entries = list(df.loc[df[col].notnull(), col])
    return non_null_entries 

torch.autograd.set_grad_enabled(False)

#%% [markdown]

parser = argparse.ArgumentParser(description="Used to control ROC plot scripts (for standardisation with other files...)")
parser.add_argument('--task', type=str, required=True, choices=['ioi', 'docstring', 'induction', 'tracr-reverse', 'tracr-proportion', 'greaterthan'], help='Choose a task from the available options: ioi, docstring, induction, tracr (WIPs)')
parser.add_argument("--mode", type=str, required=False, choices=["edges", "nodes"], help="Choose a mode from the available options: edges, nodes", default="edges") # TODO implement nodes
parser.add_argument('--zero-ablation', action='store_true', help='Use zero ablation')
parser.add_argument("--skip-sixteen-heads", action="store_true", help="Skip the 16 heads stuff")
parser.add_argument("--skip-sp", action="store_true", help="Skip the SP stuff")
parser.add_argument("--testing", action="store_true", help="Use testing data instead of validation data")

# for now, force the args to be the same as the ones in the notebook, later make this a CLI tool
if IPython.get_ipython() is not None: # heheh get around this failing in notebooks
    args = parser.parse_args("--task tracr-proportion --testing".split())
else:
    args = parser.parse_args()

TASK = args.task
METRIC = "kl_div"
DEVICE = "cpu"
ZERO_ABLATION = True if args.zero_ablation else False
SKIP_ACDC=False
SKIP_SP = True if args.skip_sp else False
SKIP_SIXTEEN_HEADS = True if args.skip_sixteen_heads else False
TESTING = True if args.testing else False

# defaults
ACDC_PROJECT_NAME = "remix_school-of-rock/acdc"
ACDC_PRE_RUN_FILTER = None
ACDC_NAME_FILTER = None

# # for SP # filters are more annoying since some things are nested in groups
SP_PROJECT_NAME = "remix_school-of-rock/induction-sp-replicate"
SP_PRE_RUN_FILTER = {}
SP_RUN_FILTER = lambda _: True

# # for 16 heads # sixteen heads is just one run
SIXTEEN_HEADS_PROJECT_NAME = "remix_school-of-rock/acdc"
SIXTEEN_HEADS_RUN = None

def task_filter(run, task, verbose = True):
    try:
        assert json.loads(run.json_config)["task"]["value"] == task
    except Exception as e:
        if verbose:
            print("errorre", e)
        return False
    return True

#%% [markdown]
# Setup
# substantial copy paste from main.py, with some new configs, directories...

if TASK == "docstring":
    num_examples = 50
    seq_len = 41
    things = get_all_docstring_things(num_examples=num_examples, seq_len=seq_len, device=DEVICE,
                                                metric_name=METRIC, correct_incorrect_wandb=False)

    get_true_edges = get_docstring_subgraph_true_edges

    ACDC_PROJECT_NAME = "remix_school-of-rock/acdc"
    ACDC_PRE_RUN_FILTER = None
    ACDC_RUN_FILTER = lambda run: run.name.startswith("agarriga-docstring")

    SP_PROJECT_NAME = "remix_school-of-rock/induction-sp-replicate"
    def sp_run_filter(run): # name):
        name = run.name
        if not name.startswith("agarriga-sp-"): return False
        try:
            int(name.split("-")[-1])
        except:
            return False
        return 0 <= int(name.split("-")[-1]) <= 319
    SP_PRE_RUN_FILTER = {"group": "docstring3"} # used for the api.run(filter=...)
    SP_RUN_FILTER = lambda run: sp_run_filter(run)

    SIXTEEN_HEADS_PROJECT_NAME = "remix_school-of-rock/acdc"
    SIXTEEN_HEADS_RUN = "mrzpsjtw"

elif TASK in ["tracr-reverse", "tracr-proportion"]: # do tracr
    tracr_task = TASK.split("-")[-1] # "reverse"

    # this implementation doesn't ablate the position embeddings (which the plots in the paper do do), so results are different. See the rust_circuit implemntation if this need be checked
    # also there's no splitting by neuron yet TODO

    use_pos_embed = True
    ZERO_ABLATION = True


    ACDC_PROJECT_NAME = None
    ACDC_PRE_RUN_FILTER = None
    ACDC_RUN_FILTER = lambda name: name == "not_a_real_run" # broken, see below
    points["ACDC"] = [(0.0, 1.0)]

    if tracr_task == "proportion":
        get_true_edges = get_tracr_proportion_edges
        num_examples = 50

        SIXTEEN_HEADS_PROJECT_NAME = "remix_school-of-rock/acdc"
        SIXTEEN_HEADS_RUN = "siurl8zp"

        # # sad, we updated the tracr stuff
        # SP_PROJECT_NAME = "remix_school-of-rock/induction-sp-replicate"
        # SP_PRE_RUN_FILTER = {"group": "complete-spreadsheet-4"}
        # three_digit_numbers = ["320", "341", "340", "339", "338", "337", "336", "335", "334", "332"]
        # SP_RUN_FILTER = lambda name: len(name)>3 and name[-3:] in three_digit_numbers

    elif tracr_task == "reverse":
        get_true_edges = get_tracr_reverse_edges
        num_examples = 6

        SIXTEEN_HEADS_PROJECT_NAME = "remix_school-of-rock/acdc"
        SIXTEEN_HEADS_RUN = "fccs2o7o"

    else:
        raise NotImplementedError("not a tracr task")

    things = get_all_tracr_things(task=tracr_task, metric_name=METRIC, num_examples=num_examples, device=DEVICE)

    # # for propotion, 
    # tl_model(toks_int_values[:1])[0, :, 0] 
    # is the proportion at each space (including irrelevant first position

elif TASK == "ioi":
    num_examples = 100
    things = get_all_ioi_things(num_examples=num_examples, device=DEVICE, metric_name=args.metric)

    ACDC_PROJECT_NAME = "remix_school-of-rock/arthur_ioi_sweep"

    ACDC_RUN_FILTER = partial(task_filter, task="ioi")
    ACDC_PRE_RUN_FILTER = None

    SP_PROJECT_NAME = "remix_school-of-rock/induction-sp-replicate"
    SP_PRE_RUN_FILTER = {"group": "complete-spreadsheet-4"}
    SP_RUN_FILTER = partial(task_filter, task="ioi", verbose=False)

    SIXTEEN_HEADS_PROJECT_NAME = "remix_school-of-rock/acdc"
    SIXTEEN_HEADS_RUN = "yjreihd0"

    get_true_edges = partial(get_ioi_true_edges, model=things.tl_model)
    
elif TASK == "greaterthan":
    num_examples = 100
    things = get_all_greaterthan_things(num_examples=num_examples, metric_name=args.metric, device=DEVICE)

    SIXTEEN_HEADS_PROJECT_NAME = "remix_school-of-rock/acdc"
    SIXTEEN_HEADS_RUN = "zcxh8rbm"

    raise NotImplementedError("TODO")

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
    use_pos_embed=use_pos_embed,
)
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
# <h2> Arthur plays about with loading in graphs </h2>
# <h3> Not relevant for doing ACDC runs </h3>
# <p> Get Adria's docstring runs! </p>

def get_acdc_runs(
    experiment,
    project_name: str = ACDC_PROJECT_NAME,
    run_filter: Callable[[Any], bool] = ACDC_RUN_FILTER,
    clip = None,
):
    if project_name is None:
        return []

    if clip is None:
        clip = 100_000 # so we don't clip anything

    api = wandb.Api()
    runs = api.runs(project_name)
    filtered_runs = []
    for run in tqdm(runs[:clip]):
        if run_filter(run):
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
            print(run.id, "and errorr", e)
            cnt+=1
            continue
    return corrs

if "acdc_corrs" not in locals() and not SKIP_ACDC: # this is slow, so run once
    acdc_corrs = get_acdc_runs(exp, clip = 100 if TESTING else None)

#%%

# Do SP stuff
def get_sp_corrs(
    experiment, 
    model = things.tl_model,
    project_name: str = SP_PROJECT_NAME,
    pre_run_filter: Dict = SP_PRE_RUN_FILTER,
    run_filter: Callable[[Any], bool] = SP_RUN_FILTER,
    clip = None,
):
    if project_name is None:
        return []

    if clip is None:
        clip = 100_000

    api = wandb.Api()
    runs=api.runs(
        project_name,
        filters=pre_run_filter,
    )
    filtered_runs = []
    for run in tqdm(runs):
        if run_filter(run):
            filtered_runs.append(run)
    cnt = 0
    corrs = []
    ret = []

    verbose = True

    for run in tqdm(filtered_runs[:clip]):
        df = pd.DataFrame(run.scan_history())

        mask_scores_entries = get_col(df, "mask_scores")
        assert len(mask_scores_entries) > 0
        entry = mask_scores_entries[-1]

        try:
            nodes_to_mask_entries = get_col(df, "nodes_to_mask")
        except Exception as e:
            print(e, "... was an error")
            continue        

        assert len(nodes_to_mask_entries) ==1, len(nodes_to_mask_entries)
        nodes_to_mask_strings = nodes_to_mask_entries[0]
        print(nodes_to_mask_strings)
        nodes_to_mask_dict = [parse_interpnode(s) for s in nodes_to_mask_strings if parse_interpnode(s, verbose=verbose) is not None]

        if len(nodes_to_mask_dict) != len(nodes_to_mask_strings):
            verbose = False # ignore future errors

        number_of_edges_entries = get_col(df, "number_of_edges")
        assert len(number_of_edges_entries) == 1, len(number_of_edges_entries)
        number_of_edges = number_of_edges_entries[0]

        kl_divs = get_col(df, "test_kl_div")
        assert len(kl_divs) == 1, len(kl_divs)
        kl_div = kl_divs[0]

        corr = correspondence_from_mask(
            model = model,
            nodes_to_mask=nodes_to_mask_dict,
        )

        if corr.count_no_edges() != number_of_edges:
            warnings.warn(str(corr.count_no_edges()) + " ooh err " + str(number_of_edges))

        ret.append(corr) # do we need KL too? I think no..

    return ret

if "sp_corrs" not in locals() and not SKIP_SP: # this is slow, so run once
    sp_corrs = get_sp_corrs(exp, clip = 10 if TESTING else None) # clip for testing

#%%

def get_sixteen_heads_corrs(
    experiment = exp,
    project_name = SIXTEEN_HEADS_PROJECT_NAME,
    run_name = SIXTEEN_HEADS_RUN,
    model= things.tl_model,
):
# experiment = exp
# project_name = SIXTEEN_HEADS_PROJECT_NAME
# run_name = SIXTEEN_HEADS_RUN
# model = tl_model
# if True:
    api = wandb.Api()
    run=api.run(SIXTEEN_HEADS_PROJECT_NAME + "/" + SIXTEEN_HEADS_RUN) # sorry fomratting..
    df = pd.DataFrame(run.scan_history()) 

    # start with everything involved except attention
    for t, e in experiment.corr.all_edges().items():
        e.present = True
    experiment.remove_all_non_attention_connections()
    corrs = [deepcopy(experiment.corr)]
    print(experiment.count_no_edges())

    history = run.scan_history()
    layer_indices = list(pd.DataFrame(history)["layer_idx"])
    try:
        int(layer_indices[0])
    except:
        pass
    else:
        raise ValueError("Expected a NAN at start, formatting different")
    
    layer_indices = layer_indices[1:]
    head_indices = list(pd.DataFrame(history)["head_idx"])[1:]
    assert len(head_indices) == len(layer_indices), (len(head_indices), len(layer_indices))
    head_indices = [int(i) for i in head_indices]
    layer_indices = [int(i) for i in layer_indices]

    nodes_to_mask_dict = heads_to_nodes_to_mask(
        heads = [(layer_idx, head_idx) for layer_idx in range(tl_model.cfg.n_layers) for head_idx in range(tl_model.cfg.n_heads)], return_dict=True
    )
    print(nodes_to_mask_dict)
    corr2 = correspondence_from_mask(
        model = model,
        use_pos_embed=exp.use_pos_embed,
        nodes_to_mask=list(nodes_to_mask_dict.values()),
    )

    assert set(corr2.all_edges().keys()) == set(experiment.corr.all_edges().keys())
    for t, e in corr2.all_edges().items():
        assert experiment.corr.edges[t[0]][t[1]][t[2]][t[3]].present == e.present, (t, e.present, experiment.corr.edges[t[0]][t[1]][t[2]][t[3]].present)

    for layer_idx, head_idx in tqdm(zip(layer_indices, head_indices)):
        # exp.add_back_head(layer_idx, head_idx)
        for letter in "qkv":
            nodes_to_mask_dict.pop(f"blocks.{layer_idx}.attn.hook_{letter}[COL, COL, {head_idx}]") # weirdly we don't have the q_input; just outputs of things....
        nodes_to_mask_dict.pop(f"blocks.{layer_idx}.attn.hook_result[COL, COL, {head_idx}]")

        corr = correspondence_from_mask(
            model = model,
            use_pos_embed = exp.use_pos_embed,
            nodes_to_mask=list(nodes_to_mask_dict.values()),
        )

        corrs.append(deepcopy(corr))

    return corrs # TODO add back in

if "sixteen_heads_corrs" not in locals() and not SKIP_SIXTEEN_HEADS: # this is slow, so run once
    sixteen_heads_corrs = get_sixteen_heads_corrs()

#%%

methods = []

if not SKIP_ACDC: methods.append("ACDC") 
if not SKIP_SP: methods.append("SP")
if not SKIP_SIXTEEN_HEADS: methods.append("16H")

#%%

# get points from correspondence
def get_points(corrs):
    points = []
    for corr in tqdm(corrs): 
        circuit_size = corr.count_no_edges()
        if circuit_size == 0:
            continue
        points.append((false_positive_rate(canonical_circuit_subgraph, corr)/circuit_size, true_positive_stat(canonical_circuit_subgraph, corr)/canonical_circuit_subgraph_size))
        print(points[-1])
        if points[-1][0] > 1:
            print(false_positive_rate(canonical_circuit_subgraph, corr, verbose=True))
            assert False
    return points

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
    ret = [(0.0, 0.0), (1.0, 1.0)]
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
    roc_figure.update_xaxes(title_text="False positive rate")
    roc_figure.update_yaxes(title_text="True positive rate")
    return roc_figure

fig = get_roc_figure(list(points.values()), list(points.keys()))
fig.show()

#%%
