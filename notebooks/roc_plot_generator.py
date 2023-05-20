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
from acdc.acdc_utils import false_positive_stat, false_negative_stat, true_positive_stat, get_node_stat
import pandas as pd
import math
import sys
import re
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
import requests
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
from acdc.greaterthan.utils import get_all_greaterthan_things, get_greaterthan_true_edges
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
parser.add_argument("--alg", type=str, default="none", choices=["none", "acdc", "sp", "16h"])
parser.add_argument("--skip-sixteen-heads", action="store_true", help="Skip the 16 heads stuff")
parser.add_argument("--skip-sp", action="store_true", help="Skip the SP stuff")
parser.add_argument("--testing", action="store_true", help="Use testing data instead of validation data")

if IPython.get_ipython() is not None:
    # args = parser.parse_args("--task=ioi --metric=logit_diff --alg=acdc".split())
    args = parser.parse_args("--task=docstring --reset-network=0 --metric=kl_div --alg=acdc --mode=nodes".split())
    # __file__ = "/Users/adria/Documents/2023/ACDC/Automatic-Circuit-Discovery/notebooks/roc_plot_generator.py"
else:
    args = parser.parse_args()


TASK = args.task
METRIC = args.metric # spreadsheet 2 number 120 looks good... rip it's gt
DEVICE = "cpu"
ZERO_ABLATION = True if args.zero_ablation else False
RESET_NETWORK = 1 if args.reset_network else 0
SKIP_ACDC = False
SKIP_SP = True if args.skip_sp else False
SKIP_SIXTEEN_HEADS = True if args.skip_sixteen_heads else False
TESTING = True if args.testing else False
MODE = args.mode

if args.alg != "none":
    SKIP_ACDC = False if args.alg == "acdc" else True
    SKIP_SP = False if args.alg == "sp" else True
    SKIP_SIXTEEN_HEADS = False if args.alg == "16h" else True
    # make a new 
    OUT_FILE = Path(__file__).resolve().parent.parent / "acdc" / "media" / "arthur_plots_data" / f"{args.alg}-{args.task}-{args.metric}-{args.zero_ablation}-{args.reset_network}.json"

    if OUT_FILE.exists():
        print("File already exists, skipping")
        sys.exit(0)
else:
    OUT_FILE = None

# defaults
ACDC_PROJECT_NAME = "remix_school-of-rock/acdc"
ACDC_PRE_RUN_FILTER = {
    # Purposefully omit ``"state": "finished"``
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

    if METRIC == "kl_div":
        ACDC_PRE_RUN_FILTER["group"] = "adria-docstring3"

elif TASK in ["tracr-reverse", "tracr-proportion"]: # do tracr
    USE_POS_EMBED = True

    tracr_task = TASK.split("-")[-1] # "reverse"/"proportion"
    if tracr_task == "proportion":
        get_true_edges = get_tracr_proportion_edges
        num_examples = 50
    elif tracr_task == "reverse":
        get_true_edges = get_tracr_reverse_edges
        num_examples = 6
    else:
        raise NotImplementedError("not a tracr task")

    SP_PRE_RUN_FILTER["group"] = "tracr-shuffled-redo"
    ACDC_PRE_RUN_FILTER["group"] = "acdc-tracr-neurips-3"

    things = get_all_tracr_things(task=tracr_task, metric_name=METRIC, num_examples=num_examples, device=DEVICE)

    # # for propotion, 
    # tl_model(toks_int_values[:1])[0, :, 0] 
    # is the proportion at each space (including irrelevant first position

elif TASK == "ioi":
    num_examples = 100
    things = get_all_ioi_things(num_examples=num_examples, device=DEVICE, metric_name=METRIC)

    if METRIC == "kl_div":
        ACDC_PRE_RUN_FILTER["group"] = "acdc-gt-ioi-redo"
        # ACDC_PROJECT_NAME = "remix_school-of-rock/arthur_ioi_sweep"
        # del ACDC_PRE_RUN_FILTER["config.reset_network"]
        # ACDC_PRE_RUN_FILTER["group"] = "default"

    get_true_edges = partial(get_ioi_true_edges, model=things.tl_model)

elif TASK == "greaterthan":
    num_examples = 100
    things = get_all_greaterthan_things(num_examples=num_examples, metric_name=METRIC, device=DEVICE)
    get_true_edges = partial(get_greaterthan_true_edges, model=things.tl_model)

    if METRIC == "kl_div":
        if ZERO_ABLATION:
            ACDC_PROJECT_NAME = "remix_school-of-rock/arthur_greaterthan_zero_sweep"
            ACDC_PRE_RUN_FILTER = {}
        else:
            # ACDC_PROJECT_NAME = "remix_school-of-rock/arthur_greaterthan_sweep"
            ACDC_PRE_RUN_FILTER["group"] = "acdc-gt-ioi-redo"


elif TASK == "induction":
    num_examples=50
    things = get_all_induction_things(num_examples=num_examples, seq_len=300, device=DEVICE, metric=METRIC)

    ACDC_PRE_RUN_FILTER["group"] = "adria-induction-2"
else:
    raise NotImplementedError("TODO " + TASK)

#%% [markdown]
# Setup the experiment for wrapping functionality nicely

if things is not None:
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

if TASK != "induction":
    d = {(d[0], d[1].hashable_tuple, d[2], d[3].hashable_tuple): False for d in exp.corr.all_edges()}
    d_trues = get_true_edges()
    for k in d_trues:
        d[k] = True
    exp.load_subgraph(d)
    canonical_circuit_subgraph = deepcopy(exp.corr)
    if MODE=="edges":
        canonical_circuit_subgraph_size = canonical_circuit_subgraph.count_no_edges() # we do things differently for nodes

#%%

def get_acdc_runs(
    experiment,
    project_name: str = ACDC_PROJECT_NAME,
    pre_run_filter: dict = ACDC_PRE_RUN_FILTER,
    run_filter: Optional[Callable[[Any], bool]] = ACDC_RUN_FILTER,
    clip: Optional[int] = None,
    maxruns=1_000_000,
):
    if clip is None:
        clip = 100_000 # so we don't clip anything

    api = wandb.Api()
    runs = api.runs(project_name, filters=pre_run_filter)
    if run_filter is None:
        filtered_runs = runs[:clip]
    else:
        filtered_runs = list(filter(run_filter, tqdm(runs[:clip])))
    print(f"loading {len(filtered_runs)} runs with filter {pre_run_filter} and {run_filter}")

    if things is None:
        return [
            (None, {"score": run.config["threshold"], **{k: v for k, v in run.summary.items() if k.startswith("test")}})
            for run in runs
        ]

    corrs = []
    for run in tqdm(filtered_runs):
        score_d = {k: v for k, v in run.summary.items() if k.startswith("test")}
        try:
            score_d["score"] = run.config["threshold"]
        except KeyError:
            score_d["score"] = float(run.name)
        threshold = score_d["score"]

        if "num_edges" in run.summary:
            print("This run n edges:", run.summary["num_edges"])
        # Try to find `edges.pth`
        edges_artifact = None
        for art in run.logged_artifacts():
            if "edges.pth" in art.name:
                edges_artifact = art
                break

        if edges_artifact is None:
            # We'll have to parse the run
            print(f"Edges.pth not found for run {run.name}, falling back to plotly")
            corr = deepcopy(exp.corr)

            # Find latest plotly file which contains the `result` for all edges
            files = run.files(per_page=100_000)
            regexp = re.compile(r"^media/plotly/results_([0-9]+)_[^.]+\.plotly\.json$")

            latest_file = None
            latest_fname_step = -1
            for f in files:
                if (m := regexp.match(f.name)):
                    fname_step = int(m.group(1))
                    if fname_step > latest_fname_step:
                        latest_fname_step = fname_step
                        latest_file = f

            try:
                if latest_file is None:
                    raise wandb.CommError("a")
                with latest_file.download(ROOT / run.name, replace=False, exist_ok=True) as f:
                    d = json.load(f)

                data = d["data"][0]
                assert len(data["text"]) == len(data["y"])

                # Mimic an ACDC run
                for edge, result in zip(data["text"], data["y"]):
                    parent, child = map(parse_interpnode, edge.split(" to "))
                    current_node = child

                    if result < threshold:
                        corr.edges[child.name][child.index][parent.name][parent.index].present = False
                        corr.remove_edge(
                            current_node.name, current_node.index, parent.name, parent.index
                        )
                    else:
                        corr.edges[child.name][child.index][parent.name][parent.index].present = True
                print("Before copying: n_edges=", corr.count_no_edges())

                corr_all_edges = corr.all_edges().items()

                corr_to_copy = deepcopy(exp.corr)
                new_all_edges = corr_to_copy.all_edges()
                for edge in new_all_edges.values():
                    edge.present = False

                for tupl, edge in corr_all_edges:
                    new_all_edges[tupl].present = edge.present

                print("After copying: n_edges=", corr_to_copy.count_no_edges())

                corrs.append((corr_to_copy, score_d))

            except (wandb.CommError, requests.exceptions.HTTPError) as e:
                print(f"Error {e}, falling back to parsing output.log")
                try:
                    with run.file("output.log").download(root=ROOT / run.name, replace=False, exist_ok=True) as f:
                        log_text = f.read()
                    experiment.load_from_wandb_run(log_text)
                    corrs.append((deepcopy(experiment.corr), score_d))
                except Exception:
                    print(f"Loading run {run.name} with state={run.state} config={run.config} totally failed.")
                    continue

        else:
            corr = deepcopy(exp.corr)
            all_edges = corr.all_edges()
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

            corrs.append((corr, score_d))
        print(f"Added run with threshold={score_d['score']}, n_edges={corrs[-1][0].count_no_edges()}")
        if len(corrs)>maxruns: break
    return corrs

if not SKIP_ACDC: # this is slow, so run once
    acdc_corrs = get_acdc_runs(None if things is None else exp, clip = 1 if TESTING else None, maxruns=100000)
    assert len(acdc_corrs) > 1
    print("acdc_corrs", len(acdc_corrs))

#%%

# Do SP stuff
def get_sp_corrs(
    model= None if things is None else things.tl_model,
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

    if things is None:
        return [
            (None, {"score": run.config["lambda_reg"], **{k: v for k, v in run.summary.items() if k.startswith("test")}})
            for run in runs
        ]

    corrs = []
    for run in filtered_runs:
        try:
            nodes_to_mask_strings = run.summary["nodes_to_mask"]
        except KeyError:
            continue
        nodes_to_mask = [parse_interpnode(s) for s in nodes_to_mask_strings]
        corr = correspondence_from_mask(
            model = model,
            nodes_to_mask=nodes_to_mask,
            use_pos_embed = USE_POS_EMBED,
        )
        score_d = {k: v for k, v in run.summary.items() if k.startswith("test")}
        score_d["score"] = run.config["lambda_reg"]
        corrs.append((corr, score_d))
    return corrs

if not SKIP_SP: # this is slow, so run once
    sp_corrs = get_sp_corrs(clip = 1 if TESTING else None) # clip for testing
    assert len(sp_corrs) > 1
    print("sp_corrs", len(sp_corrs))

#%%

def get_sixteen_heads_corrs(
    project_name = SIXTEEN_HEADS_PROJECT_NAME,
    pre_run_filter = SIXTEEN_HEADS_PRE_RUN_FILTER,
    run_filter = SIXTEEN_HEADS_RUN_FILTER,
    model= None if things is None else things.tl_model,
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

    corrs = []
    nodes_to_mask = []
    cum_score = 0.0
    for nodes, hook_name, idx, score in tqdm(nodes_names_indices):
        if score == "NaN":
            score = 0.0
        if things is None:
            corr = None
        else:
            nodes_to_mask += list(map(parse_interpnode, nodes))
            corr = correspondence_from_mask(model=model, nodes_to_mask=nodes_to_mask, use_pos_embed=exp.use_pos_embed)
        cum_score += score
        score_d = {k: v for k, v in run.summary.items() if k.startswith("test")}
        score_d["score"] = cum_score
        corrs.append((corr, score_d))
    return corrs

if "sixteen_heads_corrs" not in locals() and not SKIP_SIXTEEN_HEADS: # this is slow, so run once
    sixteen_heads_corrs = get_sixteen_heads_corrs()
    assert len(sixteen_heads_corrs) > 1
    print("sixteen_heads_corrs", len(sixteen_heads_corrs))

#%%

methods = []

if not SKIP_ACDC: methods.append("ACDC") 
if not SKIP_SP: methods.append("SP")
if not SKIP_SIXTEEN_HEADS: methods.append("16H")

#%%

# get points from correspondence
def get_points(corrs_and_scores, decreasing=True):
    keys = set()
    for _, s in corrs_and_scores:
        keys.update(s.keys())

    # if things is None:
    #     points = []
    #     n_skipped = 0
    #     for _, score in sorted(corrs_and_scores, key=lambda x: x[1]["score"], reverse=decreasing):
    #         if set(score.keys()) != keys:
    #             n_skipped += 1
    #             continue
    #         points.append(score)
    #     assert n_skipped <= 2
    #     return points

    if decreasing:
        init_point = {k: math.inf for k in keys}
        if TASK != "induction":
            init_point["fpr"] = 0.0
            init_point["tpr"] = 0.0
            init_point["precision"] = 1.0
        init_point["n_edges"] = math.nan

        end_point = {k: -math.inf for k in keys}
        if TASK != "induction":
            end_point["fpr"] = 1.0
            end_point["tpr"] = 1.0
            end_point["precision"] = 0.0
        end_point["n_edges"] = math.nan
    else:
        init_point = {k: -math.inf for k in keys}
        if TASK != "induction":
            init_point["fpr"] = 1.0
            init_point["tpr"] = 1.0
            init_point["precision"] = 0.0
        init_point["n_edges"] = math.nan

        end_point = {k: math.inf for k in keys}
        if TASK != "induction":
            end_point["fpr"] = 0.0
            end_point["tpr"] = 0.0
            end_point["precision"] = 1.0
        end_point["n_edges"] = math.nan

    points = [init_point]

    n_skipped = 0

    for corr, score in tqdm(sorted(corrs_and_scores, key=lambda x: x[1]["score"], reverse=decreasing)):
        if set(score.keys()) != keys:
            a = init_point.copy()
            a.update(score)
            score = a

        circuit_size = corr.count_no_edges()
        if circuit_size == 0:
            continue
        if TASK != "induction":
            tp_stat = true_positive_stat(ground_truth=canonical_circuit_subgraph, recovered=corr)
            if MODE == "edges":
                score.update(
                    {
                        "fpr": (
                            false_positive_stat(ground_truth=canonical_circuit_subgraph, recovered=corr)
                            / (max_subgraph_size - canonical_circuit_subgraph_size)  # FP / (TOTAL - P) = FP / N
                        ),
                        "tpr": tp_stat / canonical_circuit_subgraph_size,  # TP / P
                        "precision": tp_stat / circuit_size,  # TP / (TP + FP) = TP / (predicted positive)
                        "n_edges": circuit_size,
                    }
                )

            if MODE == "nodes":
                _, len_ground_truth_all_nodes, len_recovered_all_nodes, max_subgraph_size = get_node_stat(ground_truth=canonical_circuit_subgraph, recovered=corr, mode="true positive", meta=True)

                updater = {
                    "fpr": false_positive_stat(ground_truth=canonical_circuit_subgraph, recovered=corr, mode="nodes"),
                    "tpr": true_positive_stat(ground_truth=canonical_circuit_subgraph, recovered=corr, mode="nodes"),
                    "precision": -69.0,
                    "n_edges": len_recovered_all_nodes, # well not edges but ygm
                }

                if (max_subgraph_size-len_ground_truth_all_nodes) == 0: # grumble
                    updater["fpr"] = 0.0
                else:
                    updater["fpr"] /= (max_subgraph_size-len_ground_truth_all_nodes)
                if len_ground_truth_all_nodes == 0:
                    updater["tpr"] = 0.0
                else:
                    updater["tpr"] /= len_ground_truth_all_nodes

                score.update(
                    updater, # TODO others
                )

        else:
            score.update({"n_edges": circuit_size})
        points.append(score)
    assert n_skipped <= 2

    points.append(end_point)
    assert all(("n_edges" in p) for p in points)
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
    points["16H"].extend(get_points(sixteen_heads_corrs, decreasing=False))

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

if OUT_FILE is None:
    fig = get_roc_figure(list(points.values()), list(points.keys()))
    fig.show()

#%%

if OUT_FILE is not None:
    assert args.alg != "none"
    ALG = args.alg.upper()

    os.makedirs(OUT_FILE.parent, exist_ok=True)

    ablation = "zero_ablation" if ZERO_ABLATION else "random_ablation"
    weights = "reset" if RESET_NETWORK else "trained"

    out_dict = {
        weights: {
            ablation: {
                args.task: {
                    args.metric: {
                        ALG: {k: [p[k] for p in points[ALG]] for k in points[ALG][0].keys()},
                    },
                },
            },
        },
    }

    with open(OUT_FILE, "w") as f:
        json.dump(out_dict, f, indent=2)

# %%
