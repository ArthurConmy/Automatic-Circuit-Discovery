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

import collections
import IPython

if IPython.get_ipython() is not None:
    IPython.get_ipython().run_line_magic("load_ext", "autoreload")  # type: ignore
    IPython.get_ipython().run_line_magic("autoreload", "2")  # type: ignore

from copy import deepcopy
from subnetwork_probing.train import iterative_correspondence_from_mask
from acdc.acdc_utils import filter_nodes, get_edge_stats, get_node_stats, get_present_nodes, reset_network
import pandas as pd
import gc
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
from acdc.TLACDCInterpNode import parse_interpnode, heads_to_nodes_to_mask
import pickle
import wandb
import IPython
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
import pygraphviz as pgv
from enum import Enum
from dataclasses import dataclass
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

from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens.HookedTransformer import (
    HookedTransformer,
)
from acdc.tracr_task.utils import get_tracr_model_input_and_tl_model, get_tracr_proportion_edges, get_tracr_reverse_edges, get_all_tracr_things
from acdc.docstring.utils import get_all_docstring_things, get_docstring_model, get_docstring_subgraph_true_edges
from acdc.acdc_utils import (
    make_nd_dict,
    shuffle_tensor,
    cleanup,
    ct,
)

from acdc.TLACDCEdge import (
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
    ioi_group_colorscheme,
)
from acdc.induction.utils import (
    get_all_induction_things,
    get_validation_data,
    get_good_induction_candidates,
    get_mask_repeat_candidates,
)
from acdc.acdc_graphics import (
    build_colorscheme,
    get_node_name,
    show,
)
from acdc.ioi.utils import (
    get_all_ioi_things,
    get_gpt2_small,
)
import argparse
from acdc.greaterthan.utils import get_all_greaterthan_things, get_greaterthan_true_edges, greaterthan_group_colorscheme
from pathlib import Path

from notebooks.emacs_plotly_render import set_plotly_renderer
set_plotly_renderer("emacs")


def get_col(df, col): # dumb util
    non_null_entries = list(df.loc[df[col].notnull(), col])
    return non_null_entries 

torch.autograd.set_grad_enabled(False)

#%% [markdown]

parser = argparse.ArgumentParser(description="Used to control ROC plot scripts (for standardisation with other files...)")
parser.add_argument('--task', type=str, required=True, choices=['ioi', 'docstring', 'induction', 'tracr-reverse', 'tracr-proportion', 'greaterthan'], help='Choose a task from the available options: ioi, docstring, induction, tracr-reverse, tracr-proportion, greaterthan')
parser.add_argument("--mode", type=str, required=False, choices=["edges", "nodes"], help="Choose a mode from the available options: edges, nodes", default="edges") # TODO implement nodes
parser.add_argument('--zero-ablation', action='store_true', help='Use zero ablation')
parser.add_argument('--metric', type=str, default="kl_div", help="Which metric to use for the experiment")
parser.add_argument('--reset-network', type=int, default=0, help="Whether to reset the network we're operating on before running interp on it")
parser.add_argument("--alg", type=str, default="none", choices=["none", "acdc", "sp", "16h", "canonical"])
parser.add_argument("--skip-sixteen-heads", action="store_true", help="Skip the 16 heads stuff")
parser.add_argument("--skip-sp", action="store_true", help="Skip the SP stuff")
parser.add_argument("--testing", action="store_true", help="Use testing data instead of validation data")
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--out-dir", type=str, default="DEFAULT")
parser.add_argument('--torch-num-threads', type=int, default=0, help="How many threads to use for torch (0=all)")
parser.add_argument('--seed', type=int, default=42, help="Random seed")
parser.add_argument("--canonical-graph-save-dir", type=str, default="DEFAULT")
parser.add_argument("--only-save-canonical", action="store_true", help="Only save the canonical graph")
parser.add_argument("--ignore-missing-score", action="store_true", help="Ignore runs that are missing score")

if IPython.get_ipython() is not None:
    args = parser.parse_args("--task=ioi --metric=kl_div --alg=sp".split())
    if "arthur" not in __file__:
        __file__ = "/Users/adria/Documents/2023/ACDC/Automatic-Circuit-Discovery/notebooks/roc_plot_generator.py"
else:
    args = parser.parse_args()

if not args.mode == "edges":
    raise NotImplementedError("Only edges mode is implemented for now")


if args.torch_num_threads > 0:
    torch.set_num_threads(args.torch_num_threads)
torch.manual_seed(args.seed)


TASK = args.task
METRIC = args.metric
DEVICE = args.device
ZERO_ABLATION = True if args.zero_ablation else False
RESET_NETWORK = 1 if args.reset_network else 0
SKIP_ACDC = False
SKIP_SP = True if args.skip_sp else False
SKIP_SIXTEEN_HEADS = True if args.skip_sixteen_heads else False
SKIP_CANONICAL = True
TESTING = True if args.testing else False
ONLY_SAVE_CANONICAL = True if args.only_save_canonical else False

if args.out_dir == "DEFAULT":
    OUT_DIR = Path(__file__).resolve().parent.parent / "experiments" / "results" / f"{'arthur_' if 'arthur' in __file__ else ''}plots_data"
    CANONICAL_OUT_DIR = Path(__file__).resolve().parent.parent / "experiments" / "results" / "canonical_circuits"
else:
    OUT_DIR = Path(args.out_dir)
    CANONICAL_OUT_DIR = Path(args.canonical_graph_save_dir)
CANONICAL_OUT_DIR.mkdir(exist_ok=True, parents=True)

if args.alg != "none":
    SKIP_ACDC = False if args.alg == "acdc" else True
    SKIP_SP = False if args.alg == "sp" else True
    SKIP_SIXTEEN_HEADS = False if args.alg == "16h" else True
    SKIP_CANONICAL = False if args.alg == "canonical" else True
    OUT_FILE = OUT_DIR / f"{args.alg}-{args.task}-{args.metric}-{args.zero_ablation}-{args.reset_network}.json"

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


ROOT = Path(os.environ["HOME"]) / ".cache" / "artifacts_for_plot"
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

    if RESET_NETWORK:
        ACDC_PRE_RUN_FILTER["group"] = "reset-networks-neurips"

elif TASK in ["tracr-reverse", "tracr-proportion"]: # do tracr
    USE_POS_EMBED = True

    tracr_task = TASK.split("-")[-1] # "reverse"/"proportion"
    if tracr_task == "proportion":
        get_true_edges = get_tracr_proportion_edges
        num_examples = 50
        SP_PRE_RUN_FILTER["group"] = "tracr-shuffled-redo"
    elif tracr_task == "reverse":
        get_true_edges = get_tracr_reverse_edges
        num_examples = 6
        SP_PRE_RUN_FILTER["group"] = "tracr-shuffled-redo-2"
    else:
        raise NotImplementedError("not a tracr task")

    ACDC_PRE_RUN_FILTER["group"] = "acdc-tracr-neurips-5"
    
    if not ZERO_ABLATION:
        ACDC_PRE_RUN_FILTER.pop("group")
        ACDC_PROJECT_NAME = "remix_school-of-rock/arthur_tracr_fix"

    things = get_all_tracr_things(task=tracr_task, metric_name=METRIC, num_examples=num_examples, device=DEVICE)

    # # for propotion, 
    # tl_model(toks_int_values[:1])[0, :, 0] 
    # is the proportion at each space (including irrelevant first position

elif TASK == "ioi":
    num_examples = 100
    things = get_all_ioi_things(num_examples=num_examples, device=DEVICE, metric_name=METRIC)

    if METRIC == "kl_div" and not RESET_NETWORK:
        if ZERO_ABLATION:
            ACDC_PROJECT_NAME = "remix_school-of-rock/arthur_ioi_sweep"
            del ACDC_PRE_RUN_FILTER["config.reset_network"]
            ACDC_PRE_RUN_FILTER["group"] = "default"
        else:
            ACDC_PRE_RUN_FILTER["group"] = "acdc-ioi-gt-redo2"
    else:
        try:
            del ACDC_PRE_RUN_FILTER["group"]
        except KeyError:
            pass
        ACDC_PRE_RUN_FILTER = {
            "$or": [
                {"group": "reset-networks-neurips", **ACDC_PRE_RUN_FILTER},
                {"group": "acdc-gt-ioi-redo", **ACDC_PRE_RUN_FILTER},
                {"group": "acdc-spreadsheet2", **ACDC_PRE_RUN_FILTER},
            ]
        }

    get_true_edges = partial(get_ioi_true_edges, model=things.tl_model)

elif TASK == "greaterthan":
    num_examples = 100
    things = get_all_greaterthan_things(num_examples=num_examples, metric_name=METRIC, device=DEVICE)
    get_true_edges = partial(get_greaterthan_true_edges, model=things.tl_model)

    SP_PRE_RUN_FILTER["group"] = "tracr-shuffled-redo"
    SP_PROJECT_NAME = "remix_school-of-rock/induction_arthur"  # moved here manually

    if METRIC == "kl_div" and not RESET_NETWORK:
        if ZERO_ABLATION:
            ACDC_PROJECT_NAME = "remix_school-of-rock/arthur_greaterthan_zero_sweep"
            ACDC_PRE_RUN_FILTER = {}
        else:
            del ACDC_PRE_RUN_FILTER["group"]

    if METRIC == "greaterthan" and not RESET_NETWORK and not ZERO_ABLATION:
        ACDC_PROJECT_NAME = "remix_school-of-rock/arthur_greaterthan_sweep_fixed_random"
        ACDC_PRE_RUN_FILTER = {}
    elif METRIC == "greaterthan":
        ACDC_PRE_RUN_FILTER["group"] = "gt-fix-metric"
    elif RESET_NETWORK:
        try:
            del ACDC_PRE_RUN_FILTER["group"]
        except KeyError:
            pass
        ACDC_PRE_RUN_FILTER = {
            "$or": [
                {"group": "reset-networks-neurips", **ACDC_PRE_RUN_FILTER},
                {"group": "acdc-gt-ioi-redo", **ACDC_PRE_RUN_FILTER},
                {"group": "acdc-spreadsheet2", **ACDC_PRE_RUN_FILTER},
            ]
        }


elif TASK == "induction":
    num_examples=50
    things = get_all_induction_things(num_examples=num_examples, seq_len=300, device=DEVICE, metric=METRIC)

    if RESET_NETWORK:
        ACDC_PRE_RUN_FILTER["group"] = "reset-networks-neurips"
    else:
        # ACDC_PRE_RUN_FILTER["group"] = "adria-induction-2"
        ACDC_PRE_RUN_FILTER["group"] = "adria-induction-3"
else:
    raise NotImplementedError("TODO " + TASK)

if RESET_NETWORK and TASK != "greaterthan" and not TASK.startswith("tracr"):
    SP_PRE_RUN_FILTER["group"] = "tracr-shuffled-redo"

if RESET_NETWORK:
    reset_network(TASK, DEVICE, things.tl_model)
    gc.collect()
    torch.cuda.empty_cache()

#%% [markdown]
# Setup the experiment for wrapping functionality nicely

import gc
gc.collect()
torch.cuda.empty_cache()

things.tl_model.reset_hooks()
exp = TLACDCExperiment(
    model=things.tl_model,
    threshold=100_000,
    early_exit=SKIP_ACDC and SKIP_CANONICAL,
    using_wandb=False,
    zero_ablation=bool(ZERO_ABLATION),
    # Process very little data if just building the canonical graph
    ds=things.test_data[slice(1) if ONLY_SAVE_CANONICAL else slice(None)],
    ref_ds=things.test_patch_data[slice(1) if ONLY_SAVE_CANONICAL else slice(None)],
    metric=things.validation_metric,
    second_metric=None,
    verbose=True,
    use_pos_embed=USE_POS_EMBED,
    online_cache_cpu=False,
    corrupted_cache_cpu=False,
)
if not SKIP_ACDC and not ONLY_SAVE_CANONICAL:
    exp.setup_corrupted_cache()

max_subgraph_size = exp.corr.count_no_edges()

#%% [markdown]
# Load the *canonical* circuit

COLORSCHEME_FOR = collections.defaultdict(lambda: (lambda: "Pastel2"), {
    "ioi": ioi_group_colorscheme,
    "greaterthan": greaterthan_group_colorscheme,
})

if TASK != "induction":
    d = {(d[0], d[1].hashable_tuple, d[2], d[3].hashable_tuple): False for d in exp.corr.all_edges()}
    d_trues = get_true_edges()
    # if ONLY_SAVE_CANONICAL and TASK == "ioi":
    #     # Remove non-adjacent layer connections
    #     def layer(name):
    #         return int(name.split(".")[1])

    #     for t in list(d_trues.keys()):
    #         if abs(layer(t[0]) - layer(t[2])) > 1:
    #             del d_trues[t]

    for k in d_trues:
        d[k] = True

    exp.load_subgraph(d)
    canonical_circuit_subgraph = deepcopy(exp.corr)
    for t in exp.corr.all_edges().keys():
        exp.corr.edges[t[0]][t[1]][t[2]][t[3]].present = True
    canonical_circuit_subgraph_size = canonical_circuit_subgraph.count_no_edges()

    # and reset the sugbgraph...
    for t, e in exp.corr.all_edges().items():
        exp.corr.edges[t[0]][t[1]][t[2]][t[3]].present = True

    for edge in canonical_circuit_subgraph.all_edges().values():
        edge.effect_size = 1.0  # make it visible

    if ONLY_SAVE_CANONICAL:
        colorscheme = COLORSCHEME_FOR[TASK]()
        g: pgv.AGraph = show(
            canonical_circuit_subgraph,
            fname=CANONICAL_OUT_DIR / f"{TASK}.gv",
            colorscheme=colorscheme,
            show_full_index=False,
            layout="neato" if TASK in ["ioi", "greaterthan"] else "dot",
        )

        show(
            canonical_circuit_subgraph,
            fname=CANONICAL_OUT_DIR / f"{TASK}_mlp.gv",
            colorscheme=colorscheme,
            show_full_index=False,
            remove_qkv=True,
        )

        if TASK in ["ioi", "greaterthan"]:
            no_mlp = deepcopy(canonical_circuit_subgraph)
            for (n_to, _, n_from, _), e in no_mlp.all_edges().items():
                if "mlp" in n_to or "mlp" in n_from:
                    e.present = False
            show(
                no_mlp,
                fname=CANONICAL_OUT_DIR / f"{TASK}_heads_qkv.gv",
                colorscheme=colorscheme,
                show_full_index=False,
                remove_self_loops=True,
                remove_qkv=False,
                layout="dot",
            )
            show(
                no_mlp,
                fname=CANONICAL_OUT_DIR / f"{TASK}_heads.gv",
                colorscheme=colorscheme,
                show_full_index=False,
                remove_self_loops=True,
                remove_qkv=True,
                layout="dot",
            )

if ONLY_SAVE_CANONICAL:
    sys.exit(0)
#%%

@dataclass(frozen=True)
class AcdcRunCandidate:
    threshold: float
    steps: int
    run: wandb.apis.public.Run
    score_d: dict
    corr: TLACDCCorrespondence

def get_acdc_runs(
    exp,
    project_name: str = ACDC_PROJECT_NAME,
    pre_run_filter: dict = ACDC_PRE_RUN_FILTER,
    run_filter: Optional[Callable[[Any], bool]] = ACDC_RUN_FILTER,
    clip: Optional[int] = None,
    return_ids: bool = False,
):
# experiment = exp
# project_name = ACDC_PROJECT_NAME
# pre_run_filter = ACDC_PRE_RUN_FILTER
# run_filter = ACDC_RUN_FILTER
# clip = None
# return_ids = False
# if True:
    if clip is None:
        clip = 100_000 # so we don't clip anything

    api = wandb.Api()
    runs = api.runs(project_name, filters=pre_run_filter)
    if run_filter is None:
        filtered_runs = list(runs)[:clip]
    else:
        filtered_runs = list(filter(run_filter, tqdm(list(runs)[:clip])))
    print(f"loading {len(filtered_runs)} runs with filter {pre_run_filter} and {run_filter}")

    threshold_to_run_map: dict[float, AcdcRunCandidate] = {}

    def add_run_for_processing(candidate: AcdcRunCandidate):
        if candidate.threshold not in threshold_to_run_map:
            threshold_to_run_map[candidate.threshold] = candidate
        else:
            if candidate.steps > threshold_to_run_map[candidate.threshold].steps:
                threshold_to_run_map[candidate.threshold] = candidate

    for run in filtered_runs:
        score_d = {k: v for k, v in run.summary.items() if k.startswith("test")}
        try:
            score_d["steps"] = run.summary["_step"]
        except KeyError:
            continue  # Run has crashed too much

        try:
            score_d["score"] = run.config["threshold"]
        except KeyError:
            try:
                score_d["score"] = float(run.name)
            except ValueError:
                try:
                    score_d["score"] = float(run.name.split("_")[-1])
                except ValueError as e:
                    if args.ignore_missing_score:
                        continue
                    else:
                        raise e

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
            assert len(files)>0

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
                # replace=False because these files are never modified. Save them in a unique location, ROOT/run.id
                with latest_file.download(ROOT / run.id, replace=False, exist_ok=True) as f:
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

                # Correct score_d to reflect the actual number of steps that we are collecting
                score_d["steps"] = latest_fname_step
                add_run_for_processing(AcdcRunCandidate(
                    threshold=threshold,
                    steps=score_d["steps"],
                    run=run,
                    score_d=score_d,
                    corr=corr_to_copy,
                ))

            except (wandb.CommError, requests.exceptions.HTTPError) as e:
                print(f"Error {e}, falling back to parsing output.log")
                try:
                    with run.file("output.log").download(root=ROOT / run.id, replace=False, exist_ok=True) as f:
                        log_text = f.read()
                    exp.load_from_wandb_run(log_text)
                    add_run_for_processing(AcdcRunCandidate(
                        threshold=threshold,
                        steps=score_d["steps"],
                        run=run,
                        score_d=score_d,
                        corr=deepcopy(exp.corr),
                    ))
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
                with f.download(root=this_root, replace=True, exist_ok=True) as fopen:
                    # Sadly f.download opens in text mode
                    with open(fopen.name, "rb") as fopenb:
                        edges_pth = pickle.load(fopenb)

            for (n_to, idx_to, n_from, idx_from), _effect_size in edges_pth:
                n_to = n_to.replace("hook_resid_mid", "hook_mlp_in")
                n_from = n_from.replace("hook_resid_mid", "hook_mlp_in")
                all_edges[(n_to, idx_to, n_from, idx_from)].present = True

            add_run_for_processing(AcdcRunCandidate(
                threshold=threshold,
                steps=score_d["steps"],
                run=run,
                score_d=score_d,
                corr=corr,
            ))

    # Now add the test_fns to the score_d of the remaining runs
    def all_test_fns(data: torch.Tensor) -> dict[str, float]:
        return {f"test_{name}": fn(data).item() for name, fn in things.test_metrics.items()}

    all_candidates = list(threshold_to_run_map.values())
    for candidate in all_candidates:
        test_metrics = exp.call_metric_with_corr(candidate.corr, all_test_fns, things.test_data)
        candidate.score_d.update(test_metrics)
        print(f"Added run with threshold={candidate.threshold}, n_edges={candidate.corr.count_no_edges()}")

    corrs = [(candidate.corr, candidate.score_d) for candidate in all_candidates]
    if return_ids:
        return corrs, [candidate.run.id for candidate in all_candidates]
    return corrs

#%%

if not SKIP_ACDC: # this is slow, so run once
    print(ACDC_PROJECT_NAME, ACDC_PRE_RUN_FILTER)
    acdc_corrs, ids = get_acdc_runs(None if things is None else exp, clip = 1 if TESTING else None, return_ids = True)
    assert len(acdc_corrs) > 1
    print("acdc_corrs", len(acdc_corrs))

# %%

def get_canonical_corrs(exp):
    all_present_corr = deepcopy(exp.corr)
    for e in all_present_corr.all_edges().values():
        e.present = True

    none_present_corr = deepcopy(exp.corr)
    for e in none_present_corr.all_edges().values():
        e.present = False

    output = [
        (none_present_corr, {"score": 0.0}),
        (all_present_corr, {"score": 1.0}),
    ]

    if TASK != "induction":
        output.insert(1, (deepcopy(canonical_circuit_subgraph), {"score": 0.5}))

    for corr, score_d in output:
        old_exp_corr = exp.corr
        try:
            exp.corr = corr
            exp.model.reset_hooks()
            exp.setup_model_hooks(
                add_sender_hooks=True,
                add_receiver_hooks=True,
                doing_acdc_runs=False,
            )
            for name, fn in things.test_metrics.items():
                score_d["test_"+name] = fn(exp.model(things.test_data)).item()
        finally:
            exp.corr = old_exp_corr
    return output


if not SKIP_CANONICAL:
    canonical_corrs = get_canonical_corrs(exp)

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
    corr, head_parents = None, None
    for run in filtered_runs:
        try:
            nodes_to_mask_strings = run.summary["nodes_to_mask"]
        except KeyError:
            continue
        nodes_to_mask = [parse_interpnode(s) for s in nodes_to_mask_strings]
        corr, head_parents = iterative_correspondence_from_mask(
            model = model,
            nodes_to_mask=nodes_to_mask,
            use_pos_embed = USE_POS_EMBED,
            corr=None,
            head_parents=None
        )
        score_d = {k: v for k, v in run.summary.items() if k.startswith("test")}
        score_d["steps"] = run.summary["_step"]
        score_d["score"] = run.config["lambda_reg"]
        corrs.append((deepcopy(corr), score_d))

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

    nodes_to_mask = []
    cum_score = 0.0
    test_keys = [k for k in run.summary.keys() if k.startswith("test")]
    score_d_list = list(run.scan_history(keys=test_keys, page_size=100000))
    assert len(score_d_list) == len(nodes_names_indices) + 1

    corr, head_parents = iterative_correspondence_from_mask(model=model, nodes_to_mask=[], use_pos_embed=exp.use_pos_embed)
    corrs = [(corr, {"score": 0.0, **score_d_list[0]})]
    for (nodes, hook_name, idx, score), score_d in tqdm(zip(nodes_names_indices, score_d_list[1:])):
        if score == "NaN":
            score = 0.0
        if things is None:
            corr = None
        else:
            nodes_to_mask += list(map(parse_interpnode, nodes))
            corr, head_parents = iterative_correspondence_from_mask(model=model, nodes_to_mask=nodes_to_mask, use_pos_embed=exp.use_pos_embed, corr=corr, head_parents=head_parents)
        cum_score += score
        score_d = {"score": cum_score, **score_d}
        corrs.append((deepcopy(corr), score_d))
    return corrs

if "sixteen_heads_corrs" not in locals() and not SKIP_SIXTEEN_HEADS: # this is slow, so run once
    sixteen_heads_corrs = get_sixteen_heads_corrs()
    assert len(sixteen_heads_corrs) > 1
    print("sixteen_heads_corrs", len(sixteen_heads_corrs))

#%%

methods = []

if not SKIP_CANONICAL: methods.append("CANONICAL")
if not SKIP_ACDC: methods.append("ACDC") 
if not SKIP_SP: methods.append("SP")
if not SKIP_SIXTEEN_HEADS: methods.append("16H")

#%%

# get points from correspondence
def get_points(corrs_and_scores, decreasing=True):
# corrs_and_scores = corrs
# decreasing = True
# if True:
    keys = set()
    for _, s in corrs_and_scores:
        keys.update(s.keys())

    init_point = {k: math.inf for k in keys}
    for prefix in ["edge", "node"]:
        if TASK != "induction":
            init_point[f"{prefix}_fpr"] = 0.0
            init_point[f"{prefix}_tpr"] = 0.0
            init_point[f"{prefix}_precision"] = 1.0
        init_point[f"n_{prefix}s"] = math.nan

    end_point = {k: -math.inf for k in keys}
    for prefix in ["edge", "node"]:
        if TASK != "induction":
            end_point[f"{prefix}_fpr"] = 1.0
            end_point[f"{prefix}_tpr"] = 1.0
            end_point[f"{prefix}_precision"] = 0.0
        end_point[f"n_{prefix}s"] = math.nan

    if not decreasing:
        swap = init_point
        init_point = end_point
        end_point = swap
        del swap

    points = [init_point]

    n_skipped = 0

    for idx, (corr, score) in tqdm(enumerate(sorted(corrs_and_scores, key=lambda x: x[1]["score"], reverse=decreasing))):
        if set(score.keys()) != keys:
            a = init_point.copy()
            a.update(score)
            score = a

        n_edges = corr.count_no_edges()
        n_nodes = len(filter_nodes(get_present_nodes(corr)[0]))

        score.update({"n_edges": n_edges, "n_nodes": n_nodes})

        if TASK != "induction":
            edge_stats = get_edge_stats(ground_truth=canonical_circuit_subgraph, recovered=corr)
            node_stats = get_node_stats(ground_truth=canonical_circuit_subgraph, recovered=corr)

            assert n_edges == edge_stats["recovered"]
            assert n_nodes == node_stats["recovered"]

            assert edge_stats["all"] == max_subgraph_size
            assert edge_stats["ground truth"] == canonical_circuit_subgraph_size
            assert edge_stats["recovered"] == n_edges

            for prefix, stats in [("edge", edge_stats), ("node", node_stats)]:
                assert (stats["all"] - stats["ground truth"]) == stats["false positive"] + stats["true negative"]
                assert stats["ground truth"] == stats["true positive"] + stats["false negative"]
                assert stats["recovered"] == stats["true positive"] + stats["false positive"]

                score.update(
                    {
                        f"{prefix}_tpr": stats["true positive"] / (stats["true positive"] + stats["false negative"]),
                        f"{prefix}_fpr": stats["false positive"] / (stats["false positive"] + stats["true negative"]),
                        f"{prefix}_precision": 1
                        if stats["recovered"] == 0
                        else stats["true positive"] / (stats["recovered"]),
                    }
                )

        points.append(score)
    assert n_skipped <= 2

    points.append(end_point)
    assert all(("n_edges" in p) for p in points)
    assert len(points) > 3
    return points

points = {}

#%%

if "ACDC" in methods:
    if "ACDC" not in points: points["ACDC"] = []
    points["ACDC"].extend(get_points(acdc_corrs))
#%%

if "CANONICAL" in methods:
    if "CANONICAL" not in points: points["CANONICAL"] = []
    points["CANONICAL"].extend(get_points(canonical_corrs))


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
        try: # TODO test this try block
            points[0].keys()     
    
        except:
            x = [p[0] for p in points]
            y = [p[1] for p in points]

        else:
            x=None
            y=None
            for key in points[0].keys():
                if "fpr" in key:
                    x = [p[key] for p in points]
                if "tpr" in key:
                    y = [p[key] for p in points]
            assert x is not None and y is not None, "Could not process with either indices or keys"
        
        roc_figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
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

    assert len(points[ALG]) > 3

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
