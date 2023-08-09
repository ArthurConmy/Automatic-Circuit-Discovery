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
from subnetwork_probing.train import iterative_correspondence_from_mask, correspondence_from_mask
from acdc.TLACDCInterpNode import parse_interpnode, heads_to_nodes_to_mask
import pickle
import wandb
import IPython
from acdc.docstring.utils import AllDataThings
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


@dataclass(frozen=True)
class AcdcRunCandidate:
    threshold: float
    steps: int
    run: wandb.apis.public.Run
    score_d: dict
    corr: TLACDCCorrespondence

def get_acdc_runs( # TODO ensure this is super similar to the changes from tracr-data-update branch
    exp,
    project_name: str,
    root: Path,
    things: AllDataThings,
    pre_run_filter: Optional[Dict],
    run_filter: Optional[Callable[[Any], bool]],
    clip: Optional[int] = None,
    return_ids: bool = False,
):
    if clip is None:
        clip = 100_000 # so we don't clip anything
    if pre_run_filter is None:
        pre_run_filter = {}

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
        print(run.id, "from this many mins ago:", (time.time() - run.summary["_timestamp"])/60)
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
                    print(run.id, "errorring")
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
                # replace=False because these files are never modified. Save them in a unique location, root/run.id
                with latest_file.download(root / run.id, replace=False, exist_ok=True) as f:
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
                    with run.file("output.log").download(root=root / run.id, replace=False, exist_ok=True) as f:
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

            this_root = root / edges_artifact.name
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
    def all_test_fns(data: torch.Tensor) -> Dict[str, float]:
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

# Do SP stuff
def get_sp_corrs( # TODO ensure this is super similar to the changes from tracr-data-update branch
    model: HookedTransformer,
    project_name: str,
    things: AllDataThings,
    pre_run_filter: Dict,
    run_filter: Optional[Callable[[Any], bool]],
    clip: Optional[int] = None,
    use_pos_embed = False,
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
        corr = correspondence_from_mask( # Note that we can't use iterative correspondence since we don't know that smaller subgraphs SP recovers will always be subsets of bigger subgraphs SP recovers 
            model = model,
            nodes_to_mask=nodes_to_mask,
            use_pos_embed = use_pos_embed,
        )
        score_d = {k: v for k, v in run.summary.items() if k.startswith("test")}
        score_d["steps"] = run.summary["_step"]
        score_d["score"] = run.config["lambda_reg"]
        corrs.append((deepcopy(corr), score_d))

    return corrs

def get_sixteen_heads_corrs(
    project_name: str,
    pre_run_filter: Dict,
    run_filter: Optional[Callable[[Any], bool]],
    things: AllDataThings,
    exp: TLACDCExperiment,
    model: HookedTransformer,
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

    corr, head_parents = iterative_correspondence_from_mask( # I don't understand this error
        model=model, 
        nodes_to_mask=[], 
        use_pos_embed=exp.use_pos_embed
    )

    corrs = [(corr, {"score": 0.0, **score_d_list[0]})]

    zipped_list = list(zip(nodes_names_indices, score_d_list[1:], strict=True))

    for loop_idx, ((nodes, hook_name, idx, score), score_d) in tqdm(enumerate(zipped_list)):
        if score == "NaN":
            score = 0.0
        if things is None:
            corr = None
        else:
            nodes_to_mask += list(map(parse_interpnode, nodes))
            corr = correspondence_from_mask(
                model=model, 
                nodes_to_mask=nodes_to_mask, 
                use_pos_embed=exp.use_pos_embed, 
            )

        cum_score += score
        score_d = {"score": cum_score, **score_d}
        print(corr.count_no_edges(), score_d)
        corrs.append((deepcopy(corr), score_d))
    return corrs