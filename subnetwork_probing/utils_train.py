import argparse
import random
from collections import defaultdict
from copy import deepcopy
import warnings
from functools import partial
import sys
from pathlib import Path
from typing import Union, Dict, List, Tuple
import collections
from acdc.greaterthan.utils import get_all_greaterthan_things
from acdc.ioi.utils import get_all_ioi_things
import huggingface_hub
import gc

import networkx as nx
import numpy as np
from acdc.docstring.utils import AllDataThings, get_all_docstring_things
import pandas as pd
import torch
import torch.nn.functional as F
import subnetwork_probing.transformer_lens.transformer_lens.utils as utils
from acdc.tracr_task.utils import get_all_tracr_things
from acdc.acdc_utils import reset_network
from acdc.TLACDCEdge import (
    TorchIndex,
    Edge,
    EdgeType,
)
from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from acdc.TLACDCInterpNode import TLACDCInterpNode
from acdc.TLACDCExperiment import TLACDCExperiment
from acdc.induction.utils import get_all_induction_things, get_mask_repeat_candidates
from tqdm import tqdm
from subnetwork_probing.transformer_lens.transformer_lens.HookedTransformer import HookedTransformer as SPHookedTransformer
from subnetwork_probing.transformer_lens.transformer_lens.HookedTransformerConfig import HookedTransformerConfig as SPHookedTransformerConfig
from transformer_lens.HookedTransformer import HookedTransformer
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from subnetwork_probing.transformer_lens.transformer_lens.ioi_dataset import IOIDataset
import wandb
torch.set_grad_enabled(True)

def log_plotly_bar_chart(x: List[str], y: List[float], log_title="mask_scores", **kwargs) -> None:
    import plotly.graph_objects as go

    title = None if "title" not in kwargs else kwargs.pop("title")

    fig = go.Figure(data=[go.Bar(x=x, y=y)], **kwargs)
    fig.update_layout(title=title)
    wandb.log({log_title: fig})


def visualize_mask(model: HookedTransformer) -> tuple[int, list[TLACDCInterpNode]]:
    number_of_heads = model.cfg.n_heads
    number_of_layers = model.cfg.n_layers
    node_name_list = []
    mask_scores_for_names = []
    total_nodes = 0
    nodes_to_mask: list[TLACDCInterpNode] = []
    for layer_index, layer in enumerate(model.blocks):
        for head_index in range(number_of_heads):
            for q_k_v in ["q", "k", "v"]:
                total_nodes += 1
                if q_k_v == "q":
                    mask_sample = (
                        layer.attn.hook_q.sample_mask()[head_index].cpu().item()
                    )
                elif q_k_v == "k":
                    mask_sample = (
                        layer.attn.hook_k.sample_mask()[head_index].cpu().item()
                    )
                elif q_k_v == "v":
                    mask_sample = (
                        layer.attn.hook_v.sample_mask()[head_index].cpu().item()
                    )
                else:
                    raise ValueError(f"{q_k_v=} must be q, k, or v")

                node_name = f"blocks.{layer_index}.attn.hook_{q_k_v}"
                node_name_with_index = f"{node_name}[{head_index}]"
                node_name_list.append(node_name_with_index)
                node = TLACDCInterpNode(node_name, TorchIndex((None, None, head_index)),
                                        incoming_edge_type=EdgeType.ADDITION)

                mask_scores_for_names.append(mask_sample)
                if mask_sample < 0.5:
                    nodes_to_mask.append(node)

        # MLP
        for node_name, edge_type in [
            (f"blocks.{layer_index}.hook_mlp_out", EdgeType.PLACEHOLDER),
            (f"blocks.{layer_index}.hook_resid_mid", EdgeType.ADDITION),
        ]:
            node_name_list.append(node_name)
            node = TLACDCInterpNode(node_name, TorchIndex([None]), incoming_edge_type=edge_type)
            total_nodes += 1

        mask_sample = layer.hook_mlp_out.sample_mask().cpu().item()
        mask_scores_for_names.append(mask_sample)
        if mask_sample < 0.5:
            nodes_to_mask.append(node)

    # assert len(mask_scores_for_names) == 3 * number_of_heads * number_of_layers
    log_plotly_bar_chart(x=node_name_list, y=mask_scores_for_names)
    node_count = total_nodes - len(nodes_to_mask)
    return node_count, nodes_to_mask

def experiment_visualize_mask(
    exp: TLACDCExperiment,
) -> Tuple[int, list[TLACDCInterpNode]]:
    """The same as visualize_mask, but for experiments (used for EdgeSP).
    WARNING: we visualize scores since that seems more principled..."""

    # Do a check for now that all the scores for same parents
    
    masks_for_parents = defaultdict(dict)
    mask_scores_for_parents = defaultdict(float)
    all_edges = exp.corr.all_edges()

    for (child_name, child_index, parent_name, parent_index), edge in all_edges.items():
        if edge.edge_type == EdgeType.PLACEHOLDER:
            continue
        try:
            masks_for_parents[(parent_name, parent_index)][(child_name, child_index)] = edge.mask.item()
        except AttributeError as e:
            print(child_name, child_index, parent_name, parent_index)
            raise e
        
        mask_scores_for_parents[(parent_name, parent_index)] = edge.mask_score.item()

    if exp.sp == "node":
        for parent_tuple in masks_for_parents:
            assert (
                torch.tensor(list(masks_for_parents[parent_tuple].values())) - list(masks_for_parents[parent_tuple].values())[0]
            ).norm().item() < 1e-2, f"{parent_tuple=} {masks_for_parents[parent_tuple]=}"

        log_plotly_bar_chart(
            x = [str(parent_tuple) for parent_tuple in masks_for_parents],
            y = [list(masks_for_parents[parent_tuple].values())[0] for parent_tuple in masks_for_parents],
            title="Mask values for parents",
            log_title="Mask values for parents",
        )

        log_plotly_bar_chart(
            x = [str(parent_tuple) for parent_tuple in masks_for_parents],
            y = [mask_scores_for_parents[parent_tuple] for parent_tuple in masks_for_parents],
            title="Mask scores for parents",
            log_title="Mask scores for parents",
        )

    elif exp.sp == "edge":
        log_plotly_bar_chart(
            x = [str(edge_tuple) for edge_tuple, e in all_edges.items() if e.edge_type != EdgeType.PLACEHOLDER],
            y = [e.mask.item() for _, e in all_edges.items() if e.edge_type != EdgeType.PLACEHOLDER],
            title="Mask values for parents",
            log_title="Mask values for parents",
        )

        log_plotly_bar_chart(
            x = [str(edge_tuple) for edge_tuple, e in all_edges.items() if e.edge_type != EdgeType.PLACEHOLDER],
            y = [e.mask_score.item() for _, e in all_edges.items() if e.edge_type != EdgeType.PLACEHOLDER],
            title="Mask scores for parents",
            log_title="Mask scores for parents",
        )

    return -1, [] # not done this yet...


def get_nodes_mask_dict(model: SPHookedTransformer):
    number_of_heads = model.cfg.n_heads
    number_of_layers = model.cfg.n_layers
    mask_value_dict = {}
    for layer_index, layer in enumerate(model.blocks):
        for head_index in range(number_of_heads):
            for q_k_v in ["q", "k", "v"]:
                # total_nodes += 1
                if q_k_v == "q":
                    mask_value = (
                        layer.attn.hook_q.sample_mask()[head_index].cpu().item()
                    )
                if q_k_v == "k":
                    mask_value = (
                        layer.attn.hook_k.sample_mask()[head_index].cpu().item()
                    )
                if q_k_v == "v":
                    mask_value = (
                        layer.attn.hook_v.sample_mask()[head_index].cpu().item()
                    )
                mask_value_dict[f"{layer_index}.{head_index}.{q_k_v}"] = mask_value
    return mask_value_dict