from collections import OrderedDict
from acdc.acdc_utils import Edge, TorchIndex, EdgeType
from acdc.TLACDCInterpNode import TLACDCInterpNode
import warnings
from functools import partial
from copy import deepcopy
import torch.nn.functional as F
from typing import List
import click
from subnetwork_probing.train import correspondence_from_mask
import IPython
from acdc.acdc_utils import kl_divergence
import torch
from acdc.ioi.ioi_dataset import IOIDataset  # NOTE: we now import this LOCALLY so it is deterministic
from tqdm import tqdm
import wandb
from acdc.HookedTransformer import HookedTransformer

def get_gpt2_small(device="cuda", sixteen_heads=False):
    tl_model = HookedTransformer.from_pretrained("gpt2", use_global_cache=True, sixteen_heads=sixteen_heads)
    tl_model = tl_model.to(device)
    tl_model.set_use_attn_result(True)
    if not sixteen_heads: # fight the OOM!
        tl_model.set_use_split_qkv_input(True)
    return tl_model

def get_ioi_gpt2_small(device="cuda", sixteen_heads=False):
    """For backwards compat"""
    return get_gpt2_small(device=device, sixteen_heads=sixteen_heads) # TODO continue adding sixteen_heads...

def get_ioi_data(tl_model, N, kl_return_one_element=True):
    ioi_dataset = IOIDataset(
        prompt_type="ABBA",
        N=N,
        nb_templates=1,
        seed = 0,
    )

    abc_dataset = (
        ioi_dataset.gen_flipped_prompts(("IO", "RAND"), seed=1)
        .gen_flipped_prompts(("S", "RAND"), seed=2)
        .gen_flipped_prompts(("S1", "RAND"), seed=3)
    )

    seq_len = ioi_dataset.toks.shape[1]
    assert seq_len == 16, f"Well, I thought ABBA #1 was 16 not {seq_len} tokens long..."

    default_data = ioi_dataset.toks.long()[:N, : seq_len - 1]
    patch_data = abc_dataset.toks.long()[:N, : seq_len - 1]

    base_model_logits = tl_model(default_data)
    base_model_logprobs = F.log_softmax(base_model_logits, dim=-1)
    base_model_logprobs = base_model_logprobs[:, -1]

    metric = partial(kl_divergence, base_model_logprobs=base_model_logprobs, last_seq_element_only=True, return_one_element=kl_return_one_element)
    return default_data, patch_data, metric

def get_ioi_true_edges(model):
    nodes_to_mask = []
    
    CIRCUIT = {
        "name mover": [
            (9, 9),  # by importance
            (10, 0),
            (9, 6),
            (10, 10),
            (10, 6),
            (10, 2),
            (10, 1),
            (11, 2),
            (9, 7),
            (9, 0),
            (11, 9),
        ],
        "negative": [(10, 7), (11, 10)],
        "s2 inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
        "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
        "duplicate token": [
            (0, 1),
            (0, 10),
            (3, 0),
            # (7, 1),
        ],  # unclear exactly what (7,1) does
        "previous token": [
            (2, 2),
            # (2, 9),
            (4, 11),
            # (4, 3),
            # (4, 7),
            # (5, 6),
            # (3, 3),
            # (3, 7),
            # (3, 6),
        ],
    }
    all_groups_of_nodes = [group for _, group in CIRCUIT.items()]
    all_nodes = [node for group in all_groups_of_nodes for node in group]
    assert len(all_nodes) == 26, len(all_nodes)

    nodes_to_mask = []

    for layer_idx in range(12):
        for head_idx in range(12):
            if (layer_idx, head_idx) not in all_nodes:
                nodes_to_mask.append(
                    TLACDCInterpNode(name=f"blocks.{layer_idx}.attn.hook_result", index = TorchIndex([None, None, head_idx]), incoming_edge_type=EdgeType.DIRECT_COMPUTATION),
                )

    corr = correspondence_from_mask(
        nodes_to_mask=nodes_to_mask,
        model = model,
    )

    # remove input -> induction heads connections
    for layer_idx, head_idx in CIRCUIT["induction"]:
        for letter in "qkv":
            corr.edges[f"blocks.{layer_idx}.hook_{letter}_input"][TorchIndex([None, None, head_idx])][f"blocks.0.hook_resid_pre"][TorchIndex([None])].present = False

    special_connections = {
        ("s2 inhibition", "name mover"),
        ("s2 inhibition", "negative"),
        ("induction", "s2 inhibition"),
        ("induction"),
        ("previous token", "induction"),
        # ("duplicate token", "induction"),
    }

    for group_name_1 in CIRCUIT.keys():
        for group_name_2 in CIRCUIT.keys():
            if group_name_1 == group_name_2:
                continue
            if (group_name_1, group_name_2) in special_connections:
                continue

            for layer_idx1, head_idx1 in CIRCUIT[group_name_1]:
                for layer_idx2, head_idx2 in CIRCUIT[group_name_2]:
                    if layer_idx1 >= layer_idx2:
                        continue # no connection..
                    for letter in "qkv":
                        corr.edges[f"blocks.{layer_idx2}.hook_{letter}_input"][TorchIndex([None, None, head_idx2])][f"blocks.{layer_idx1}.attn.hook_result"][TorchIndex([None, None, head_idx1])].present = False

    ret =  OrderedDict({(t[0], t[1].hashable_tuple, t[2], t[3].hashable_tuple): e.present for t, e in corr.all_edges().items() if e.present})
    return ret