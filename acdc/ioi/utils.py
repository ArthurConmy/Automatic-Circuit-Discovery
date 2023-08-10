from collections import OrderedDict
from dataclasses import dataclass
from acdc.TLACDCEdge import (
    Edge,
    EdgeType,
    TorchIndex,
)
from acdc.acdc_utils import filter_nodes, get_present_nodes
from acdc.TLACDCInterpNode import TLACDCInterpNode
import warnings
from functools import partial
from copy import deepcopy
import torch.nn.functional as F
from typing import List
import click
import IPython
from acdc.acdc_utils import MatchNLLMetric, frac_correct_metric, logit_diff_metric, kl_divergence, negative_log_probs
import torch
from acdc.docstring.utils import AllDataThings
from acdc.ioi.ioi_dataset import IOIDataset  # NOTE: we now import this LOCALLY so it is deterministic
from tqdm import tqdm
import wandb
from transformer_lens.HookedTransformer import HookedTransformer

def get_gpt2_small(device="cuda") -> HookedTransformer:
    tl_model = HookedTransformer.from_pretrained("gpt2")
    tl_model = tl_model.to(device)
    tl_model.set_use_attn_result(True)
    tl_model.set_use_split_qkv_input(True)
    if "use_hook_mlp_in" in tl_model.cfg.to_dict():
        tl_model.set_use_hook_mlp_in(True)
    return tl_model

def get_ioi_gpt2_small(device="cuda"):
    """For backwards compat"""
    return get_gpt2_small(device=device)

def get_all_ioi_things(num_examples, device, metric_name, kl_return_one_element=True):
    tl_model = get_gpt2_small(device=device)
    ioi_dataset = IOIDataset(
        prompt_type="ABBA",
        N=num_examples*2,
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

    default_data = ioi_dataset.toks.long()[:num_examples*2, : seq_len - 1].to(device)
    patch_data = abc_dataset.toks.long()[:num_examples*2, : seq_len - 1].to(device)
    labels = ioi_dataset.toks.long()[:num_examples*2, seq_len-1]
    wrong_labels = torch.as_tensor(ioi_dataset.s_tokenIDs[:num_examples*2], dtype=torch.long, device=device)

    assert torch.equal(labels, torch.as_tensor(ioi_dataset.io_tokenIDs, dtype=torch.long))
    labels = labels.to(device)

    validation_data = default_data[:num_examples, :]
    validation_patch_data = patch_data[:num_examples, :]
    validation_labels = labels[:num_examples]
    validation_wrong_labels = wrong_labels[:num_examples]

    test_data = default_data[num_examples:, :]
    test_patch_data = patch_data[num_examples:, :]
    test_labels = labels[num_examples:]
    test_wrong_labels = wrong_labels[num_examples:]


    with torch.no_grad():
        base_model_logits = tl_model(default_data)[:, -1, :]
        base_model_logprobs = F.log_softmax(base_model_logits, dim=-1)

    base_validation_logprobs = base_model_logprobs[:num_examples, :]
    base_test_logprobs = base_model_logprobs[num_examples:, :]


    if metric_name == "kl_div":
        validation_metric = partial(
            kl_divergence,
            base_model_logprobs=base_validation_logprobs,
            last_seq_element_only=True,
            base_model_probs_last_seq_element_only=False,
            return_one_element=kl_return_one_element,
        )
    elif metric_name == "logit_diff":
        validation_metric = partial(
            logit_diff_metric,
            correct_labels=validation_labels,
            wrong_labels=validation_wrong_labels,
        )
    elif metric_name == "frac_correct":
        validation_metric = partial(
            frac_correct_metric,
            correct_labels=validation_labels,
            wrong_labels=validation_wrong_labels,
        )
    elif metric_name == "nll":
        validation_metric = partial(
            negative_log_probs,
            labels=validation_labels,
            last_seq_element_only=True,
        )
    elif metric_name == "match_nll":
        validation_metric = MatchNLLMetric(
            labels=validation_labels,
            base_model_logprobs=base_validation_logprobs,
            last_seq_element_only=True,
        )
    else:
        raise ValueError(f"metric_name {metric_name} not recognized")

    test_metrics = {
        "kl_div": partial(
            kl_divergence,
            base_model_logprobs=base_test_logprobs,
            last_seq_element_only=True,
            base_model_probs_last_seq_element_only=False,
        ),
        "logit_diff": partial(
            logit_diff_metric,
            correct_labels=test_labels,
            wrong_labels=test_wrong_labels,
        ),
        "frac_correct": partial(
            frac_correct_metric,
            correct_labels=test_labels,
            wrong_labels=test_wrong_labels,
        ),
        "nll": partial(
            negative_log_probs,
            labels=test_labels,
            last_seq_element_only=True,
        ),
        "match_nll": MatchNLLMetric(
            labels=test_labels,
            base_model_logprobs=base_test_logprobs,
            last_seq_element_only=True,
        ),
    }

    return AllDataThings(
        tl_model=tl_model,
        validation_metric=validation_metric,
        validation_data=validation_data,
        validation_labels=validation_labels,
        validation_mask=None,
        validation_patch_data=validation_patch_data,
        test_metrics=test_metrics,
        test_data=test_data,
        test_labels=test_labels,
        test_mask=None,
        test_patch_data=test_patch_data,
    )

IOI_CIRCUIT = {
    "name mover": [
        (9, 9),  # by importance
        (10, 0),
        (9, 6),
    ],
    "backup name mover": [
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

@dataclass(frozen=True)
class Conn:
    inp: str
    out: str
    qkv: tuple[str, ...]

def get_ioi_true_edges(model):
    nodes_to_mask = []
    
    all_groups_of_nodes = [group for _, group in IOI_CIRCUIT.items()]
    all_nodes = [node for group in all_groups_of_nodes for node in group]
    assert len(all_nodes) == 26, len(all_nodes)

    nodes_to_mask = []

    for layer_idx in range(12):
        for head_idx in range(12):
            if (layer_idx, head_idx) not in all_nodes:
                for letter in ["q", "k", "v"]:
                    nodes_to_mask.append(
                        TLACDCInterpNode(name=f"blocks.{layer_idx}.attn.hook_{letter}", index = TorchIndex([None, None, head_idx]), incoming_edge_type=EdgeType.DIRECT_COMPUTATION),
                    )


    from subnetwork_probing.train import iterative_correspondence_from_mask
    corr, _ = iterative_correspondence_from_mask(
        nodes_to_mask = nodes_to_mask,
        model = model,
    )

    # For all heads...
    for layer_idx, head_idx in all_nodes:
        for letter in "qkv":
            # remove input -> head connection
            edge_to = corr.edges[f"blocks.{layer_idx}.hook_{letter}_input"][TorchIndex([None, None, head_idx])]
            edge_to[f"blocks.0.hook_resid_pre"][TorchIndex([None])].present = False

            # Remove all other_head->this_head connections in the circuit
            for layer_from in range(layer_idx):
                for head_from in range(12):
                    edge_to[f"blocks.{layer_from}.attn.hook_result"][TorchIndex([None, None, head_from])].present = False

            # Remove connection from this head to the output
            corr.edges["blocks.11.hook_resid_post"][TorchIndex([None])][f"blocks.{layer_idx}.attn.hook_result"][TorchIndex([None, None, head_idx])].present = False



    special_connections: set[Conn] = {
        Conn("INPUT", "previous token", ("q", "k", "v")),
        Conn("INPUT", "duplicate token", ("q", "k", "v")),
        Conn("INPUT", "s2 inhibition", ("q",)),
        Conn("INPUT", "negative", ("k", "v")),
        Conn("INPUT", "name mover", ("k", "v")),
        Conn("INPUT", "backup name mover", ("k", "v")),
        Conn("previous token", "induction", ("k", "v")),
        Conn("induction", "s2 inhibition", ("k", "v")),
        Conn("duplicate token", "s2 inhibition", ("k", "v")),
        Conn("s2 inhibition", "negative", ("q",)),
        Conn("s2 inhibition", "name mover", ("q",)),
        Conn("s2 inhibition", "backup name mover", ("q",)),
        Conn("negative", "OUTPUT", ()),
        Conn("name mover", "OUTPUT", ()),
        Conn("backup name mover", "OUTPUT", ()),
    }

    for conn in special_connections:
        if conn.inp == "INPUT":
            idx_from = [(-1, "blocks.0.hook_resid_pre", TorchIndex([None]))]
            for mlp_layer_idx in range(12):
                idx_from.append((mlp_layer_idx, f"blocks.{mlp_layer_idx}.hook_mlp_out", TorchIndex([None])))
        else:
            idx_from = [(layer_idx, f"blocks.{layer_idx}.attn.hook_result", TorchIndex([None, None, head_idx])) for layer_idx, head_idx in IOI_CIRCUIT[conn.inp]]

        if conn.out == "OUTPUT":
            idx_to = [(13, "blocks.11.hook_resid_post", TorchIndex([None]))]
            for mlp_layer_idx in range(12):
                idx_to.append((mlp_layer_idx, f"blocks.{mlp_layer_idx}.hook_mlp_in", TorchIndex([None])))
        else:
            idx_to = [
                (layer_idx, f"blocks.{layer_idx}.hook_{letter}_input", TorchIndex([None, None, head_idx]))
                for layer_idx, head_idx in IOI_CIRCUIT[conn.out]
                for letter in conn.qkv
            ]

        for layer_from, layer_name_from, which_idx_from in idx_from:
            for layer_to, layer_name_to, which_idx_to in idx_to:
                if layer_to > layer_from:
                    corr.edges[layer_name_to][which_idx_to][layer_name_from][which_idx_from].present = True

    ret =  OrderedDict({(t[0], t[1].hashable_tuple, t[2], t[3].hashable_tuple): e.present for t, e in corr.all_edges().items() if e.present})
    return ret


GROUP_COLORS = {
    "name mover": "#d7f8ee",
    "backup name mover": "#e7f2da",
    "negative": "#fee7d5",
    "s2 inhibition": "#ececf5",
    "induction": "#fff6db",
    "duplicate token": "#fad6e9",
    "previous token": "#f9ecd7",
}
MLP_COLOR = "#f0f0f0"

def ioi_group_colorscheme():
    assert set(GROUP_COLORS.keys()) == set(IOI_CIRCUIT.keys())

    scheme = {
        "embed": "#cbd5e8",
        "<resid_post>": "#fff2ae",
    }

    for i in range(12):
        scheme[f"<m{i}>"] = MLP_COLOR

    for k, heads in IOI_CIRCUIT.items():
        for (layer, head) in heads:
            for qkv in ["", "_q", "_k", "_v"]:
                scheme[f"<a{layer}.{head}{qkv}>"] = GROUP_COLORS[k]

    for layer in range(12):
        scheme[f"<m{layer}>"] = "#f0f0f0"
    return scheme
