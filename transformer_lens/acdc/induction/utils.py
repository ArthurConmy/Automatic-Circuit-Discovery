import wandb
import os
from collections import defaultdict
import pickle
import torch
import huggingface_hub
import datetime
from typing import Dict
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import warnings
import networkx as nx
from transformer_lens.acdc.utils import (
    make_nd_dict,
    TorchIndex,
    Edge, 
    EdgeType,
)  # these introduce several important classes !!!
from transformer_lens import HookedTransformer

def get_model():
    tl_model = HookedTransformer.from_pretrained(
        "redwood_attn_2l",
        use_global_cache=True,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )
    tl_model.set_use_attn_result(True)
    tl_model.set_use_split_qkv_input(True) 
    return tl_model

def get_validation_data(num_examples=None, seq_len=None):
    validation_fname = huggingface_hub.hf_hub_download(
        repo_id="ArthurConmy/redwood_attn_2l", filename="validation_data.pt"
    )
    validation_data = torch.load(validation_fname)

    if num_examples is None:
        return validation_data
    else:
        return validation_data[:num_examples][:seq_len]

def get_good_induction_candidates(num_examples=None, seq_len=None):
    """Not needed?"""
    good_induction_candidates_fname = huggingface_hub.hf_hub_download(
        repo_id="ArthurConmy/redwood_attn_2l", filename="good_induction_candidates.pt"
    )
    good_induction_candidates = torch.load(good_induction_candidates_fname)

    if num_examples is None:
        return good_induction_candidates
    else:
        return good_induction_candidates[:num_examples][:seq_len]

def get_mask_repeat_candidates(num_examples=None, seq_len=None):
    mask_repeat_candidates_fname = huggingface_hub.hf_hub_download(
        repo_id="ArthurConmy/redwood_attn_2l", filename="mask_repeat_candidates.pkl"
    )
    mask_repeat_candidates = torch.load(mask_repeat_candidates_fname)
    mask_repeat_candidates.requires_grad = False

    if num_examples is None:
        return mask_repeat_candidates
    else:
        return mask_repeat_candidates[:num_examples, :seq_len]

def kl_divergence(
    logits: torch.Tensor,
    base_model_probs: torch.Tensor,
    mask_repeat_candidates: torch.Tensor,
):
    """Compute KL divergence between base_model_probs and probs"""
    probs = F.softmax(logits, dim=-1)

    assert probs.min() >= 0.0
    assert probs.max() <= 1.0

    kl_div = (base_model_probs * (base_model_probs.log() - probs.log())).sum(dim=-1)

    assert kl_div.shape == mask_repeat_candidates.shape, (
        kl_div.shape,
        mask_repeat_candidates.shape,
    )
    kl_div = kl_div * mask_repeat_candidates.long()

    answer = (kl_div.sum() / mask_repeat_candidates.int().sum().item()).item()

    return answer