from functools import partial
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
from typing import (
    List,
    Tuple,
    Dict,
    Any,
    Optional,
)
import warnings
import networkx as nx
from acdc.acdc_utils import (
    make_nd_dict,
    TorchIndex,
    Edge, 
    EdgeType,
    shuffle_tensor,
)  # these introduce several important classes !!!
from acdc import HookedTransformer
from acdc.acdc_utils import kl_divergence

def get_model():
    tl_model = HookedTransformer.from_pretrained(
        "redwood_attn_2l",  # load Redwood's model
        use_global_cache=True,  # use the global cache: this is needed for ACDC to work
        center_writing_weights=False,  # these are needed as this model is a Shortformer; this is a technical detail
        center_unembed=False,
        fold_ln=False,
    )

    # standard ACDC options
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

def get_all_induction_things(num_examples, seq_len, device, randomize_data=True, data_seed=42, kl_return_tensor=False, return_mask_rep=False, return_base_model_probs=False):
    tl_model = get_model()
    tl_model.to(device)

    validation_data = get_validation_data()
    mask_repeat_candidates = get_mask_repeat_candidates(num_examples=None) # None so we get all

    assert len(mask_repeat_candidates) == len(validation_data), (len(mask_repeat_candidates), len(validation_data))

    if not randomize_data:
        rand_perm = torch.arange(len(validation_data))
    else:
        if isinstance(randomize_data, int):
            torch.random.manual_seed(randomize_data)
        rand_perm = torch.randperm(len(validation_data))

    rand_perm = rand_perm[:num_examples]
    mask_repeat_candidates = mask_repeat_candidates[rand_perm][:num_examples, :seq_len]

    toks_int_values = validation_data[rand_perm][:num_examples, :seq_len].to(device).long()
    toks_int_values_other = shuffle_tensor(
        validation_data[rand_perm][:num_examples, :seq_len].to(device).long(), seed=data_seed,
    )

    base_model_logits = tl_model(toks_int_values)
    base_model_probs = F.softmax(base_model_logits, dim=-1)

    metric = partial(kl_divergence, base_model_probs=base_model_probs, mask_repeat_candidates=mask_repeat_candidates, last_seq_element_only=False, return_tensor=kl_return_tensor)

    return_list = [
        tl_model,
        toks_int_values,
        toks_int_values_other,
        metric,
    ]

    if return_mask_rep:
        return_list.append(mask_repeat_candidates)

    if return_base_model_probs:
        return_list.append(base_model_probs.cpu().clone())

    return tuple(return_list)

def one_item_per_batch(toks_int_values, toks_int_values_other, mask_rep, base_model_probs):
    """Returns each instance of induction as its own batch idx"""

    end_positions = []
    batch_size, seq_len = toks_int_values.shape
    new_tensors = []

    toks_int_values_other_batch_list = []
    new_base_model_probs_list = []

    for i in range(batch_size):
        for j in range(seq_len - 1): # -1 because we don't know what follows the last token so can't calculate losses
            if mask_rep[i, j]:
                end_positions.append(j)
                new_tensors.append(toks_int_values[i].cpu().clone())
                toks_int_values_other_batch_list.append(toks_int_values_other[i].cpu().clone())
                new_base_model_probs_list.append(base_model_probs[i].cpu().clone())

    toks_int_values_other_batch = torch.stack(toks_int_values_other_batch_list).to(toks_int_values.device)
    return_tensor = torch.stack(new_tensors).to(toks_int_values.device)
    end_positions_tensor = torch.tensor(end_positions).long()

    new_base_model_probs = torch.stack(new_base_model_probs_list)[torch.arange(len(end_positions_tensor)), end_positions_tensor].to(toks_int_values.device)
    metric = partial(kl_divergence, base_model_probs=new_base_model_probs, end_positions=end_positions_tensor, mask_repeat_candidates=None, last_seq_element_only=False, return_tensor=True)
    
    return return_tensor, toks_int_values_other_batch, end_positions_tensor, metric
