import dataclasses
from functools import partial
import wandb
import os
from collections import defaultdict
import pickle
import torch
import huggingface_hub
import datetime
from typing import Dict, Callable
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
    MatchNLLMetric,
    make_nd_dict,
    TorchIndex,
    Edge, 
    EdgeType,
    shuffle_tensor,
)  # these introduce several important classes !!!
from acdc import HookedTransformer
from acdc.acdc_utils import kl_divergence, negative_log_probs

def get_model(sixteen_heads=False):
    tl_model = HookedTransformer.from_pretrained(
        "redwood_attn_2l",  # load Redwood's model
        use_global_cache=True,  # use the global cache: this is needed for ACDC to work
        center_writing_weights=False,  # these are needed as this model is a Shortformer; this is a technical detail
        center_unembed=False,
        fold_ln=False,
        sixteen_heads=sixteen_heads,
    )

    # standard ACDC options
    tl_model.set_use_attn_result(True)
    tl_model.set_use_split_qkv_input(True) 
    return tl_model

def get_validation_data(num_examples=None, seq_len=None, device=None):
    validation_fname = huggingface_hub.hf_hub_download(
        repo_id="ArthurConmy/redwood_attn_2l", filename="validation_data.pt"
    )
    validation_data = torch.load(validation_fname, map_location=device).long()

    if num_examples is None:
        return validation_data
    else:
        return validation_data[:num_examples][:seq_len]

def get_good_induction_candidates(num_examples=None, seq_len=None, device=None):
    """Not needed?"""
    good_induction_candidates_fname = huggingface_hub.hf_hub_download(
        repo_id="ArthurConmy/redwood_attn_2l", filename="good_induction_candidates.pt"
    )
    good_induction_candidates = torch.load(good_induction_candidates_fname, map_location=device)

    if num_examples is None:
        return good_induction_candidates
    else:
        return good_induction_candidates[:num_examples][:seq_len]

def get_mask_repeat_candidates(num_examples=None, seq_len=None, device=None):
    mask_repeat_candidates_fname = huggingface_hub.hf_hub_download(
        repo_id="ArthurConmy/redwood_attn_2l", filename="mask_repeat_candidates.pkl"
    )
    mask_repeat_candidates = torch.load(mask_repeat_candidates_fname, map_location=device)
    mask_repeat_candidates.requires_grad = False

    if num_examples is None:
        return mask_repeat_candidates
    else:
        return mask_repeat_candidates[:num_examples, :seq_len]


@dataclasses.dataclass(frozen=False)
class AllInductionThings:
    tl_model: HookedTransformer
    validation_metric: Callable[[torch.Tensor], torch.Tensor]
    validation_data: torch.Tensor
    validation_labels: torch.Tensor
    validation_mask: torch.Tensor
    validation_patch_data: torch.Tensor
    test_metric: Any
    test_data: torch.Tensor
    test_labels: torch.Tensor
    test_mask: torch.Tensor
    test_patch_data: torch.Tensor


def get_all_induction_things(num_examples, seq_len, device, data_seed=42, metric="kl_div", sixteen_heads=False, return_one_element=True):
    tl_model = get_model(sixteen_heads=sixteen_heads)
    tl_model.to(device)

    validation_data_orig = get_validation_data(device=device)
    mask_orig = get_mask_repeat_candidates(num_examples=None, device=device) # None so we get all
    assert validation_data_orig.shape == mask_orig.shape

    assert seq_len <= validation_data_orig.shape[1]-1

    validation_slice = slice(0, num_examples)
    validation_data = validation_data_orig[validation_slice, :seq_len].contiguous()
    validation_labels = validation_data_orig[validation_slice, 1:seq_len+1].contiguous()
    validation_mask = mask_orig[validation_slice, :seq_len].contiguous()

    validation_patch_data = shuffle_tensor(validation_data, seed=data_seed).contiguous()

    test_slice = slice(num_examples, num_examples*2)
    test_data = validation_data_orig[test_slice, :seq_len].contiguous()
    test_labels = validation_data_orig[test_slice, 1:seq_len+1].contiguous()
    test_mask = mask_orig[test_slice, :seq_len].contiguous()

    # data_seed+1: different shuffling
    test_patch_data = shuffle_tensor(test_data, seed=data_seed).contiguous()

    with torch.no_grad():
        base_val_logprobs = F.log_softmax(tl_model(validation_data), dim=-1)
        base_test_logprobs = F.log_softmax(tl_model(test_data), dim=-1)

    if metric == "kl_div":
        validation_metric = partial(
            kl_divergence,
            base_model_logprobs=base_val_logprobs,
            mask_repeat_candidates=validation_mask,
            last_seq_element_only=False,
            return_one_element=return_one_element,
        )
        test_metric = partial(
            kl_divergence,
            base_model_logprobs=base_test_logprobs,
            mask_repeat_candidates=test_mask,
            last_seq_element_only=False,
            return_one_element=return_one_element,
        )
    elif metric == "nll":
        validation_metric = partial(
            negative_log_probs,
            labels=validation_labels,
            mask_repeat_candidates=validation_mask,
            last_seq_element_only=False,
        )
        test_metric = partial(
            negative_log_probs,
            labels=test_labels,
            mask_repeat_candidates=test_mask,
            last_seq_element_only=False,
        )
    elif metric == "match_nll":
        validation_metric = MatchNLLMetric(
            labels=validation_labels, base_model_logprobs=base_val_logprobs, mask_repeat_candidates=validation_mask,
            last_seq_element_only=False,
        )
        test_metric = MatchNLLMetric(
            labels=test_labels, base_model_logprobs=base_test_logprobs, mask_repeat_candidates=test_mask,
            last_seq_element_only=False,
        )
    else:
        raise ValueError(f"Unknown metric {metric}")

    return AllInductionThings(
        tl_model=tl_model,
        validation_metric=validation_metric,
        validation_data=validation_data,
        validation_labels=validation_labels,
        validation_mask=validation_mask,
        validation_patch_data=validation_patch_data,
        test_metric=test_metric,
        test_data=test_data,
        test_labels=test_labels,
        test_mask=test_mask,
        test_patch_data=test_patch_data,
    )


def one_item_per_batch(toks_int_values, toks_int_values_other, mask_rep, base_model_logprobs, kl_take_mean=True):
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
                new_base_model_logprobs_list.append(base_model_logprobs[i].cpu().clone())

    toks_int_values_other_batch = torch.stack(toks_int_values_other_batch_list).to(toks_int_values.device).clone()
    return_tensor = torch.stack(new_tensors).to(toks_int_values.device).clone()
    end_positions_tensor = torch.tensor(end_positions).long()

    new_base_model_logprobs = torch.stack(new_base_model_probs_list)[torch.arange(len(end_positions_tensor)), end_positions_tensor].to(toks_int_values.device).clone()
    metric = partial(
        kl_divergence, 
        base_model_logprobs=new_base_model_logprobs, 
        end_positions=end_positions_tensor, 
        mask_repeat_candidates=None, # !!! 
        last_seq_element_only=False, 
        return_one_element=False
    )
    
    return return_tensor, toks_int_values_other_batch, end_positions_tensor, metric

### WARNING: copy + paste of one loss thing per batch...

# from functools import partial
# import wandb
# import os
# from collections import defaultdict
# import pickle
# import torch
# import huggingface_hub
# import datetime
# from typing import Dict
# import torch
# import random
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import (
#     List,
#     Tuple,
#     Dict,
#     Any,
#     Optional,
# )
# import warnings
# import networkx as nx
# from acdc.acdc_utils import (
#     make_nd_dict,
#     TorchIndex,
#     Edge, 
#     EdgeType,
#     shuffle_tensor,
# )  # these introduce several important classes !!!
# from acdc import HookedTransformer
# from acdc.acdc_utils import kl_divergence

# def get_model():
#     tl_model = HookedTransformer.from_pretrained(
#         "redwood_attn_2l",  # load Redwood's model
#         use_global_cache=True,  # use the global cache: this is needed for ACDC to work
#         center_writing_weights=False,  # these are needed as this model is a Shortformer; this is a technical detail
#         center_unembed=False,
#         fold_ln=False,
#     )

#     # standard ACDC options
#     tl_model.set_use_attn_result(True)
#     tl_model.set_use_split_qkv_input(True) 
#     return tl_model

# def get_validation_data(num_examples=None, seq_len=None):
#     validation_fname = huggingface_hub.hf_hub_download(
#         repo_id="ArthurConmy/redwood_attn_2l", filename="validation_data.pt"
#     )
#     validation_data = torch.load(validation_fname)

#     if num_examples is None:
#         return validation_data
#     else:
#         return validation_data[:num_examples][:seq_len]

# def get_good_induction_candidates(num_examples=None, seq_len=None):
#     """Not needed?"""
#     good_induction_candidates_fname = huggingface_hub.hf_hub_download(
#         repo_id="ArthurConmy/redwood_attn_2l", filename="good_induction_candidates.pt"
#     )
#     good_induction_candidates = torch.load(good_induction_candidates_fname)

#     if num_examples is None:
#         return good_induction_candidates
#     else:
#         return good_induction_candidates[:num_examples][:seq_len]

# def get_mask_repeat_candidates(num_examples=None, seq_len=None):
#     mask_repeat_candidates_fname = huggingface_hub.hf_hub_download(
#         repo_id="ArthurConmy/redwood_attn_2l", filename="mask_repeat_candidates.pkl"
#     )
#     mask_repeat_candidates = torch.load(mask_repeat_candidates_fname)
#     mask_repeat_candidates.requires_grad = False

#     if num_examples is None:
#         return mask_repeat_candidates
#     else:
#         return mask_repeat_candidates[:num_examples, :seq_len]

# def get_all_induction_things(num_examples, seq_len, device, randomize_data=True, data_seed=42, kl_return_tensor=False, return_mask_rep=False, return_base_model_probs=False, kl_take_mean=True):
#     tl_model = get_model()
#     tl_model.to(device)

#     validation_data = get_validation_data()
#     mask_repeat_candidates = get_mask_repeat_candidates(num_examples=None) # None so we get all

#     assert len(mask_repeat_candidates) == len(validation_data), (len(mask_repeat_candidates), len(validation_data))

#     if not randomize_data:
#         rand_perm = torch.arange(len(validation_data))
#     else:
#         if isinstance(randomize_data, int):
#             torch.random.manual_seed(randomize_data)
#         rand_perm = torch.randperm(len(validation_data))

#     rand_perm = rand_perm[:num_examples]
#     mask_repeat_candidates = mask_repeat_candidates[rand_perm][:num_examples, :seq_len]

#     toks_int_values = validation_data[rand_perm][:num_examples, :seq_len].to(device).long()
#     toks_int_values_other = shuffle_tensor(
#         validation_data[rand_perm][:num_examples, :seq_len].to(device).long(), seed=data_seed,
#     )

#     base_model_logits = tl_model(toks_int_values)
#     base_model_probs = F.softmax(base_model_logits, dim=-1)

#     metric = partial(kl_divergence, base_model_probs=base_model_probs, mask_repeat_candidates=mask_repeat_candidates, last_seq_element_only=False, return_tensor=kl_return_tensor, take_mean=kl_take_mean)

#     return_list = [
#         tl_model,
#         toks_int_values,
#         toks_int_values_other,
#         metric,
#     ]

#     if return_mask_rep:
#         return_list.append(mask_repeat_candidates)

#     if return_base_model_probs:
#         return_list.append(base_model_probs.cpu().clone())

#     return tuple(return_list)
