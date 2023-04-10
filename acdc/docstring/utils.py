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
import acdc.docstring.prompts as prompts
from acdc.acdc_utils import (
    make_nd_dict,
    TorchIndex,
    Edge, 
    EdgeType,
    shuffle_tensor,
)  # these introduce several important classes !!!
from acdc import HookedTransformer
from acdc.acdc_utils import kl_divergence

def get_all_docstring_things(
    num_examples, 
    seq_len, 
    device, 
    metric_name="kl_divergence", 
    dataset_version="random_random", 
    correct_incorrect_wandb=False,
):
    tl_model = HookedTransformer.from_pretrained(
        "attn-only-4l",
        use_global_cache=True,
    )
    tl_model.set_use_attn_result(True)
    tl_model.set_use_split_qkv_input(True)
    tl_model.to(device)

    docstring_ind_prompt_kwargs = dict(
        n_matching_args=3,
        n_def_prefix_args=2,
        n_def_suffix_args=1,
        n_doc_prefix_args=0,
        met_desc_len=3,
        arg_desc_len=2
    )

    raw_prompts = [prompts.docstring_induction_prompt_generator("rest", **docstring_ind_prompt_kwargs, seed=i) for i in range(num_examples)]
    batched_prompts = prompts.BatchedPrompts(prompts=raw_prompts, model=tl_model)
    toks_int_values = batched_prompts.clean_tokens
    toks_int_values_other = batched_prompts.corrupt_tokens[dataset_version]

    base_model_logits = tl_model(toks_int_values)[:, -1]
    base_model_probs = F.softmax(base_model_logits, dim=-1)
    assert len(base_model_probs.shape) == 2, base_model_probs.shape

    kl_metric = partial(
        kl_divergence, 
        base_model_probs=base_model_probs, 
        last_seq_element_only=True,
    )

    def raw_docstring_metric(
        logits: torch.Tensor,
        log_correct_incorrect_wandb: bool = False,
    ):
        """With neg sign so we minimize this"""
        
        correct_logits = logits[torch.arange(len(logits)), -1, batched_prompts.correct_tokens.cpu().squeeze()]
        incorrect_logits = logits[torch.arange(len(logits)).unsqueeze(-1), -1, batched_prompts.wrong_tokens]
        assert incorrect_logits.shape == batched_prompts.wrong_tokens.shape, (incorrect_logits.shape, batched_prompts.wrong_tokens.shape)
        
        if log_correct_incorrect_wandb:
            wandb.log({"correct_logits": correct_logits.mean().item(), "incorrect_logits": incorrect_logits.max(dim=-1).values.mean().item()})

        # note neg sign!!!
        return - (correct_logits.mean() - incorrect_logits.max(dim=-1).values.mean()).item()

    docstring_metric = partial(raw_docstring_metric, log_correct_incorrect_wandb=correct_incorrect_wandb)
    
    if metric_name == "kl_divergence":
        metric = kl_metric
        second_metric = docstring_metric
    elif metric_name == "docstring_metric":
        metric = docstring_metric
        second_metric = kl_metric
    else:
        raise ValueError(f"metric_name {metric_name} not recognized")

    return (
        tl_model,
        toks_int_values,
        toks_int_values_other,
        metric,
        second_metric,
    )