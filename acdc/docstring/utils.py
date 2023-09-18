import dataclasses
from functools import partial
from types import NoneType
import wandb
import os
from collections import defaultdict
import pickle
import torch
import huggingface_hub
import datetime
from typing import Callable, Dict
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
    MatchNLLMetric,
    make_nd_dict,
    negative_log_probs,
    shuffle_tensor,
)  # these introduce several important classes !!!

from acdc.TLACDCEdge import (
    TorchIndex,
    Edge, 
    EdgeType,
)  # these introduce several important classes !!!

from transformer_lens import HookedTransformer
from acdc.acdc_utils import kl_divergence


@dataclasses.dataclass(frozen=False)
class AllDataThings:
    tl_model: HookedTransformer
    validation_metric: Callable[[torch.Tensor], torch.Tensor]
    validation_data: torch.Tensor
    validation_labels: Optional[torch.Tensor]
    validation_mask: Optional[torch.Tensor]
    validation_patch_data: torch.Tensor
    test_metrics: dict[str, Any]
    test_data: torch.Tensor
    test_labels: Optional[torch.Tensor]
    test_mask: Optional[torch.Tensor]
    test_patch_data: torch.Tensor

def get_docstring_model(device="cuda"):
    tl_model = HookedTransformer.from_pretrained(
        "attn-only-4l",
    )
    tl_model.set_use_attn_result(True)
    tl_model.set_use_split_qkv_input(True)
    tl_model.to(device)
    return tl_model

def get_all_docstring_things(
    num_examples,
    seq_len,
    device,
    metric_name="kl_div",
    dataset_version="random_random",
    correct_incorrect_wandb=True,
    return_one_element=True,
) -> AllDataThings:
    tl_model = get_docstring_model(device=device)

    docstring_ind_prompt_kwargs = dict(
        n_matching_args=3, n_def_prefix_args=2, n_def_suffix_args=1, n_doc_prefix_args=0, met_desc_len=3, arg_desc_len=2
    )

    raw_prompts = [
        prompts.docstring_induction_prompt_generator("rest", **docstring_ind_prompt_kwargs, seed=i)
        for i in range(num_examples * 2)
    ]
    batched_prompts = prompts.BatchedPrompts(prompts=raw_prompts, model=tl_model)
    toks_int_values = batched_prompts.clean_tokens
    toks_int_values_other = batched_prompts.corrupt_tokens[dataset_version]
    toks_int_labels = batched_prompts.correct_tokens.squeeze(-1)
    toks_int_wrong_labels = batched_prompts.wrong_tokens
    assert toks_int_labels.ndim == 1
    assert toks_int_wrong_labels.ndim == 2

    validation_data = toks_int_values[:num_examples]
    validation_labels = toks_int_labels[:num_examples]
    validation_wrong_labels = toks_int_wrong_labels[:num_examples]
    validation_mask = None
    validation_patch_data = toks_int_values_other[:num_examples]

    test_data = toks_int_values[num_examples:]
    test_labels = toks_int_labels[num_examples:]
    test_wrong_labels = toks_int_wrong_labels[num_examples:]
    test_mask = None
    test_patch_data = toks_int_values_other[num_examples:]

    with torch.no_grad():
        base_validation_logprobs = F.log_softmax(tl_model(validation_data)[:, -1], dim=-1)
        base_test_logprobs = F.log_softmax(tl_model(test_data)[:, -1], dim=-1)
        assert len(base_validation_logprobs.shape) == 2, base_validation_logprobs.shape

    def raw_docstring_metric(
        logits: torch.Tensor,
        correct_labels: torch.Tensor,
        wrong_labels: torch.Tensor,
        log_correct_incorrect_wandb: bool = False,
        return_one_element: bool = True,
    ):
        """With neg sign so we minimize this"""

        correct_logits = logits[torch.arange(len(logits)), -1, correct_labels]
        incorrect_logits = logits[torch.arange(len(logits)).unsqueeze(-1), -1, wrong_labels]

        if log_correct_incorrect_wandb:
            wandb.log(
                {
                    "correct_logits": correct_logits.mean().item(),
                    "incorrect_logits": incorrect_logits.max(dim=-1).values.mean().item(),
                }
            )

        # note neg sign!!!
        answer = -(correct_logits - incorrect_logits.max(dim=-1).values)
        if return_one_element: 
            answer = answer.mean()
        return answer


    def ldgz_docstring_metric(
        logits: torch.Tensor,
        correct_labels: torch.Tensor,
        wrong_labels: torch.Tensor,
        return_one_element: bool = True,
    ):
        """Logit diff greater zero fraction (with neg sign)"""
        pos_logits = logits[:, -1, :]
        max_correct, _ = torch.gather(pos_logits, index=correct_labels[..., None], dim=1).max(dim=1)
        max_wrong, _ = torch.gather(pos_logits, index=wrong_labels, dim=1).max(dim=1)
        
        answer = -(max_correct - max_wrong > 0).float()
        if return_one_element:
            answer = answer.sum()
            answer /= len(max_correct)

        return answer

    if metric_name == "kl_div":
        validation_metric = partial(
            kl_divergence,
            base_model_logprobs=base_validation_logprobs,
            last_seq_element_only=True,
            base_model_probs_last_seq_element_only=False,
            return_one_element=return_one_element,
        )
    elif metric_name == "docstring_metric":
        validation_metric = partial(
            raw_docstring_metric,
            correct_labels=validation_labels,
            wrong_labels=validation_wrong_labels,
            log_correct_incorrect_wandb=correct_incorrect_wandb,
            return_one_element=return_one_element,
        )
    elif metric_name == "docstring_stefan":
        validation_metric = partial(
            ldgz_docstring_metric,
            correct_labels=validation_labels,
            wrong_labels=validation_wrong_labels,
            return_one_element=return_one_element,
        )
    elif metric_name == "nll":
        validation_metric = partial(
            negative_log_probs,
            labels=validation_labels,
            last_seq_element_only=True,
            return_one_element=return_one_element,
        )
    elif metric_name == "match_nll":
        validation_metric = MatchNLLMetric(
            labels=validation_labels,
            base_model_logprobs=base_validation_logprobs,
            last_seq_element_only=True,
            return_one_element=return_one_element,
        )
    else:
        raise ValueError(f"metric_name {metric_name} not recognized")


    test_metrics = {
        "kl_div": partial(
            kl_divergence,
            base_model_logprobs=base_test_logprobs,
            last_seq_element_only=True,
            base_model_probs_last_seq_element_only=False,
            return_one_element=return_one_element,
        ),
        "docstring_metric": partial(
            raw_docstring_metric,
            correct_labels=test_labels,
            wrong_labels=test_wrong_labels,
            log_correct_incorrect_wandb=correct_incorrect_wandb,
            return_one_element=return_one_element,
        ),
        "docstring_stefan": partial(
            ldgz_docstring_metric,
            correct_labels=test_labels,
            wrong_labels=test_wrong_labels,
            return_one_element=return_one_element,
        ),
        "nll": partial(
            negative_log_probs,
            labels=test_labels,
            last_seq_element_only=True,
            return_one_element=return_one_element,
        ),
        "match_nll": MatchNLLMetric(
            labels=test_labels,
            base_model_logprobs=base_test_logprobs,
            last_seq_element_only=True,
            return_one_element=return_one_element,
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

def get_docstring_subgraph_true_edges():

    # the manual graph, from Stefan

    edges_to_keep = []

    COL = TorchIndex([None])
    H = lambda i: TorchIndex([None, None, i])   

    edges_to_keep.append(("blocks.1.hook_v_input", H(4), "blocks.0.attn.hook_result", H(5)))
    edges_to_keep.append(("blocks.0.attn.hook_v", H(5), "blocks.0.hook_v_input", H(5)))
    edges_to_keep.append(("blocks.0.hook_v_input", H(5), "blocks.0.hook_resid_pre", COL))
    edges_to_keep.append(("blocks.2.attn.hook_q", H(0), "blocks.2.hook_q_input", H(0)))
    edges_to_keep.append(("blocks.2.hook_q_input", H(0), "blocks.0.hook_resid_pre", COL))
    edges_to_keep.append(("blocks.2.hook_q_input", H(0), "blocks.0.attn.hook_result", H(5)))
    edges_to_keep.append(("blocks.2.attn.hook_k", H(0), "blocks.2.hook_k_input", H(0)))
    edges_to_keep.append(("blocks.2.hook_k_input", H(0), "blocks.0.hook_resid_pre", COL))
    edges_to_keep.append(("blocks.2.hook_k_input", H(0), "blocks.0.attn.hook_result", H(5)))
    edges_to_keep.append(("blocks.2.attn.hook_v", H(0), "blocks.2.hook_v_input", H(0)))
    edges_to_keep.append(("blocks.2.hook_v_input", H(0), "blocks.1.attn.hook_result", H(4)))
    edges_to_keep.append(("blocks.1.attn.hook_v", H(4), "blocks.1.hook_v_input", H(4)))
    edges_to_keep.append(("blocks.1.hook_v_input", H(4), "blocks.0.hook_resid_pre", COL))
    edges_to_keep.append(("blocks.1.attn.hook_q", H(2), "blocks.1.hook_q_input", H(2)))
    edges_to_keep.append(("blocks.1.attn.hook_k", H(2), "blocks.1.hook_k_input", H(2)))
    edges_to_keep.append(("blocks.1.hook_q_input", H(2), "blocks.0.hook_resid_pre", COL))
    edges_to_keep.append(("blocks.1.hook_k_input", H(2), "blocks.0.hook_resid_pre", COL))
    edges_to_keep.append(("blocks.1.hook_q_input", H(2), "blocks.0.attn.hook_result", H(5)))
    edges_to_keep.append(("blocks.1.hook_k_input", H(2), "blocks.0.attn.hook_result", H(5)))

    for L3H in [H(0), H(6)]:
        edges_to_keep.append(("blocks.3.hook_resid_post", COL, "blocks.3.attn.hook_result", L3H))
        edges_to_keep.append(("blocks.3.attn.hook_q", L3H, "blocks.3.hook_q_input", L3H))
        edges_to_keep.append(("blocks.3.hook_q_input", L3H, "blocks.1.attn.hook_result", H(4)))
        edges_to_keep.append(("blocks.3.attn.hook_v", L3H, "blocks.3.hook_v_input", L3H))
        edges_to_keep.append(("blocks.3.hook_v_input", L3H, "blocks.0.hook_resid_pre", COL))
        edges_to_keep.append(("blocks.3.hook_v_input", L3H, "blocks.0.attn.hook_result", H(5)))
        edges_to_keep.append(("blocks.3.attn.hook_k", L3H, "blocks.3.hook_k_input", L3H))
        edges_to_keep.append(("blocks.3.hook_k_input", L3H, "blocks.2.attn.hook_result", H(0)))
        edges_to_keep.append(("blocks.3.hook_k_input", L3H, "blocks.1.attn.hook_result", H(2)))

    assert len(edges_to_keep) == 37, len(edges_to_keep) # reflects the value in the docstring appendix of the manual circuit as of 12th June

    # format this into the dict thing... munging ugh
    # d = {(d[0], d[1].hashable_tuple, d[2], d[3].hashable_tuple): False for d in exp.corr.all_edges()}
    d = {}

    for k in edges_to_keep:
        tupl = (k[0], k[1].hashable_tuple, k[2], k[3].hashable_tuple)
        d[tupl] = True

    return d
