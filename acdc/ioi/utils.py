import warnings
from functools import partial
from copy import deepcopy
import torch.nn.functional as F
from typing import List
import click
import IPython
from acdc.acdc_utils import MatchNLLMetric, frac_correct_metric, logit_diff_metric, kl_divergence, negative_log_probs
import torch
from acdc.docstring.utils import AllDocstringThings
from acdc.ioi.ioi_dataset import IOIDataset  # NOTE: we now import this LOCALLY so it is deterministic
from tqdm import tqdm
import wandb
from acdc.HookedTransformer import HookedTransformer

def get_gpt2_small(device="cuda") -> HookedTransformer:
    tl_model = HookedTransformer.from_pretrained("gpt2", use_global_cache=True)
    tl_model = tl_model.to(device)
    tl_model.set_use_attn_result(True)
    tl_model.set_use_split_qkv_input(True)
    return tl_model

def get_ioi_gpt2_small(device="cuda"):
    """For backwards compat"""
    return get_gpt2_small(device=device)

def get_all_ioi_things(num_examples, device, metric_name):
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

    default_data = ioi_dataset.toks.long()[:num_examples*2, : seq_len - 1]
    patch_data = abc_dataset.toks.long()[:num_examples*2, : seq_len - 1]
    labels = ioi_dataset.toks.long()[:num_examples*2, seq_len-1]
    wrong_labels = torch.as_tensor(ioi_dataset.s_tokenIDs[:num_examples*2], dtype=torch.long)

    assert torch.equal(labels, torch.as_tensor(ioi_dataset.io_tokenIDs, dtype=torch.long))

    validation_data = default_data[:num_examples, :]
    validation_patch_data = patch_data[:num_examples, :]
    validation_labels = labels[:num_examples]
    validation_wrong_labels = wrong_labels[:num_examples]

    test_data = default_data[num_examples:, :]
    test_patch_data = patch_data[num_examples:, :]
    test_labels = labels[num_examples:]
    test_wrong_labels = wrong_labels[num_examples:]


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

    return AllDocstringThings(
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
