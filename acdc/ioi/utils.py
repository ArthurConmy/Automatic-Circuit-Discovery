import warnings
from functools import partial
from copy import deepcopy
import torch.nn.functional as F
from typing import List
import click
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

def get_ioi_data(tl_model, N, kl_return_one_element):
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
