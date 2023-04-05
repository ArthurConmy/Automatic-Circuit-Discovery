import warnings
from functools import partial
from copy import deepcopy
import torch.nn.functional as F
from typing import List
import click
import IPython
from acdc.utils import kl_divergence
import torch
from acdc.ioi.ioi_dataset import IOIDataset  # NOTE: we now import this LOCALLY so it is deterministic
from tqdm import tqdm
import wandb
from acdc.HookedTransformer import HookedTransformer

def get_ioi_gpt2_small():
    tl_model = HookedTransformer.from_pretrained("gpt2", use_global_cache=True)
    tl_model.set_use_attn_result(True)
    tl_model.set_use_split_qkv_input(True)
    return tl_model

def get_ioi_data(tl_model, N):
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
    base_model_probs = F.softmax(base_model_logits, dim=-1)
    base_model_probs = base_model_probs[:, -1]
    warnings.warn("Test this!")

    metric = partial(kl_divergence, base_model_probs=base_model_probs, last_seq_element_only=True)
    return default_data, patch_data, metric