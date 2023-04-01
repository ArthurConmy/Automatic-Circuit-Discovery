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

# DEVICE = "cuda:0"
# MODEL_ID = "attention_only_2"
# PRINT_CIRCUITS = True
# ACTUALLY_RUN = True
# SLOW_EXPERIMENTS = True
# EVAL_DEVICE = "cuda:0"
# MAX_MEMORY = 20000000000
# # BATCH_SIZE = 2000
# USING_WANDB = True
# MONOTONE_METRIC = "maximize"
# START_TIME = datetime.datetime.now().strftime("%a-%d%b_%H%M%S")
# PROJECT_NAME = f"induction_arthur"

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