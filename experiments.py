#%% [markdown]
## Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small
# <h1><b>Intro</b></h1>

# This notebook implements all experiments in our paper (which is available on arXiv).

# For background on the task, see the paper.

# Refer to the demo of the <a href="https://github.com/neelnanda-io/Easy-Transformer">Easy-Transformer</a> library here: <a href="https://github.com/neelnanda-io/Easy-Transformer/blob/main/HookedTransformer_Demo.ipynb">demo with ablation and patching</a>.
#
# Reminder of the circuit:
# <img src="https://i.imgur.com/arokEMj.png">
#%% [markdown]
## Setup
from copy import deepcopy
import torch
import matplotlib.pyplot as plt
assert torch.cuda.device_count() == 1
from tqdm import tqdm
import pandas as pd
import torch
import torch as t
from transformer_lens.HookedTransformer import (
    HookedTransformer,
)
from time import ctime
from functools import partial

import numpy as np
from tqdm import tqdm
import pandas as pd

import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import random
import einops
from IPython import get_ipython
from copy import deepcopy
# from ioi_dataset import (
#     IOIDataset,
# )
from ioi_utils import (
    path_patching,
    max_2d,
    CLASS_COLORS,
    show_pp,
    show_attention_patterns,
    scatter_attention_and_contribution,
)

from random import randint as ri

# from ioi_circuit_extraction import (
#     do_circuit_extraction,
#     get_heads_circuit,
#     CIRCUIT,
# )
# from ioi_utils import logit_diff, probs
# from ioi_utils import get_top_tokens_and_probs as g

ipython = get_ipython()
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
#%% [markdown]
# Initialise model (use larger N or fewer templates for no warnings about in-template ablation)
model = HookedTransformer.from_pretrained("gpt2").cuda()
model.set_use_attn_result(True)

#%%

days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

for day_idx in list(range(7)): # this just shows the last two cases. They are remarkably different!!!
    yesterday = days[day_idx]
    today = days[(1+day_idx)%7]
    example_prompt = f"At this time yesterday it was {yesterday}, so today it is" #@ param
    example_answer = f" {today}" #@ param

    today_index = model.tokenizer.encode(f" {today}")[0]
    yesterday_index = model.tokenizer.encode(f" {yesterday}")[0]
    print(f"Index of John token: {today_index}. Index of Mary token: {yesterday_index}")

    def get_logit_diff(logits):
    # Takes in a batch x position x vocab tensor of logits, and returns the difference between the John and Mary logit
        return logits[0, -1, today_index] - logits[0, -1, yesterday_index]

    def ablate_head_hook(value, hook, head_index):
        # Shape of value: batch x position x head_index x d_head
        value[:, -1, head_index] = 0. # just ablate at the final position
        return value

    example_logits = model(example_prompt) # Shape batch x position x vocab
    example_logit_diff = get_logit_diff(example_logits)

    head_ablation = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))
    for layer in tqdm(range(model.cfg.n_layers)):
        for head_index in range(model.cfg.n_heads):
            logits = model.run_with_hooks(example_prompt, fwd_hooks=[(f"blocks.{layer}.attn.hook_v", partial(ablate_head_hook, head_index=head_index))])
            ablated_logit_diff = get_logit_diff(logits)
            change_in_logit_diff = ablated_logit_diff - example_logit_diff #Negative = strong effect
            head_ablation[layer, head_index]=((change_in_logit_diff))
    show_pp(head_ablation.detach().numpy(), title=f'Change in logit difference when ablating each head, for sentence "{example_prompt}"', xlabel='Head', ylabel='Layer')
    print(f"Logit difference between {yesterday} and {today}")

#%%

@torch.no_grad()
def single_head_v2(model, prompts, layer, head, correct_ids, correct_mode=False, no_hooks=False):
    model.set_use_attn_result(True)
    batch_size = len(prompts)

    arr = []

    def save_head(tensor, hook):
        # saved_head = tensor[:, :, head, :].clone() # (batch_size, seq_len, d model)
        arr.append(tensor[:, :, head, :].clone())
        return tensor

    def save_mlp(tensor, hook):
        arr.append(tensor.clone())
        return tensor

    def set_out(tensor, hook):  
        # tensor.copy_(saved_head)
        print(len(arr), [a.shape for a in arr])
        assert len(arr) == 1
        # tensor.copy_(arr[0])
        tensor[:] = arr[0]
        return tensor

    model.reset_hooks()
    logits = model.run_with_hooks(
        prompts,
        fwd_hooks=[
            (f"blocks.{layer}.attn.hook_result", save_head) if head != "mlp" else (f"blocks.{layer}.hook_mlp_out", save_mlp),
            ("blocks.11.hook_resid_post", set_out),
        ] if not no_hooks else [],
        prepend_bos=True,
    )

    probs = torch.nn.functional.softmax(logits, dim=-1) # (batch_size, 1, vocab_size)
    correct_logits = logits[torch.arange(batch_size), -1, correct_ids] # (batch_size)
    correct_probs = probs[torch.arange(batch_size), -1, correct_ids]

    if correct_mode:
        return correct_logits, correct_probs
    else:
        return logits, probs

def get_top_logits(
    ls, # batch size, seq_len, vocab_size
):
    """Print the top 10 tokens and their probabilities"""

    top_logits, top_indices = torch.topk(ls[:, -1], 10, dim=-1)
    top_probs = torch.softmax(top_logits, dim=-1)
    
    for i in range(len(top_indices)):
        print(f"Top 10 tokens for sentence {i}")
        for j in range(len(top_indices[i])):
            print(f"{model.tokenizer.decode([top_indices[i][j].item()])}: {top_probs[i][j].item()}")

correct_ls, correct_ps = single_head_v2(
    model,
    prompts=["At this time yesterday it was Monday, so today it is", "At this time yesterday it was Tuesday, so today it is"],
    layer=-23480237,
    head=-2357892,
    correct_ids=[model.tokenizer.encode(" Tuesday")[0], model.tokenizer.encode(" Wednesday")[0]],
    correct_mode=True,
    no_hooks=True,
)

res = t.zeros(12, 13)

for layer in tqdm(range(12)):
    for head in range(13):
        ls, ps = single_head_v2(
            model,
            prompts=["At this time yesterday it was Monday, so today it is", "At this time yesterday it was Tuesday, so today it is"],
            layer=layer,
            head=head if head != 12 else "mlp",
            correct_ids=[model.tokenizer.encode(" Tuesday")[0], model.tokenizer.encode(" Wednesday")[0]],
            correct_mode=True,
            no_hooks=False,
        )
        
        res[layer, head] = ps[0] # - correct_ps[0]

show_pp(res, title="probs on correct when set to ...")# removing direct effect of head...")

#%%