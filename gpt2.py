#%%
s="""Model Name nparams nlayers dmodel nheads dhead Batch Size Learning Rate
GPT-3 Small 125M 12 768 12 64 0.5M 6.0 × 10−4
GPT-3 Medium 350M 24 1024 16 64 0.5M 3.0 × 10−4
GPT-3 Large 760M 24 1536 16 96 0.5M 2.5 × 10−4
GPT-3 XL 1.3B 24 2048 24 128 1M 2.0 × 10−4
GPT-3 2.7B 2.7B 32 2560 32 80 1M 1.6 × 10−4
GPT-3 6.7B 6.7B 32 4096 32 128 2M 1.2 × 10−4
GPT-3 13B 13.0B 40 5140 40 128 2M 1.0 × 10−4
GPT-3 175B or “GPT-3” 175.0B 96 12288 96 128 3.2M 0.6 × 10−4"""

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
model = HookedTransformer.from_pretrained("gpt2", center_unembed=False, center_writing_weights=False, fold_ln=False).cuda()
model.set_use_attn_result(True)

#%%
model.unembed.W_U = torch.nn.Parameter(model.embed.W_E.T)

#%%
import json
with open("openwebtext-10k.jsonl", "r") as f:
    lines = [json.loads(l)["text"] for l in f.readlines()]
#%%

import time

toks = model.to_tokens(lines[0])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
 
lis = []

for i in range(100):
    start_time = time.time()
    optimizer.zero_grad()
    logs = model(toks)
    loss = torch.sum(logs)
    loss.backward()
    optimizer.step()
    end_time = time.time()
    print(end_time - start_time)
    lis.append(end_time - start_time)

#%%
# def proc
def get_no_parameters(model):
    ans = 0
    lis = [l for l in list(model.parameters()) if l.requires_grad]
    # print(sum(lis))
    return lis[0], lis[1], lis[-2]
    # if p.requires_grad:
    #     shap = list(p.shape)
    #     print(shap)

    # return list(list(p.shape) for p in model.parameters() if p.requires_grad)

a, b, c = get_no_parameters(model)
print(a.shape, b.shape, c.shape)

for v in [a.T, c, a.T - c]:
    print(torch.norm(v))

assert torch.allclose(a.T, c)

#%%

b = torch.randn(2, 2)
a = b.T
print(torch.norm(b))
b+=1
print(torch.norm(b), torch.norm(a))

#%%

def get_top_logits(
    ls, # batch size, seq_len, vocab_size
):
    """Print the top 10 tokens and their probabilities"""

    top_logits, top_indices = torch.topk(ls[:, -1], 10, dim=-1)
    top_probs = torch.softmax(top_logits, dim=-1)
    
    # for i in range(len(top_indices)):
    #     print(f"Top 10 tokens for sentence {i}")
    #     for j in range(len(top_indices[i])):
    #         print(f"{model.tokenizer.decode([top_indices[i][j].item()])}: {top_probs[i][j].item()}")
    # print("Done")

@torch.no_grad()
def single_head_v2(model, prompts, layer, head, correct_ids, incorrect_ids=False, correct_mode=False, no_hooks=False):
    model.set_use_attn_result(True)
    batch_size = len(prompts)

    arr = []

    def save_head(tensor, hook):
        # saved_head = tensor[:, :, head, :].clone() # (batch_size, seq_len, d model)
        arr.append(tensor[:, :, head, :].clone())
        # print("read", torch.norm(arr[0]))
        return tensor
    


    def save_mlp(tensor, hook):
        arr.append(tensor.clone())
        return tensor

    def set_out(tensor, hook):  
        # tensor.copy_(saved_head)
        # print(len(arr), [a.shape for a in arr])
        assert len(arr) == 1
        # tensor.copy_(arr[0])
        tensor[:] -= arr[0]
        # print("write", torch.norm(arr[0]))
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
    correct_logits = logits[torch.arange(batch_size), -1, correct_ids] - (0.0 if incorrect_ids is None else logits[torch.arange(batch_size), -1, incorrect_ids]) # (batch_size)
    correct_probs = probs[torch.arange(batch_size), -1, correct_ids]
    incorrect_probs = probs[torch.arange(batch_size), -1, incorrect_ids]

    for i in range(7):
        get_top_logits(logits[i:i+1])

    if correct_mode:
        return correct_logits, correct_probs, incorrect_probs
    else:
        return logits, probs


correct_ls, correct_ps, incorrect_ps = single_head_v2(
    model,
    prompts=["If today is Monday, tomorrow is", "If today is Tuesday, tomorrow is"],
    layer=-23480237,
    head=-2357892,
    correct_ids=[model.tokenizer.encode(" Tuesday")[-1], model.tokenizer.encode(" Wednesday")[-1]],
    incorrect_ids=[model.tokenizer.encode(" Monday")[-1], model.tokenizer.encode(" Tuesday")[-1]],
    correct_mode=True,
    no_hooks=True,
)

for i in range(7):
    res = t.zeros(12, 13)
    lres = t.zeros(12, 13)
    ares = t.zeros(12, 13)

    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    days_of_week_rot = days_of_week[1:] + [days_of_week[0]]

    get_top_logits(single_head_v2(model, ["If today is Monday, tomorrow is"], layer=0, head=0, correct_ids=[model.tokenizer.encode(" Tuesday")[0]], correct_mode=False, no_hooks=True)[0])

    for layer in tqdm(range(12)):
        for head in range(13):
            ls, ps, ps2 = single_head_v2(
                model=model,
                prompts=["Today is {day}, then tomorrow is".format(day=days_of_week[i])], # for day in days_of_week],
                head=head if head != 12 else "mlp",
                correct_ids=[model.tokenizer.encode(" "+day)[0] for day in [days_of_week_rot[i]]],
                incorrect_ids=[model.tokenizer.encode(" "+day)[0] for day in [days_of_week[i]]],
                correct_mode=True,
                layer=layer,
                # head=2s42,
                no_hooks=False,
            )
            
            ares[layer, head] = ps.mean() # - correct_ps[0
            res[layer, head] = ps2.mean() # - correct_ps[0]
            lres[layer, head] = ls.item() # - correct_ls[0]
            # print()

    # variances = t.var(lres, dim=2)

    show_pp(ares, title="Probs on correct when removing effect of heads and MLPs for day " + str(days_of_week[i]), xlabel="Head (pos 12 is MLP)", ylabel="Layer") # removing direct effect of head...")
# %%
