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
import os
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
import wandb
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
from random import randint as ri

ipython = get_ipython()
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
#%% [markdown]

# Initialise model (use larger N or fewer templates for no warnings about in-template ablation)
model = HookedTransformer.from_pretrained("gpt2", center_unembed=False, center_writing_weights=False, fold_ln=False).cuda()
model.set_use_attn_result(True)

#%%

import time
toks = model.to_tokens("Hello, my name is Arthur")

optimizer = torch.optim.Adam(model.parameters(), lr=6e-4)
 
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
def get_no_parameters(model, d, l=10):
    lis = [list(l.shape) for l in list(model.parameters()) if l.requires_grad] # [2:-4]
    print(lis)
    ans = 0
    for thing in lis[2:-4]:
        l2 = list(thing)
        for i in range(len(l2)):
            if l2[i] == 768:
                l2[i] = d
            if l2[i] == 4*768:
                l2[i] = 4*d
            if l2[i] == 768//12:
                l2[i] = d // l
        ans += np.prod(l2)
    ans *= (l/12)
    ans += 50257
#    print(sum(lis[2:-4]) * (10/12))
    return (ans + (d/768) * sum(np.prod(l) for l in lis[:2] + lis[-4:-1]))

def get_no_params_raw(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

original = 124_000_000 # get_no_parameters(model, 768, 12)
# binary search for d so that the number of parameters is 1/2 of the original

fails = 0
passes = 768
while fails + 1 < passes:
    mid = (fails + passes) // 2
    print(mid)
    if get_no_parameters(model, mid, 10) < original:
        fails = mid
    else:
        passes = mid
print(passes)

# print(sum(lis))
# return lis[0].numel(), lis[1], lis[-2]
# if p.requires_grad:
#     shap = list(p.shape)
#     print(shap)

# return list(list(p.shape) for p in model.parameters() if p.requires_grad)

# a, b, c = get_no_parameters(model, 768)
# print(a.shape, b.shape, c.shape)

# for v in [a.T, c, a.T - c]:
#     print(torch.norm(v))

# assert torch.allclose(a.T, c)

#%%

d_model = 768
n_layers = 12
n_heads = 12
assert d_model % n_heads == 0
d_head = d_model // n_heads

from transformer_lens import HookedTransformerConfig, HookedTransformer

cfg = HookedTransformerConfig.from_dict({
    "n_layers": n_layers,
    "n_heads": n_heads,
    "d_model": d_model,
    "n_ctx": 1024,
    "d_head": d_head,
    "act_fn": "gelu",
    "d_vocab": 50257,
})
trans = HookedTransformer(cfg)
trans.tokenizer = model.tokenizer

#%%

from datasets import list_datasets, load_dataset
datasets_list = list_datasets()
for s in datasets_list:
    if "pile" in s:
        print(s)
dataset = load_dataset("openwebtext")

#%%
mean=0
for i in tqdm(range(1000)):
    sample = dataset["train"][i]
    mean+=((model.to_tokens(sample["text"])).numel())
mean/=1000
print(mean)

#%%
trans.generate("There should be non-sensical completion here:")

#%%

USING_WANDB = True

if USING_WANDB:
    wandb.init(project="gpt-a", name="arthurs-run")


#%%

# To train all versions of GPT-3, we use Adam with β1 = 0.9, β2 = 0.95, and  = 10−8
# , we clip the global norm of the
# gradient at 1.0, and we use cosine decay for learning rate down to 10% of its value, over 260 billion tokens (after 260
# billion tokens, training continues at 10% of the original learning rate). There is a linear LR warmup over the first 375
# million tokens. We also gradually increase the batch size linearly from a small value (32k tokens) to the full value over
# the first 4-12 billion tokens of training, depending on the model size. Data are sampled without replacement during
# training (until an epoch boundary is reached) to minimize overfitting. All models use weight decay of 0.1 to provide a
# small amount of regularization [LH17].

#%% [markdown]
# TODO:
# - [ ] Get learning rate warmup working
# - [ ] Get gradient clipping working
# - [ ] Saving of the text of the running code.

#%% [markdown]
# Save / load in a base model (rm the fpath in orde to save)

def save_model_weights(model, path):
    torch.save(model.state_dict(), path)

fpath = "pts/initial_trans.pt"

if not os.path.exists(fpath):
    save_model_weights(trans, "pts/initial_trans.pt")

# load trans to be the model we're training

if "is_loaded" not in dir(trans):
    # load trans from pt file
    print("yee haw")
    trans.load_state_dict(torch.load(fpath))
    trans.is_loaded = True

# Adam optimizer, LR 6e-4
opt = torch.optim.Adam(
    trans.parameters(),
    lr=6e-4,
    betas=(0.9, 0.95),
    eps=1e-8,
)

#%%

ds = dataset["train"]

for step in tqdm(range(len(ds))):
    log = {}
    sample = ds[step]
    toks = model.to_tokens(sample["text"]).cuda()

    opt.zero_grad()
    loss = trans(toks, return_type="loss", loss_per_token=False) # i think that this computes mean...
    log["loss"] = loss.item()

    # loss *= loss.numel() / 1024
    loss.backward()
    opt.step()

    if USING_WANDB:
        wandb.log(log)  