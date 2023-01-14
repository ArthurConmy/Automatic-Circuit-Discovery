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
# from ioi_utils import (
#     path_patching,
#     max_2d,
#     CLASS_COLORS,
#     show_pp,
#     show_attention_patterns,
#     scatter_attention_and_contribution,
# )
import json
from random import randint as ri
ipython = get_ipython()
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
#%%
model = HookedTransformer.from_pretrained("gpt2").cuda()
model.set_use_attn_result(True)
#%%
with open("openwebtext-10k.jsonl", "r") as f:
    lines = [json.loads(l)["text"] for l in f.readlines()]
#%%
base_loss = model(lines[:10], return_type="loss", loss_per_token=True)
# %%
receiver_components = [
    [("blocks.11.hook_resid_post", None)],
]
for layer in range(11, -1, -1):    
    receiver_components.append(
        [("blocks.{}.hook_resid_mid".format(layer), None)], # MLP inputs
    )
    receiver_components.append(
        [("blocks.{}.hook_head_input".format(layer), head_idx) for head_idx in range(12)], # head inputs
    )
# TODO look at token and positional embeddings
#%%
sender_components = [[]]
for layer in range(11, -1, -1):
    sender_components.append(
        [("blocks.{}.hook_resid_mid".format(layer), None)], # MLP inputs
    )
    sender_components.append(
        [("blocks.{}.hook_attn_result".format(layer), head_idx) for head_idx in range(12)], # head inputs
    )
#%%
cache={}
model.cache_all(cache)
base_loss = model(lines[:10], return_type="loss", loss_per_token=True)
print(cache.keys())
#%%

model.reset_hooks()
for idx in range(len(receiver_components)):
    activation = None
    def save_acti