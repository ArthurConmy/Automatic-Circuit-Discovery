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

def get_act_hook(fn, alt_act=None, idx=None, dim=None, name=None, message=None):
    """Return an hook that modify the activation on the fly. alt_act (Alternative activations) is a tensor of the same shape of the z.
    E.g. It can be the mean activation or the activations on other dataset."""
    if alt_act is not None:

        def custom_hook(z, hook):
            hook.ctx["idx"] = idx
            hook.ctx["dim"] = dim
            hook.ctx["name"] = name

            if message is not None:
                print(message)

            if (
                dim is None
            ):  # mean and z have the same shape, the mean is constant along the batch dimension
                return fn(z, alt_act, hook)
            if dim == 0:
                z[idx] = fn(z[idx], alt_act[idx], hook)
            elif dim == 1:
                z[:, idx] = fn(z[:, idx], alt_act[:, idx], hook)
            elif dim == 2:
                z[:, :, idx] = fn(z[:, :, idx], alt_act[:, :, idx], hook)
            return z

    else:

        def custom_hook(z, hook):
            hook.ctx["idx"] = idx
            hook.ctx["dim"] = dim
            hook.ctx["name"] = name

            if message is not None:
                print(message)

            if dim is None:
                return fn(z, hook)
            if dim == 0:
                z[idx] = fn(z[idx], hook)
            elif dim == 1:
                z[:, idx] = fn(z[:, idx], hook)
            elif dim == 2:
                z[:, :, idx] = fn(z[:, :, idx], hook)
            return z

    return custom_hook
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
        [("blocks.{}.attn.hook_head_input".format(layer), head_idx) for head_idx in range(12)], # head inputs
    )
# TODO look at token and positional embeddings
#%%
sender_components = [
    [("blocks.11.hook_resid_post", None)],
]
for layer in range(11, -1, -1):
    sender_components.append(
        [("blocks.{}.hook_mlp_out".format(layer), None)], # MLP inputs
    )
    sender_components.append(
        [("blocks.{}.attn.hook_result".format(layer), head_idx) for head_idx in range(12)], # head inputs
    )
#%%
cache={}
model.cache_all(cache)
# model.reset_hooks()
base_loss = model(lines[10:20], return_type="loss", loss_per_token=True).detach().cpu()
print(cache.keys())
#%%
def append_to_json(filename, key, value):
    """filename: name of the json file
    key: key of the dictionary
    value: value of the dictionary"""
    with open(filename, 'r+') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data[key] = value
        file.seek(0)
        # Sets file's current position at offset.
        json.dump(file_data, file, indent = 4)

# # add to JSON file
# data = {}
# with open('data.json', 'w') as outfile:
#     json.dump(data, outfile)

model.reset_hooks()
for idx in range(len(receiver_components)):
    for name, dim_idx in receiver_components[idx]:
        answers = []
        names = []
        for sender_idx in tqdm(range(idx+1, len(sender_components))):
            for name2, dim_idx2 in sender_components[sender_idx]:
                model.reset_hooks()
                print(name2, dim_idx2, name, dim_idx)
                activation = None
                cur = []
                def saver(z, hook):
                    # nonlocal activation
                    global cur
                    cur = [z.clone()]
                    print("savin", torch.norm(z))
                    return z
                save_act_hook = get_act_hook(
                    saver, idx=dim_idx2, dim=None if dim_idx2 is None else 2, # name=name2
                )
                model.add_hook(name2, save_act_hook)
                def replacer(z, hook):
                    # nonlocal activation
                    global cur
                    print("replacin", torch.norm(cur[0]))
                    z[:] -= cur[0]
                    return z
                replace_act_hook = get_act_hook(
                    replacer, idx=dim_idx, dim=None if dim_idx is None else 2, # name=name
                )
                model.add_hook(name, replace_act_hook)
                loss = model(lines[10:20], return_type="loss", loss_per_token=True)
                append_to_json("data.json", f"{name}Z{dim_idx}Z{name2}Z{dim_idx2}", loss.mean().detach().cpu().item())
                model.reset_hooks()
                del cur
                torch.cuda.empty_cache()
#%%
if False: # extra dull stuff on plotting
    # sort the answers, and change the order of names and answers
    answers, names = zip(*sorted(zip(answers, names)))
    answers = list(answers)
    names = list(names)

    # plot the answers
    plt.figure(figsize=(10, 10))
    plt.barh(range(len(answers)), answers)
    plt.yticks(range(len(answers)), names)

    # set a white background
    ax = plt.gca()
    ax.set_facecolor("white")

    from time import ctime
    plt.title("Losses for {} {} {}".format(name, dim_idx, ctime()))
    plt.show()
    assert False
    # plt.savefig("losses_{}_{}.png".format(name, dim_idx))
    # plt.close()
#%%
def zero_ablate(z, hook):
    z[:] = 0.0
    return z

model.reset_hooks()
# model.add_hook(zero_ablate, name="blocks.11.hook_mlp_out") # , dim=None)

for i in range(3000, 10000, 10):
    model.reset_hooks()
    loss = model(lines[i:i+10], return_type="loss", loss_per_token=True).detach().cpu()
    print(loss.mean(), end=" ")
    model.add_hook("blocks.11.hook_mlp_out", zero_ablate)
    new_loss = model(lines[i:i+10], return_type="loss", loss_per_token=True).detach().cpu()
    print(new_loss.mean()) # .mean().detach().cpu().item())
    torch.cuda.empty_cache()


def ppl(model):
    with open("openwebtext-10k.jsonl", "r") as f:
        lines = [json.loads(l)["text"] for l in f.readlines()]
    model.reset_hooks()
    loss = model(lines, return_type="loss", loss_per_token=True).detach().cpu()
    return torch.exp(loss.mean())
