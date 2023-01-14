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

def get_losses(model, list_of_texts):
    losses = []
    for texts in list_of_texts:
        losses.append(model(texts, return_type="loss", loss_per_token=True).detach().cpu().mean())
    return sum(losses)/len(losses)

model.reset_hooks()
base_loss = get_losses(model, lines[10:20]) #  model(lines[10:20], return_type="loss", loss_per_token=True).detach().cpu()

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
                # tokens = model.to_tokens(lines[10:20])
                # ismask = tokens[:, 1:] == model.mask_token_id
                # loss = model(lines[10:20], return_type="loss", loss_per_token=True)
                loss = get_losses(model, lines[10:20])

                append_to_json("data2.json", f"{name}Z{dim_idx}Z{name2}Z{dim_idx2}", loss.mean().detach().cpu().item())
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

#%%
# DATA SHIT
print("WARN: not using Ryan stuff")
data_rrfs = os.path.expanduser(f"~/rrfs/pretraining_datasets/owt_tokens_int16_val/0.pt")
data_suffix = "name_data/data-2022-07-30.pt"
data_local = os.path.expanduser(f"~/{data_suffix}")
try:
    data_full = torch.load(data_local)
except FileNotFoundError:
    data_full = torch.load(data_rrfs)
toks = data_full["tokens"].long() + 32768
lens = data_full["lens"].long()


def d(tokens, tokenizer=model.tokenizer):
    return tokenizer.decode(tokens)


if False:
    SEQ_LEN = 10
    print(f"WARN: SEQ_LEN = {SEQ_LEN}")
    MODEL_ID = "attention_only_four_layers_untied"
    MODEL_ID = "attention_only_two_layers_untied"
    # MODEL_ID = "jan5_attn_only_two_layers"
    DATASET_SIZE = 8000  # total data points is twice this...
    DATASET_DIR = PP("/home/arthur/rrfs/arthur/induction/data7/")
    MODIFY_DATASETS = False
    TRIM_TO_SIZE = False
    FIND_SAME_TOKEN = False

    DATASET_PATH = DATASET_DIR / "ind.pt"
    MADE_DATA = os.path.exists(DATASET_PATH)
    VOCAB_SIZE = 50259
    if os.path.exists(DATASET_PATH):
        smol = torch.load(str(DATASET_PATH))
        print("Trying to decode ...")
        print(d(smol[0, :]))
        print("... done.")
    else:
        print("Rip, no smol found")
        if not os.path.exists(DATASET_DIR):
            print(f"Made {str(DATASET_DIR)}")
            os.mkdir(DATASET_DIR)

#%% [markdown]
# Check that the model BPBs roughly agree with https://arxiv.org/pdf/2101.00027v1.pdf page 8


def perplexity(losses):
    return torch.exp(torch.mean(losses))


def bpb(losses):
    """Cursed EleutherAI value"""
    return (0.29335 / np.log(2)) * losses


def get_loss(model, tokens, return_per_token=False):
    losses = model(
        tokens,
        return_type="loss",
        loss_per_token=return_per_token,        
    )
    return losses


model_name_list = [
    "gpt2",
    "EleutherAI/gpt-neo-125M",
    "gpt2-large",
    "EleutherAI/gpt-neo-1.3B",
    "gpt2-xl",
    "EleutherAI/gpt-neo-2.7B",
]


def check_some_losses(model, toks, lens, samples=100, manual_eos=None):
    # model = EasyTransformer.from_pretrained(model_name).cuda()
    # model.set_use_attn_result(True)
    loss_list = []
    for idx in tqdm(range(samples)):
        cur = torch.cat(
            (
                torch.tensor([model.tokenizer.pad_token_id])
                if manual_eos is None
                else torch.tensor([manual_eos]),
                toks[torch.sum(lens[:idx]) : torch.sum(lens[: idx + 1])],
            )
        )
        cur_tokens = cur.unsqueeze(0)[:, :1024]
        cur_tokens[:, 0] = model.tokenizer.pad_token_id
        loss = get_loss(model, cur_tokens, return_per_token=True)

        model.reset_hooks()
        model.add_hook("blocks.11.hook_mlp_out", zero_ablate)
        new_loss = get_loss(model, cur_tokens, return_per_token=True)
        model.reset_hooks()

        print(loss.mean().item(), new_loss.mean().item(), loss.shape)

# %%
check_some_losses(model, toks, lens, samples=100, manual_eos=50256)