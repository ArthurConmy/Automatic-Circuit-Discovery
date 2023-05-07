#%%

import IPython
if IPython.get_ipython() is not None:
    IPython.get_ipython().run_line_magic("load_ext", "autoreload")  # type: ignore
    IPython.get_ipython().run_line_magic("autoreload", "2")  # type: ignore

from copy import deepcopy
import acdc
from collections import defaultdict
from typing import List
import wandb
from acdc.graphics import show_pp
import IPython
from functools import partial
import torch
from tqdm import tqdm

import json
import pathlib
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer


import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go

PATH = "/mnt/ssd-0/arthurworkspace/TransformerLens/dist/counterfact.json"

from copy import deepcopy
from typing import (
    List,
    Tuple,
    Dict,
    Any,
    Optional,
    Union,
    Callable,
    TypeVar,
    Iterable,
    Set,
)
import pickle
import wandb
import IPython
import torch

# from easy_transformer.ioi_dataset import IOIDataset  # type: ignore
from tqdm import tqdm
import random
from functools import *
import json
import pathlib
import warnings
import time
import networkx as nx
import os
import torch
import huggingface_hub
import graphviz
from enum import Enum
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from tqdm import tqdm
import yaml
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go

pio.renderers.default = "colab"
from acdc.hook_points import HookedRootModule, HookPoint
from acdc.HookedTransformer import (
    HookedTransformer,
)
from acdc.tracr.utils import get_tracr_data, get_tracr_model_input_and_tl_model
from acdc.docstring.utils import get_all_docstring_things
from acdc.acdc_utils import (
    make_nd_dict,
    shuffle_tensor,
    cleanup,
    ct,
    TorchIndex,
    Edge,
    EdgeType,
)  # these introduce several important classes !!!

from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from acdc.TLACDCInterpNode import TLACDCInterpNode
from acdc.TLACDCExperiment import TLACDCExperiment

from collections import defaultdict, deque, OrderedDict
from acdc.acdc_utils import (
    kl_divergence,
)
from acdc.ioi.utils import (
    get_ioi_data,
    get_ioi_gpt2_small,
)
from acdc.induction.utils import (
    get_all_induction_things,
    get_model,
    get_validation_data,
    get_good_induction_candidates,
    get_mask_repeat_candidates,
)
from acdc.graphics import (
    build_colorscheme,
    show,
)
from IPython import get_ipython
import argparse
torch.autograd.set_grad_enabled(False)

#%%

# load model
model_name = "gpt2-xl"  # @param ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'facebook/opt-125m', 'facebook/opt-1.3b', 'facebook/opt-2.7b', 'facebook/opt-6.7b', 'facebook/opt-13b', 'facebook/opt-30b', 'facebook/opt-66b', 'EleutherAI/gpt-neo-125M', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-2.7B', 'EleutherAI/gpt-j-6B', 'EleutherAI/gpt-neox-20b']
model = acdc.HookedTransformer.from_pretrained(model_name, use_global_cache=True)
model.set_use_attn_result(True)
model.set_use_split_qkv_input(True)

print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
print("Reference: Hyperparameters for the model")

# In[3]:

# some util functions
def show_tokens(tokens):
    # Prints the tokens as text, separated by |
    if type(tokens) == str:
        # If we input text, tokenize first
        tokens = model.to_tokens(tokens)
    text_tokens = [model.tokenizer.decode(t) for t in tokens.squeeze()]
    print("|".join(text_tokens))

def sample_next_token(
    model: acdc.HookedTransformer, input_ids: torch.Tensor, temperature=1.0, freq_penalty=0.0, top_k=0, top_p=0.0, cache=None
) -> torch.Tensor:
    assert input_ids.ndim == 1, "input_ids should be a 1D sequence of token ids"
    model.eval()
    with torch.inference_mode():
        all_logits = model(input_ids.unsqueeze(0))  # TODO: cache
    B, S, E = all_logits.shape
    logits = all_logits[0, -1]
    return logits

# In[4]:

# sampling example
input = "The Eiffel Tower is in"
input_tokens = torch.tensor(model.tokenizer.encode(input))

logits = sample_next_token(model, input_tokens.long().to("cuda"))

values, indices = torch.topk(logits, k=20)
print(f"Model name: {model.cfg.model_name}")
print(f"Input: {input}")
print(f"token {'':<9} logits")
for i in range(20):
    print(f"{model.tokenizer.decode(indices[i]) :<15} {values[i].item()}")

print("\ngrr dumb model thinks London")

#%%

print("This can take minutes......")

# # The CounterFact dataset
import os
with open(os.path.expanduser(PATH), "rb") as f:
    counterfact = json.load(f)
ranks = []
prompts = [c["requested_rewrite"]["prompt"] for c in counterfact]
pdict = {}
for i, p in enumerate(prompts):
    if p not in pdict:
        pdict[p] = [i]
    pdict[p].append(i)

# ids = pdict["The official religion of {} is"]
ids = pdict["The mother tongue of {} is"]

lens = {i : 0 for i in range(len(ids))}
ids2 = []

for i in ids:
    data = counterfact[i]
    rr = data["requested_rewrite"]
    cur = " "+rr["subject"]
    print(cur)
    tokens = model.tokenizer.encode(cur)
    lens[len(tokens)] += 1

    if len(tokens) == 4:
        ids2.append(i)

data = []
labels = []

for datapoint in [counterfact[i] for i in ids2]:
    rr = datapoint["requested_rewrite"]
    input = rr["prompt"].format(rr["subject"])
    target = " " + rr["target_true"]["str"]
    false_target = " " + rr["target_new"]["str"]
    input_tokens = model.to_tokens(input, prepend_bos=True, move_to_device=False)[0]
    target_tokens = model.to_tokens(target, prepend_bos=True, move_to_device=False)[0]
    false_target_tokens = model.to_tokens(false_target, prepend_bos=True)[0]
    logits = sample_next_token(model, input_tokens.long())
    top_token = torch.argmax(logits).item()

    if target_tokens[-1].item() == 4302:
        target_tokens[-1] = 13624 # Christian -> Christianity, this makes more sense

    # data.append(torch.cat((input_tokens, target_tokens[1:]))) 
    data.append(input_tokens)
    labels.append(target_tokens[1:])

    # rank = torch.argsort(logits, descending=True).tolist().index(target_tokens[0])
    # ranks.append(rank)

#%%

data = torch.stack(tuple(row for row in data)).long().to("cuda")
labels = torch.stack(tuple(row for row in labels)).long().to("cuda")
labels = labels.squeeze(-1) # can't see why you left an extra dim...

if get_ipython() is None:
    length = data.shape[0] // 2 # trim size cos memory
else:
    length = data.shape[0]

data = data[:length]
labels = labels[:length]

# Make a good baseline
patch_data = model.to_tokens("The obvious feature about that person over there is", prepend_bos=True)

patch_data = patch_data[0].long().to("cuda")
patch_data = patch_data.unsqueeze(0).repeat(data.shape[0], 1)
assert patch_data.shape == data.shape, (patch_data.shape, data.shape)

#%%

print("All the facts:")
for i in range(len(data)):
    print(model.tokenizer.decode(data[i]))
seq_len = data.shape[1]
N = data.shape[0]

#%% [markdown]

# some testing that we can predict facts...
old_data = deepcopy(data).cpu() # lol the new data gets cursed!!!
logits = model(data)

#%%

original_probs = torch.nn.functional.softmax(logits, dim=-1)

# correct_log_probs = log_probs[torch.arange(len(labels)).to(log_probs.device), -1, labels.to(log_probs.device)] 
device = original_probs.device
labels = labels.to(device).view(-1, 1, 1)

# Replace 1 with 0 in the gather() call to simulate the incorrect version

labels = labels.squeeze(-1).squeeze(-1)
new_correct_probs = original_probs[torch.arange(len(labels)), -1, labels]

#%%

if get_ipython() is not None:
    # bar chart with hoverable sentences 
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[model.tokenizer.decode(old_data[i]) for i in range(len(old_data))],
            y=new_correct_probs.tolist(),
            hovertext=[model.tokenizer.decode(labels[i]) for i in range(len(data))],
        ))

# sum is roughly 3.27

#%%

relevant_positions = {
    " is": 9,
    " subject_end": 8,
    " subject_start": 8-4,
}

#%%

if get_ipython() is not None:
    def mask_attention(z, hook, key_pos):
        # print(z.shape) # batch heads query (I think) key
        assert relevant_positions[" is"] == z.shape[2]-1, (relevant_positions, z.shape)
        z[:, :, -1, key_pos] = 0

    answers = []

    for i in range(4, model.cfg.n_layers-4):
        # Reproduce Figure 2 from the paper?

        model.reset_hooks()

        for layer in range(i-4, i+5):
            model.add_hook(
                f"blocks.{layer}.attn.hook_pattern",
                partial(mask_attention, key_pos=relevant_positions[" subject_end"]),
            )

        logits = model(data)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        correct_probs = probs[torch.arange(len(labels)).to(probs.device), -1, labels.to(probs.device)]
        assert len(list(correct_probs.shape))==1, probs.shape
        answers.append(correct_probs.sum().cpu()) 

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[i for i in range(4, model.cfg.n_layers-4)],
            y=[a.cpu() for a in answers],
        ))
    # add title
    fig.update_layout(
        title_text=f"Key position end_subject",
        xaxis_title="Layer",
        yaxis_title="Sum of correct probs",
    )
    fig.show()

#%%

cache = {}
model.cache_all(cache)
model(patch_data[:1]) # same throughout

#%%
# let's move beyond zero ablation...

if get_ipython() is not None:
    answers = []

    for layer in tqdm(range(model.cfg.n_layers-1)):
        def zero_out(z, hook):
            # for pos in range(relevant_positions[" subject_start"], relevant_positions[" subject_end"]+1): 
            for pos in [-1]:
                z[:, pos] = 0.0

        def patch_out(z, hook):
            # for pos in range(relevant_positions[" subject_start"], relevant_positions[" subject_end"]+1): 
            for pos in [-1]:
                z[:, pos] = cache[hook.name][:, pos]

        # ooh quite a lot like path patch

        model.reset_hooks()
        for layer_prime in range(layer+1, model.cfg.n_layers):
            for hook_name in [
                f"blocks.{layer_prime}.hook_attn_out",
                f"blocks.{layer_prime}.hook_mlp_out",
            ]:
                model.add_hook(hook_name, patch_out)

        logits = model(data)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        correct_probs = probs[torch.arange(len(labels)).to(probs.device), -1, labels.to(probs.device)]
        # correct_log_probs = torch.log(correct_probs)
        answers.append(correct_probs.sum().cpu())

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[i for i in range(model.cfg.n_layers-1)],
            y=[a.cpu() for a in answers],
    ))

    # add title
    fig.update_layout(
        title_text=f"Key position end_subject layer {layer}",
        xaxis_title="Layer",
        yaxis_title="Sum of correct probs",
    )
    fig.show()

#%% [markdown]

def my_metric(logits):
    probs = torch.nn.functional.softmax(logits, dim=-1)
    correct_probs = probs[torch.arange(len(labels)).to(probs.device), -1, labels.to(probs.device)]
    return correct_probs.sum().item()

exp = TLACDCExperiment( # ugh so this takes... > 2min42... for sure
    model=model,
    threshold = - 100_000.0, # lol let's see all effect sizes
    using_wandb=True,
    zero_ablation=False, # 
    ds=data,
    ref_ds=patch_data,
    metric=my_metric,
    verbose=True,
    wandb_entity_name="remix_school-of-rock",
    wandb_project_name="acdc",
    wandb_run_name="my_run_new",
    # indices_mode=INDICES_MODE,
    # names_mode=NAMES_MODE,
    # second_cache_cpu=SECOND_CACHE_CPU,
    # hook_verbose=False,
    # first_cache_cpu=FIRST_CACHE_CPU,
    # add_sender_hooks=True,
    # add_receiver_hooks=False,
    # remove_redundant=False,
)

# probably this is too slow to be usable : (

#%%

while exp.current_node is not None:
    exp.step()

# %%

the_token = model.tokenizer.encode(" the")[0]

correct_direction = model.unembed.W_U[:, labels]
incorrect_direction = model.unembed.W_U[:, the_token]

cache = {}
def cacher(z, hook):
    cache[hook.name] = z.clone()
    return z

model.reset_hooks()
for layer in range(model.cfg.n_layers):
    for name in [
        f"blocks.{layer}.attn.hook_result",
        f"blocks.{layer}.hook_mlp_out",
    ]:
        model.add_hook(
            name,
            cacher,
        )
        
logits = model(data)

# %%

answers = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))

for layer in range(model.cfg.n_layers):
    results = torch.einsum(
        "bhd,db->bh",
        cache[f"blocks.{layer}.attn.hook_result"][:, -1],
        correct_direction - incorrect_direction.unsqueeze(-1),
    )
    for head in range(model.cfg.n_heads):
        answers[layer, head] = results[:, head].mean()

fig = show_pp(
    answers,
    return_fig=True,
)

# %%
