#%% [markdown]
# # GPT-2 small Indirect Object Identification
# <h1><b>Intro</b></h1>
# This notebook is an implementation of the IOI experiments (all from the <a href="https://docs.google.com/presentation/d/19H__CYCBL5F3M-UaBB-685J-AuJZNsXqIXZR-O4j9J8/edit#slide=id.g14659e4d87a_0_290">paper</a>.
# It should be able to be run as by just git cloning this repo (+ some easy installs).
#
# ### Task
# We're interested in GPT-2 ability to complete sentences like "Alice and Bob went to the store, Alice gave a bottle of milk to"...
# GPT-2 knows that it have to output a name that is not the subject (Alice) and that was present in the context: Bob.
# The first apparition of Alice is called "S" (or sometimes "S1") for "Subject", and Bob is the indirect object ("IO"). Even if the sentences we generate contains the last word of the sentence "Bob", we'll never look at the transformer output here. What's matter is the next-token prediction on the token "to", sometime called the "end" token.
#
# ### Tools
# In this notebook, we define a class `IOIDataset` to handle the generation and utils for this particular dataset.
#
# Refer to the demo of the [`easy_transformer` library](https://github.com/neelnanda-io/Easy-Transformer) here: <a href="https://colab.research.google.com/drive/1MLwJ7P94cizVs2LD8Qwi-vLGSoH-cHxq?usp=sharing">demo with ablation & patching</a>.
#
# Reminder of the circuit:
# <img src="https://i.imgur.com/PPtTQRh.png">

# ## Imports
import abc
from csv import excel
from imp import init_builtin
import os
import time
from turtle import width

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch

assert torch.cuda.device_count() == 1
from ioi_utils import all_subsets
from ioi_circuit_extraction import (
    CIRCUIT,
    RELEVANT_TOKENS,
    do_circuit_extraction,
    get_heads_circuit,
)
from copy import deepcopy
from time import ctime
import io
from random import randint as ri
from easy_transformer import EasyTransformer
from functools import partial
from ioi_utils import all_subsets, logit_diff, probs, show_attention_patterns
from ioi_dataset import BABA_EARLY_IOS, BABA_LATE_IOS, ABBA_EARLY_IOS, ABBA_LATE_IOS
import logging
import sys
from ioi_dataset import *
from ioi_utils import max_2d
import IPython
from tqdm import tqdm
import pandas as pd
import torch
import torch as t
from easy_transformer.utils import (
    gelu_new,
    to_numpy,
    get_corner,
    print_gpu_mem,
)  # helper functions
from easy_transformer.hook_points import HookedRootModule, HookPoint
from easy_transformer.EasyTransformer import (
    EasyTransformer,
    TransformerBlock,
    MLP,
    Attention,
    LayerNormPre,
    PosEmbed,
    Unembed,
    Embed,
)
from easy_transformer.experiments import (
    ExperimentMetric,
    AblationConfig,
    EasyAblation,
    EasyPatching,
    PatchingConfig,
    get_act_hook,
)
from typing import Any, Callable, Dict, List, Set, Tuple, Union, Optional, Iterable
import itertools
import numpy as np
from tqdm import tqdm
import pandas as pd
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly
from sklearn.linear_model import LinearRegression
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import spacy
import re
from einops import rearrange
import einops
from pprint import pprint
import gc
from datasets import load_dataset
from IPython import get_ipython
import matplotlib.pyplot as plt
import random as rd

from ioi_dataset import (
    IOIDataset,
    NOUNS_DICT,
    NAMES,
    gen_prompt_uniform,
    BABA_TEMPLATES,
    BABA_EARLY_IOS,
    BABA_LATE_IOS,
    ABBA_TEMPLATES,
)
from ioi_utils import (
    attention_on_token,
    # patch_positions,
    clear_gpu_mem,
    show_tokens,
    show_pp,
    # show_attention_patterns,
    safe_del,
)

ipython = get_ipython()
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")


def e(mess=""):
    print_gpu_mem(mess)
    torch.cuda.empty_cache()


#%% [markdown] The model, and loads and loads of datasets
model = EasyTransformer.from_pretrained("gpt2").cuda()
model.set_use_attn_result(True)

N = 100
ioi_dataset = IOIDataset(prompt_type="mixed", N=N, tokenizer=model.tokenizer)
abca_dataset = ioi_dataset.gen_flipped_prompts(
    ("S2", "RAND")
)  # we flip the second b for a random c
abca_dataset.word_idx = ioi_dataset.word_idx
acca_dataset = ioi_dataset.gen_flipped_prompts(("S", "RAND"))
dcc_dataset = acca_dataset.gen_flipped_prompts(("IO", "RAND"))
cbb_dataset = ioi_dataset.gen_flipped_prompts(("IO", "RAND"))
acba_dataset = ioi_dataset.gen_flipped_prompts(("S1", "RAND"))
adea_dataset = ioi_dataset.gen_flipped_prompts(("S", "RAND")).gen_flipped_prompts(
    ("S1", "RAND")
)
totally_diff_dataset = IOIDataset(N=ioi_dataset.N, prompt_type=ioi_dataset.prompt_type)
all_diff_dataset = (
    ioi_dataset.gen_flipped_prompts(("IO", "RAND"))
    .gen_flipped_prompts(("S", "RAND"))
    .gen_flipped_prompts(("S1", "RAND"), manual_word_idx=ioi_dataset.word_idx)
)
bcca_dataset = ioi_dataset.gen_flipped_prompts(("IO", "RAND")).gen_flipped_prompts(
    ("S", "RAND")
)
from ioi_utils import logit_diff

ABCm_dataset = IOIDataset(prompt_type="ABC mixed", N=N, tokenizer=model.tokenizer)
ABC_dataset = IOIDataset(prompt_type="ABC", N=N, tokenizer=model.tokenizer)
BAC_dataset = IOIDataset("BAC", N, model.tokenizer)
mixed_dataset = IOIDataset("ABC mixed", N, model.tokenizer)

baba_dataset = IOIDataset(N=100, prompt_type="BABA")
abba_dataset = IOIDataset(N=100, prompt_type="ABBA")

baba_all_diff = (
    baba_dataset.gen_flipped_prompts(("IO", "RAND"))
    .gen_flipped_prompts(("S", "RAND"))
    .gen_flipped_prompts(("S1", "RAND"), manual_word_idx=baba_dataset.word_idx)
)
abba_all_diff = (
    abba_dataset.gen_flipped_prompts(("IO", "RAND"))
    .gen_flipped_prompts(("S", "RAND"))
    .gen_flipped_prompts(("S1", "RAND"), manual_word_idx=abba_dataset.word_idx)
)

circuit = deepcopy(CIRCUIT)

#%%
def patch_positions(z, source_act, hook, positions=["end"]):
    for pos in positions:
        z[torch.arange(ioi_dataset.N), ioi_dataset.word_idx[pos]] = source_act[
            torch.arange(ioi_dataset.N), ioi_dataset.word_idx[pos]
        ]
    return z


def patch_all(z, source_act, hook):
    return source_act


#%% [markdown] define the patch and freeze function


def direct_patch_and_freeze(
    model,
    source_dataset,
    target_dataset,
    ioi_dataset,
    sender_heads,
    receiver_hooks,
    max_layer,
    positions=["end"],
    verbose=False,
    return_hooks=False,
    extra_hooks=[],  # when we call reset hooks, we may want to add some extra hooks after this, add these here
):
    """
    Patch in the effect of `sender_heads` on `receiver_hooks` only
    (though MLPs are "ignored", so are slight confounders)

    If max_layer < 12, then let some part of the model do computations (not frozen)
    """

    sender_hooks = []

    for layer, head_idx in sender_heads:
        if head_idx is None:
            sender_hooks.append((f"blocks.{layer}.hook_mlp_out", None))

        else:
            sender_hooks.append((f"blocks.{layer}.attn.hook_result", head_idx))

    sender_hook_names = [x[0] for x in sender_hooks]
    receiver_hook_names = [x[0] for x in receiver_hooks]

    sender_cache = {}
    model.reset_hooks()
    for hook in extra_hooks:
        model.add_hook(*hook)
    model.cache_some(sender_cache, lambda x: x in sender_hook_names)
    # print(f"{sender_hook_names=}")
    source_logits = model(
        source_dataset.text_prompts
    )  # this should see what the logits are when i) main heads are ablated + ii) we're also ablating (lay, head_idx)

    target_cache = {}
    model.reset_hooks()
    for hook in extra_hooks:
        model.add_hook(*hook)
    model.cache_all(target_cache)
    target_logits = model(target_dataset.text_prompts)

    # for all the Q, K, V things
    model.reset_hooks()
    for layer in range(max_layer):
        for head_idx in range(12):
            for hook_template in [
                "blocks.{}.attn.hook_q",
                "blocks.{}.attn.hook_k",
                "blocks.{}.attn.hook_v",
            ]:
                hook_name = hook_template.format(layer)

                if hook_name in receiver_hook_names:
                    continue

                hook = get_act_hook(
                    patch_all,
                    alt_act=target_cache[hook_name],
                    idx=head_idx,
                    dim=2 if head_idx is not None else None,
                )
                model.add_hook(hook_name, hook)
    for hook in extra_hooks:
        # ughhh, think that this is what we want, this should override the QKV above
        model.add_hook(*hook)

    # we can override the hooks above for the sender heads, though
    for hook_name, head_idx in sender_hooks:
        assert not torch.allclose(sender_cache[hook_name], target_cache[hook_name]), (
            hook_name,
            head_idx,
        )
        hook = get_act_hook(
            partial(patch_positions, positions=positions),
            alt_act=sender_cache[hook_name],
            idx=head_idx,
            dim=2 if head_idx is not None else None,
        )
        model.add_hook(hook_name, hook)

    # measure the receiver heads' values
    receiver_cache = {}
    model.cache_some(receiver_cache, lambda x: x in receiver_hook_names)
    receiver_logits = model(target_dataset.text_prompts)

    # patch these values in
    model.reset_hooks()
    for hook in extra_hooks:
        model.add_hook(
            *hook
        )  # ehh probably doesn't actually matter cos end thing hooked

    hooks = []
    for hook_name, head_idx in receiver_hooks:
        hook = get_act_hook(
            partial(patch_positions, positions=positions),
            alt_act=receiver_cache[hook_name],
            idx=head_idx,
            dim=2 if head_idx is not None else None,
        )
        hooks.append((hook_name, hook))

    if return_hooks:
        return hooks
    else:
        for hook_name, hook in hooks:
            model.add_hook(hook_name, hook)
        return model


#%% [markdown] first patch-and-freeze experiments
# TODO why are there effects that come AFTER the patching?? it's before 36 mins in voko I think

dataset_names = [
    # "ioi_dataset",
    # "abca_dataset",
    # "dcc_dataset",
    # "acca_dataset",
    # "acba_dataset",
    "all_diff_dataset",
    # "totally_diff_dataset",
]

# patch all heads into the name mover input (hopefully find S2 Inhibition)

exclude_heads = [(layer, head_idx) for layer in range(12) for head_idx in range(12)]
for head in [(9, 9), (9, 6), (10, 0)]:
    exclude_heads.remove(head)

model.reset_hooks()
default_logit_diff = logit_diff(model, ioi_dataset)

# for pos in ["S+1", "S", "IO", "S2", "end"]:
for pos in ["end"]:
    print(pos)
    results = torch.zeros(size=(12, 12))
    mlp_results = torch.zeros(size=(12, 1))
    for source_layer in tqdm(range(12)):
        for source_head_idx in [None] + list(range(12)):
            model.reset_hooks()
            receiver_hooks = []
            # for layer, head_idx in circuit["name mover"]:
            # receiver_hooks.append((f"blocks.{layer}.attn.hook_q", head_idx))
            # receiver_hooks.append((f"blocks.{layer}.attn.hook_v", head_idx))
            # receiver_hooks.append((f"blocks.{layer}.attn.hook_k", head_idx))
            receiver_hooks.append(("blocks.11.hook_resid_post", None))

            model = direct_patch_and_freeze(
                model=model,
                source_dataset=all_diff_dataset,
                target_dataset=ioi_dataset,
                ioi_dataset=ioi_dataset,
                sender_heads=[(source_layer, source_head_idx)],
                receiver_hooks=receiver_hooks,
                max_layer=12,
                positions=[pos],
                verbose=False,
                return_hooks=False,
            )

            cur_logit_diff = logit_diff(model, ioi_dataset)

            if source_head_idx is None:
                mlp_results[source_layer] = cur_logit_diff - default_logit_diff
            else:
                results[source_layer][source_head_idx] = (
                    cur_logit_diff - default_logit_diff
                )

            if source_layer == 11 and source_head_idx == 11:
                # show attention head results
                fname = f"svgs/patch_and_freeze_{pos}_{ctime()}_{ri(2134, 123759)}"
                fig = show_pp(
                    results.T,
                    title=f"{fname=} {pos=} patching NMs",
                    return_fig=True,
                    show_fig=False,
                )

                fig.write_image(
                    f"svgs/patch_and_freezes/to_duplicate_token_K_{pos}.png"
                )

                fig.write_image(fname + ".png")
                fig.write_image(fname + ".svg")
                fig.show()

                # # show mlp results # mlps are (hopefully not anymore???) fucked
                fname = f"svgs/patch_and_freeze_mlp_{ctime()}_{ri(2134, 123759)}"
                fig = show_pp(
                    mlp_results.T,
                    title="Direct effect of MLPs on Logit Difference",
                    return_fig=True,
                    show_fig=False,
                )
                fig.write_image(fname + ".png")
                fig.write_image(fname + ".svg")
                fig.show()


#%% [markdown] MLP indirect effect

mlp_hooks = do_circuit_extraction(
    model=model,
    heads_to_remove={},
    mlps_to_keep={},
    ioi_dataset=ioi_dataset,
    mean_dataset=all_diff_dataset,
    return_hooks=True,
)

model.reset_hooks()
default_logit_diff = logit_diff(model, ioi_dataset)

mlp_results = torch.zeros(size=(12, 1))

for layer, hook in enumerate(tqdm(mlp_hooks)):
    model.reset_hooks()
    model.add_hook(*hook)
    new_logit_diff = logit_diff(model, ioi_dataset)
    change = new_logit_diff - default_logit_diff
    mlp_results[layer] = change

# # show mlp results # mlps are (hopefully not anymore???) fucked

# mlp_results[0] = 0
fname = f"svgs/mlp_indirect_{ctime()}_{ri(2134, 123759)}"
fig = show_pp(
    mlp_results,
    title="Indirect effect of MLPs on Logit Difference (after KO)",
    return_fig=True,
    show_fig=False,
)

# remove y labels
fig.update_layout(
    yaxis=dict(
        tickmode="array",
        tickvals=[0],
        ticktext=[""],
    )
)


fig.write_image(fname + ".png")
fig.write_image(fname + ".svg")
fig.show()

#%% [markdown] hack some LD and IO probs stuff

from ioi_circuit_extraction import RELEVANT_TOKENS, CIRCUIT


circuit = deepcopy(CIRCUIT)
# circuit["duplicate token"].remove((7, 1))
print(f"{circuit=}")
model.reset_hooks()

for dataset_name, mean_dataset in [
    ("ioi_dataset", all_diff_dataset),
    ("abba_dataset", abba_all_diff),
    ("baba_dataset", baba_all_diff),
]:
    print(dataset_name)
    dataset = eval(dataset_name)
    for make_circuit in [False, True]:
        model.reset_hooks()

        if make_circuit:
            heads_circuit = get_heads_circuit(ioi_dataset=dataset, circuit=circuit)
            model, _ = do_circuit_extraction(
                model=model,
                heads_to_keep=heads_circuit,
                mlps_to_remove={},
                ioi_dataset=dataset,
                mean_dataset=mean_dataset,
            )

        cur_logit_diff = logit_diff(model, dataset)
        cur_io_probs = probs(model, dataset)
        print(f"{make_circuit=} {cur_logit_diff=} {cur_io_probs=}")
    print()

#%% [markdown] brief dive into 7.1, ignore (this cell vizualizes the average attention)
ys = []
average_attention = {}

baba_dataset = IOIDataset(N=100, prompt_type="BABA")
abba_dataset = IOIDataset(N=100, prompt_type="ABBA")

for dataset_name in [
    "abba_dataset",
    "baba_dataset",
    "ioi_dataset",
    "all_diff_dataset",
    "totally_diff_dataset",
]:
    fig = go.Figure()
    print(dataset_name)
    dataset = eval(dataset_name)

    for heads_raw in [(6, 9)]:  # circuit["duplicate token"]:  # ["induction"]:
        heads = [heads_raw]
        average_attention[heads_raw] = {}
        cur_ys = []
        cur_stds = []
        att = torch.zeros(size=(dataset.N, dataset.max_len, dataset.max_len))
        for head in tqdm(heads):
            att += show_attention_patterns(
                model,
                [head],
                dataset,
                return_mtx=True,
                mode="val",  # , mode="scores"
            )
        att /= len(heads)

        vals = att[torch.arange(dataset.N), dataset.word_idx["S2"][: dataset.N], :]
        evals = torch.exp(vals)
        val_sum = torch.sum(evals, dim=1)
        assert val_sum.shape == (dataset.N,), val_sum.shape
        print(f"{heads=} {val_sum.mean()=}")

        for key in dataset.word_idx.keys():
            end_to_s2 = att[
                torch.arange(dataset.N),
                dataset.word_idx["S2"][: dataset.N],
                dataset.word_idx[key][: dataset.N],
            ]
            # ABCA dataset calculates S2 in trash way... so we use the IOI dataset indices
            cur_ys.append(end_to_s2.mean().item())
            cur_stds.append(end_to_s2.std().item())
            average_attention[heads_raw][key] = end_to_s2.mean().item()
        fig.add_trace(
            go.Bar(
                x=list(dataset.word_idx.keys()),
                y=cur_ys,
                error_y=dict(type="data", array=cur_stds),
                name=str(heads_raw),
            )
        )  # ["IOI", "ABCA"][idx]))

        fig.update_layout(title_text="Attention probs; from S2 to blah")
    fig.show()
#%% [markdown] plotting (your downfalls!)

from ioi_utils import CLASS_COLORS

cc = deepcopy(CLASS_COLORS)
no_112 = deepcopy(CIRCUIT)
no_112["name mover"].remove((11, 2))


def what_class(layer, head, circuit):
    for circuit_class in circuit:
        if (layer, head) in circuit[circuit_class]:
            return circuit_class
    return "duplicate token"
    raise ValueError((layer, head), circuit)


# plot the most important heads by

for results in [backup_results]:  # backup results or initial results
    k = 15
    top_heads = max_2d(torch.abs(results), k=k)[
        0
    ]  # initial results is the patch with no KOs; direct effect on logits

    exclude_heads = []
    exclude_heads = [
        (layer_idx, head)
        for layer_idx in range(12)
        for head in range(12)
        if what_class(layer_idx, head, circuit=circuit)
        not in ["name mover", "negative", "s2 inhibition"]
    ]

    heights = [
        results[layer][head]
        for layer, head in top_heads
        if (layer, head) not in exclude_heads
    ]
    colors = [
        cc[what_class(layer, head_idx, circuit=circuit)]
        for layer, head_idx in top_heads
        if (layer, head_idx) not in exclude_heads
    ]

    fig = go.Figure()
    heads = [str(x) for x in top_heads if x not in exclude_heads]
    fig.add_trace(
        go.Bar(
            x=heads,
            y=heights,
            orientation="v",
            marker_color=colors,
            showlegend=False,
            # width=0.5,
            # offset=0,
        )
    )

    used_col = []
    for i, head in enumerate(heads):  # pretty hacky, adds relevant legend things

        if colors[i] in used_col:
            continue
        used_col.append(colors[i])

        fig.add_trace(
            go.Scatter(
                x=["(11, 10)"],
                y=[0],
                mode="lines",
                line=dict(color=colors[i], width=10),
                showlegend=True,
                name=what_class(*eval(head), circuit=circuit),
            )
        )

    # no legend
    fig.update_layout(showlegend=False)
    fig.update_layout(
        title_text=f"Heads with largest absolute effect on logit difference after patching",
    )
    fig.update_layout(barmode="group")
    fig.update_layout(bargap=0.0, bargroupgap=0.0)

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    if not torch.allclose(results, initial_results):
        fig.update_layout(
            title_text=f"Heads with largest absolute effect on logit difference after patching, and KO of [(9, 9), (10, 0), (9, 6)]",
        )
        fig.update_layout(showlegend=True)
        # fig.update_layout(title_text="Initial results")
        backup_names = []
        backup_values = []

        for head in circuit["name mover"]:
            if head in [(9, 9), (9, 6), (10, 0)]:
                continue
            backup_names.append(str(head))
            backup_values.append(initial_results[head[0]][head[1]])

        fig.add_trace(
            go.Bar(
                x=backup_names,
                y=backup_values,
                orientation="v",
                marker_color="rgb(27, 300, 119)",
                name="name mover (before name mover KO)",
                # width=0.5,
            )
        )

    # add legend
    fig.update_layout(
        # title_text="Heads with largest effect on logits",
        xaxis_title="(Layer, Head)",
        yaxis_title="Change in logit difference",
        legend_title="Head type",
        # font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"),
    )

    # relative barmode
    fig.update_layout(barmode="overlay")

# set y axis range to [-1, 1]
fig.update_yaxes(range=[-3, 3])
fname = f"backup_nm_plot_{ctime()}"

# add a legend

# update title
# fig.update_layout(title=fname)
fig.write_image(f"svgs/{fname}.svg")
fig.show()
#%% [markdown] do the scatter plot, but now with direct effect on logit diff

receiver_hooks = [("blocks.11.hook_resid_post", None)]
layer = 9
head_idx = 9

hooks = direct_patch_and_freeze(
    model=model,
    source_dataset=all_diff_dataset,
    target_dataset=ioi_dataset,
    ioi_dataset=ioi_dataset,
    sender_heads=[(layer, head_idx)],
    receiver_hooks=receiver_hooks,
    max_layer=12,
    positions=["end"],
    verbose=False,
    return_hooks=True,
)

attention_hook_name = f"blocks.{layer}.attn.hook_attn"
# (batch, head, from, to)

ys = []

all_io_attentions = []
all_s_attentions = []
all_io_logits = []
all_s_logits = []

for add_hooks in [False, True]:
    model.reset_hooks()
    if add_hooks:
        for hook in hooks:
            model.add_hook(*hook)

    cache = {}
    model.cache_some(cache, lambda name: name == attention_hook_name)

    io_logits, s_logits = logit_diff(model, ioi_dataset, all=True, both=True)
    io_logits = io_logits.detach().cpu()
    s_logits = s_logits.detach().cpu()

    all_io_logits.append(io_logits)
    all_s_logits.append(s_logits)

    for token, all_attentions in zip(
        ["IO", "S"], [all_io_attentions, all_s_attentions]
    ):
        attention = (
            cache[attention_hook_name][
                torch.arange(ioi_dataset.N),
                head_idx,
                ioi_dataset.word_idx["end"],
                ioi_dataset.word_idx[token],
            ]
            .detach()
            .cpu()
        )
        all_attentions.append(attention.clone())

# df.append([prob, dot, tok_type, prompt["text"]])

# # most of the pandas stuff is intuitive, no need to deeply understand
# viz_df = pd.DataFrame(
#     df, columns=[f"Attn Prob on Name", f"Dot w Name Embed", "Name Type", "text"]
# )
# fig = px.scatter(
#     viz_df,
#     x=f"Attn Prob on Name",
#     y=f"Dot w Name Embed",
#     color="Name Type",
#     hover_data=["text"],
#     color_discrete_sequence=["rgb(114,255,100)", "rgb(201,165,247)"],
#     title=f"How Strong {layer_no}.{head_no} Writes in the Name Embed Direction Relative to Attn Prob",
# )

df = pd.concat(
    [
        pd.DataFrame(
            {
                "attention": all_s_attentions[0],
                "change": -(all_s_logits[1] - all_s_logits[0]),
                "token": "S",
                "text": ioi_dataset.text_prompts,
                # "change": (all_io_logits[0] - all_s_logits[0])
                # - (all_io_logits[1] - all_s_logits[1]),
            }
        ),
        pd.DataFrame(
            {
                "attention": all_io_attentions[0],
                "change": -(all_io_logits[1] - all_io_logits[0]),
                "token": "IO",
                "text": ioi_dataset.text_prompts,
            }
        ),
    ]
)


fig = px.scatter(
    df,
    x=f"attention",
    y=f"change",
    color="token",
    hover_data=["text"],
    color_discrete_sequence=["rgb(114,255,100)", "rgb(201,165,247)"],
    title=f"How {layer}.{head_idx} affects logits (change after patch-and-freeze)",
)

# update y axis label
y_label = "Change in logits"
fig.update_yaxes(title_text=y_label)
fig.update_xaxes(title_text="Attention on token")

fig.write_image(f"svgs/attention_scatter_{y_label}_{ctime()}.svg")
fig.write_image(f"svgs/attention_scatter_{y_label}_{ctime()}.png")
fig.show()

#%%

# select where token == "IO" in df
xs = df[df["token"] == "IO"]["attention"]
ys = df[df["token"] == "IO"]["change"]

# correlation coefficient
np.corrcoef(xs, ys)
