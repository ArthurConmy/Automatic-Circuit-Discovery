#%%
# Arthur investigation into dropout
from copy import deepcopy
from inspect import stack
from tkinter import wantobjects
import torch
from queue import Queue
from easy_transformer.experiments import get_act_hook
from utils_induction import *

assert torch.cuda.device_count() == 1
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch as t
from easy_transformer.EasyTransformer import (
    EasyTransformer,
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
from ioi_dataset import (
    IOIDataset,
)
from ioi_utils import (
    path_patching,
    max_2d,
    CLASS_COLORS,
    e,
    show_pp,
    show_attention_patterns,
    scatter_attention_and_contribution,
)
from random import randint as ri
from easy_transformer.experiments import get_act_hook
from ioi_circuit_extraction import (
    do_circuit_extraction,
    get_heads_circuit,
    CIRCUIT,
)
import random as rd
from ioi_utils import logit_diff, probs
from ioi_utils import get_top_tokens_and_probs as g

ipython = get_ipython()
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
#%% [markdown]
# Make models

gpt2 = EasyTransformer.from_pretrained("gpt2").cuda()
gpt2.set_use_attn_result(True)

opt = EasyTransformer.from_pretrained("facebook/opt-125m").cuda()
opt.set_use_attn_result(True)

neo = EasyTransformer.from_pretrained("EleutherAI/gpt-neo-125M").cuda()
neo.set_use_attn_result(True)

solu = EasyTransformer.from_pretrained("solu-10l-old").cuda()
solu.set_use_attn_result(True)

model_names = ["gpt2", "opt", "neo", "solu"]
model_name = "gpt2"
print(f"USING {model_name}")
model = eval(model_name)

saved_tensors = []
    #%% [markdown]
# Make induction dataset

seq_len = 100
batch_size = 20
interweave = 100  # have this many things before a repeat

rand_tokens = torch.randint(1000, 10000, (batch_size, seq_len))
rand_tokens_repeat = torch.zeros(
    size=(batch_size, seq_len * 2)
).long()  # einops.repeat(rand_tokens, "batch pos -> batch (2 pos)")

for i in range(seq_len // interweave):
    rand_tokens_repeat[
        :, i * (2 * interweave) : i * (2 * interweave) + interweave
    ] = rand_tokens[:, i * interweave : i * interweave + interweave]
    rand_tokens_repeat[
        :, i * (2 * interweave) + interweave : i * (2 * interweave) + 2 * interweave
    ] = rand_tokens[:, i * interweave : i * interweave + interweave]
rand_tokens_control = torch.randint(1000, 10000, (batch_size, seq_len * 2))

rand_tokens = prepend_padding(rand_tokens, model.tokenizer)
rand_tokens_repeat = prepend_padding(rand_tokens_repeat, model.tokenizer)
rand_tokens_control = prepend_padding(rand_tokens_control, model.tokenizer)


def calc_score(attn_pattern, hook, offset, arr):
    # Pattern has shape [batch, index, query_pos, key_pos]
    stripe = attn_pattern.diagonal(offset, dim1=-2, dim2=-1)
    scores = einops.reduce(stripe, "batch index pos -> index", "mean")
    # Store the scores in a common array
    arr[hook.layer()] = scores.detach().cpu().numpy()
    # return arr
    return attn_pattern

def filter_attn_hooks(hook_name):
    split_name = hook_name.split(".")
    return split_name[-1] == "hook_attn"

arrs = []

#%% [markdown]
# do logit lens

n_heads = model.cfg.n_heads
n_layers = model.cfg.n_layers
d_model = model.cfg.d_model

model_unembed = (
    model.unembed.W_U.detach().cpu()
)  # note that for GPT2 embeddings and unembeddings are tides such that W_E = Transpose(W_U)

unembed_bias = model.unembed.b_U.detach().cpu()

attn_vals = torch.zeros(size=(n_heads, n_layers)).cuda()
mlp_vals = torch.zeros(size=(n_layers,)).cuda()
model.reset_hooks()
cache = {}
model.cache_some(cache, lambda x: True)
logits = model(rand_tokens_repeat) # (model, ioi_dataset, all=True).cpu()

for b in tqdm(range(batch_size)):
    for i in range(seq_len + 1, 2 * seq_len + 1):
        # toks = ioi_dataset[i : i + 1].toks.long()
        tok = rand_tokens_repeat[b, i]
        dire = model_unembed[:, tok]
        unembed_b = unembed_bias[tok]
        dire = dire.to("cuda")

        for lay in range(n_layers):
            cur_attn = (
                cache[f"blocks.{lay}.attn.hook_result"][b, i-1, :, :]
            )
            cur_mlp = cache[f"blocks.{lay}.hook_mlp_out"][b, i-1, :]
            attn_vals[lay, :] += torch.einsum("d,hd->h", dire, cur_attn) + unembed_b.to("cuda")
            mlp_vals[lay] += torch.einsum("h,h->", dire, cur_mlp) + unembed_b.to("cuda")

attn_vals /= (batch_size * seq_len)
mlp_vals /= (batch_size * seq_len)
all_figs = []

title = "Hello, world!"

show_pp(
    attn_vals,
    xlabel="head no",
    ylabel="layer no",
    title=title,
    # return_fig=True,
)

#%%
# plot a bar chart of the top 10 most important heads
mack_2d = max_2d(attn_vals, k=20)
x_labels = mack_2d[0]
x_labels = [f"head {x}" for x in x_labels]
values = mack_2d[1]

# plot a bar chart of values with x_labels as the x axis
fig = plt.figure(figsize=(10, 10))
plt.bar(x_labels, values.cpu())
plt.title(title)
plt.xlabel("head no")
plt.ylabel("value")

# vertical labels
plt.xticks(rotation=90)
plt.show()
#%%
if return_figs and return_vals:
    return all_figs, attn_vals, mlp_vals
if return_vals:
    return attn_vals, mlp_vals
if return_figs:
    return all_figs

logits = model(rand_tokens_repeat)
logits_on_correct = logits[torch.arange(batch_size), rand_tokens_repeat[:, 1:].argmax(-1)]

#%% [markdown]
# sweeeeeet plot

show_losses(
    models=[neo], # eval(model_name) for model_name in model_names],
    model_names=model_names[:1],
    rand_tokens_repeat=rand_tokens_repeat.cuda(),
    seq_len=seq_len,
    mode="loss",
)
#%% [markdown]
# Induction scores
# Use this to get a "shortlist" of the heads that matter most for ind

def filter_attn_hooks(hook_name):
    split_name = hook_name.split(".")
    return split_name[-1] == "hook_attn"

model.reset_hooks()
more_hooks = []

# for head in [(11, head_idx) for head_idx in range(5)]: # nduct_heads[:5]:
    # more_hooks.append(hooks[head])

def get_induction_scores(model, rand_tokens_repeat, title=""):

    def calc_induction_score(attn_pattern, hook):
        # Pattern has shape [batch, index, query_pos, key_pos]
        induction_stripe = attn_pattern.diagonal(1 - seq_len, dim1=-2, dim2=-1)
        induction_scores = einops.reduce(
            induction_stripe, "batch index pos -> index", "mean"
            )

        # Store the scores in a common arraymlp_ = saved_tensors[-2].clone()
        induction_scores_array[hook.layer()] = induction_scores.detach().cpu().numpy()

    model = eval(model_name)
    induction_scores_array = np.zeros((model.cfg.n_layers, model.cfg.n_heads))
    induction_logits = model.run_with_hooks(
        rand_tokens_repeat, fwd_hooks= more_hooks + [(filter_attn_hooks, calc_induction_score)], # , reset_hooks_start=False,
    )
    induction_scores_array = torch.tensor(induction_scores_array)
    fig = px.imshow(
        induction_scores_array,
        labels={"y": "Layer", "x": "Head"},
        color_continuous_scale="Blues",       
    )
    # add title
    fig.update_layout(
        title_text=f"Induction scores for "+ title,
        title_x=0.5,
        title_font_size=20,
    )
    fig.show()
    return induction_scores_array

induction_scores_array = get_induction_scores(model, rand_tokens_repeat, title=model_name)
#%% [markdown]
# is GPT-Neo behaving right?

logits_and_loss = model(
    rand_tokens_repeat, return_type="both", loss_return_per_token=True
)
logits = logits_and_loss["logits"].cpu()[:, :-1] # remove unguessable next token
loss = logits_and_loss["loss"].cpu()

probs_denoms = torch.sum(torch.exp(logits), dim=-1, keepdim=True)
probs_num = torch.exp(logits)
probs = probs_num / probs_denoms

# probs = torch.softmax(logits, dim=-1)

batch_size, _, vocab_size = logits.shape
seq_indices = einops.repeat(torch.arange(_), "a -> b a", b=batch_size)
batch_indices = einops.repeat(torch.arange(batch_size), "b -> b a", a=_)
probs_on_correct = probs[batch_indices, seq_indices, rand_tokens_repeat[:, 1:]]
log_probs = - torch.log(probs_on_correct)

assert torch.allclose(
    log_probs, loss, rtol=1e-3, atol=1e-3, # torch.exp(log_probs.gather(-1, rand_tokens_repeat[:, 1:].unsqueeze(-1)).squeeze(-1))
)
#%% [markdown]
# THIS CELL MAKES ALL THE HOOKS : )

def random_patching(z, act, hook):
    """This keeps position the same, but changes the sequence
    Since they're all generated from RANDOM     tokens, will be different each time
    WARNING: can produce non-zero effects e.g even when reeiver hooks are after sender hooks cos random
    is different each time"""
    b = z.shape[0]
    z[torch.arange(b)] = act[torch.randperm(b)]
    return z

cache = {}
model.reset_hooks()
model.cache_some(
    cache,
    lambda x: "attn.hook_result" in x or "mlp_out" in x,
    suppress_warning=True,
)
logits, loss = model(
    rand_tokens_control, return_type="both", loss_return_per_token=True
).values()

hooks = {}
all_heads_and_mlps = [(layer, head_idx) for layer in range(model.cfg.n_layers) for head_idx in [None] + list(range(model.cfg.n_heads))]

for layer, head_idx in all_heads_and_mlps:
    hook_name = f"blocks.{layer}.attn.hook_result"
    if head_idx is None:
        hook_name = f"blocks.{layer}.hook_mlp_out"

    hooks[(layer, head_idx)] = (
        hook_name,
        get_act_hook(
            random_patching,
            alt_act=cache[hook_name].clone(),
            idx=head_idx,
            dim=2 if head_idx is not None else None,
            name=hook_name,
        ),
    )
model.reset_hooks()

#%%

for hook_list in [[], [hooks[(9, 9)], hooks[(9, 6)], hooks[(10, 0)]]]:
    model.reset_hooks()
    for hook in hook_list:
        model.add_hook(*hook)
    losses = model(rand_tokens_repeat, return_type="loss", loss_return_per_token=True)
    losses2 = losses[:, seq_len+1:]
    print(losses2.mean().item())

#%% # hmm this seems weird but we

model.reset_hooks()
initial_loss = model(rand_tokens_repeat, return_type="loss", loss_return_per_token=True)[:, seq_len:].mean().item()

results = []
labels = []

for head in mack_2d[0]:
    model.reset_hooks()
    model.add_hook(*hooks[head])
    losses = model(rand_tokens_repeat, return_type="loss", loss_return_per_token=True)[:, seq_len:]
    results.append(losses.mean().item() - initial_loss)
    labels.append(str(head))

# plot bar chart
fig = go.Figure(
    data=[
        go.Bar(
            x=labels,
            y=results,
            # text=[f"{(loss - initial_loss):.2f}" for loss in results],
            textposition="auto",
        )
    ]
)
fig.show()

#%%

for ablate_61 in [False, True]:
    for ablate_other_6 in [False, True]:

        model.reset_hooks()
        if ablate_61:
            model.add_hook(*hooks[(6, 1)])
        if ablate_other_6:
            for head_idx in [0, 6, 11]: 
                model.add_hook(*hooks[(6, head_idx)])

        ans = loss_metric(model, rand_tokens_repeat, seq_len)
        print(ans, ablate_61, ablate_other_6)
#%% [markdown]
# use this cell to get a rough grip on which heads matter the most
model.reset_hooks()
both_results = []
the_extra_hooks = None

metric = logits_metric

for idx, extra_hooks in enumerate([[]]): # , [hooks[((6, 1))]], [hooks[(11, 4)]], the_extra_hooks]):
    if extra_hooks is None:
        break
    results = torch.zeros(size=(model.cfg.n_layers, model.cfg.n_heads))
    mlp_results = torch.zeros(size=(model.cfg.n_layers, 1))
    model.reset_hooks()
    for hook in extra_hooks:
        model.add_hook(*hook)
    # initial_loss = model(
    #     rand_tokens_repeat, return_type="both", loss_return_per_token=True
    # )["loss"][:, -seq_len // 2 :].mean()
    initial_metric = metric(model, rand_tokens_repeat)
    print(f"Initial initial_metric: {initial_metric}")

    for source_layer in tqdm(range(model.cfg.n_layers)):
        for source_head_idx in [None] + list(range(model.cfg.n_heads)):
            model.reset_hooks()
            receiver_hooks = []
            receiver_hooks.append((f"blocks.{model.cfg.n_layers-1}.hook_resid_post", None))
            # receiver_hooks.append((f"blocks.11.attn.hook_result", 4))
            # receiver_hooks.append((f"blocks.10.hook_mlp_out", None))


            # for layer in range(7, model.cfg.n_layers): # model.cfg.n_layers):
            #     for head_idx in list(range(model.cfg.n_heads)) + [None]:
            #         hook_name = f"blocks.{layer}.attn.hook_result"
            #         if head_idx is None:
            #             hook_name = f"blocks.{layer}.hook_mlp_out"
            #         receiver_hooks.append((hook_name, head_idx))

            if False:
                model = path_patching_attribution(
                    model=model,
                    tokens=rand_tokens_repeat,
                    patch_tokens=rand_tokens_control,
                    sender_heads=[(source_layer, source_head_idx)],
                    receiver_hooks=receiver_hooks,
                    device="cuda",
                    freeze_mlps=True,
                    return_hooks=False,
                    max_layer=11,
                    extra_hooks=extra_hooks,
                )
                title="Direct"

            else:
                # model.add_hook(*hooks[(6, 1)])
                model.add_hook(*hooks[(source_layer, source_head_idx)])
                title="Indirect"

            # model.reset_hooks()
            # for hook in hooks:
            #     model.add_hook(*hook)
            # loss = model(
            #     rand_tokens_repeat, return_type="both", loss_return_per_token=True
            # )["loss"][:, -seq_len // 2 :].mean()
            cur_metric = metric(model, rand_tokens_repeat)

            a = hooks.pop((source_layer, source_head_idx))
            e("a")

            if source_head_idx is None:
                mlp_results[source_layer] = cur_metric - initial_metric
            else:
                results[source_layer][source_head_idx] = cur_metric - initial_metric

            if source_layer == model.cfg.n_layers-1 and source_head_idx == model.cfg.n_heads-1:
                fname = f"svgs/patch_and_freeze_{ctime()}_{ri(2134, 123759)}"
                fig = show_pp(
                    results.detach(),
                    title=f"{title} effect of path patching heads with metric {metric} {fname}",
                    # + ("" if idx == 0 else " (with top 3 name movers knocked out)"),
                    return_fig=True,
                    show_fig=False,
                )
                both_results.append(results.clone())
                fig.show()
                show_pp(mlp_results.detach().cpu())
                saved_tensors.append(results.clone().cpu())
                saved_tensors.append(mlp_results.clone().cpu())

#%% [markdown]
# Sort induction heads into positive and negative heads

no_heads = 10
heads_by_induction = max_2d(induction_scores_array, 144)[0]
induct_heads = []
neg_heads = []
idx = 0

mult_factor = 1.0 if "logit" in metric.__name__ else -1.0

while len(induct_heads) < no_heads:
    head = heads_by_induction[idx]
    idx+=1
    if "results" in dir() and (mult_factor * results[head]) <= 0:
        induct_heads.append(head)
    else:
        neg_heads.append(head)
        print(f" {head} is a candidate `negative induction head`, with {metric.__name__}={results[head]:.2f} and induction score {induction_scores_array[head]:.2f}")

# sort the induction heads by their results
if "results" in dir():
    induct_heads = sorted(induct_heads, key=lambda x: -(results[x]), reverse=True)

# have a look at these numbers
for layer, head_idx in induct_heads:
    print(f"Layer: {layer}, Head: {head_idx}, {metric.__name__}: {results[layer][head_idx]:.2f}, Induction score: {induction_scores_array[layer][head_idx]:.2f}")

print(induct_heads)


#%% [markdown] 
# What about the direct effects, on hook 8 resid post???

for layer in range(6, model.cfg.n_layers):
    for suffix in ["mid", "post"]:
        for extra_hooks in [[], [hooks[(6, 1)]]]:
            print(f"Layer {layer} {suffix} {extra_hooks}")
            results = torch.zeros(size=(model.cfg.n_layers, model.cfg.n_heads))
            mlp_results = torch.zeros(size=(model.cfg.n_layers, 1))

            model.reset_hooks()
            for hook in extra_hooks:
                model.add_hook(*hook)
            initial_metric = metric(model, rand_tokens_repeat)
            print(f"Initial initial_metric: {initial_metric}")

            receiver_hooks = [get_hook(8, i) for i in range(12)]
            receiver_hooks = []
            for letter in ["q", "k", "v"]:
                template = "blocks.8.attn.hook_" + letter
                for head_idx in range(12):
                    receiver_hooks.append((template, head_idx))
            receiver_hooks = [get_hook(6, None), get_hook(7, None), get_hook(8, None)]
            receiver_hooks = [(f"blocks.{layer}.hook_resid_{suffix}", None)]

            for source_layer in tqdm(range(model.cfg.n_layers)):
                for source_head_idx in [None] + list(range(model.cfg.n_heads)):
                    model.reset_hooks()
                    model = path_patching_attribution(
                        model=model,
                        tokens=rand_tokens_repeat,
                        patch_tokens=rand_tokens_control,
                        sender_heads=[(source_layer, source_head_idx)],
                        receiver_hooks=receiver_hooks,
                        device="cuda",
                        freeze_mlps=True,
                        return_hooks=False,
                        max_layer=11,
                        extra_hooks=extra_hooks,
                    )
                    cur_metric = metric(model, rand_tokens_repeat)

                    if source_head_idx is None:
                        mlp_results[source_layer] = cur_metric - initial_metric
                    else:
                        results[source_layer][source_head_idx] = cur_metric - initial_metric

                    if source_layer != 0:
                        assert not torch.allclose(results, 0.0 * results)

                    if source_layer == model.cfg.n_layers-1 and source_head_idx == model.cfg.n_heads-1:
                        fname = f"svgs/patch_and_freeze_{ctime()}_{ri(2134, 123759)}"
                        fig = show_pp(
                            results.detach(), # TODO this must be bugged, because we're getting effects from AFTER the receiver hook
                            title="Change in logits on correct, path patching -> layer 8, post attention layer" + (", with 6.1 knocked out." if len(extra_hooks)>0 else ""), # f"{title} effect of path patching heads with metric {metric} {fname}",
                            # + ("" if idx == 0 else " (with top 3 name movers knocked out)"),
                            return_fig=True,
                            show_fig=False,
                            xlabel="Head",
                            ylabel="Layer",
                        )
                        both_results.append(results.clone())
                        fig.show()
                        show_pp(mlp_results.detach().cpu(), title="MLP results")
                        saved_tensors.append(results.clone().cpu())
                        saved_tensors.append(mlp_results.clone().cpu())


#%%
for prefix in ["no_", ""]:
    for suffix in ["ko_mlp_results", "ko_results"]:
        exec(f"m = {prefix}{suffix} = torch.load('pts/{prefix}{suffix}.pt')")
        print(torch.sum(m), prefix, suffix)

diff_results = ko_results - no_ko_results
diff_results[6, 1] = 0
show_pp(diff_results, title="Difference in results with and without 6.1 knocked out2")

diff_results2 = ko_mlp_results - no_ko_mlp_results
# diff_results2[6, 0] = 0
show_pp(diff_results2)
#%% [markdown]
# Is Layer 8 updating on Layer 6? On MLP 6 or 7?

model.reset_hooks()
initial_metric = metric(model, rand_tokens_repeat)

# receiver_hooks = []
# for i in range(12):
#     if i == 1: continue
#     receiver_hooks.append((f"blocks.8.attn.hook_result", i)) # HMM something bugged as the effect seems same size if 11 or 1 head here...

model = path_patching_attribution(
    model=model,
    tokens=rand_tokens_repeat,
    patch_tokens=rand_tokens_control,
    sender_heads=[(6, None)],
    receiver_hooks=receiver_hooks,
    device="cuda",
    freeze_mlps=True,
    return_hooks=False,
    max_layer=11,
    extra_hooks=extra_hooks,
)

new_metric = metric(model, rand_tokens_repeat)
print(initial_metric, new_metric)
#%% [markdown] 
# Can we ~retain performance while making no cross position stuff happen?

answers = []
from easy_transformer.utils import get_corner

def remove_all_off_diagonal(z, hook): # ablates all off diagonal stuff
    z = z.clone()
    print(hook.name, "hook") # : )
    for head_idx in range(12):
        z[:, head_idx, (1.0 - torch.eye(z.shape[-1])).bool()] = 0
        # if head_idx not in [2, 4] or "11" not in hook.name: # literally ablate all the things that aren't 11.4
        #     z[:, head_idx, (torch.eye(z.shape[-1])).bool()] = 0
        print(get_corner(z[0, head_idx]))
    return z

model.reset_hooks()
for layer in range(9, 12):
    hook_name = f"blocks.{layer}.attn.hook_attn"
    model.add_hook(hook_name, remove_all_off_diagonal)

new_value = logits_metric(model, rand_tokens_repeat)
print((new_value - initial_metric) / initial_metric)
print(f"Layer: {layer}, Head: {head_idx}, {metric.__name__}: {new_value:.2f}, Induction score: {induction_scores_array[layer][head_idx]:.2f}")
#%% [markdown]

all_heads = induct_heads + neg_heads
pre_heads = [h for h in induct_heads if h[0] <= 6]
post_heads = [h for h in all_heads if h[0] > 8]
model.reset_hooks()

patch_results = torch.zeros_like(results)
patch_results_mlp = torch.zeros_like(mlp_results)

# for layer, head_idx in tqdm(all_heads):
for layer in tqdm(range(12)):
    for head_idx in [None] + list(range(model.cfg.n_heads)):
        model.reset_hooks()
        model = path_patching_attribution(
            model=model,
            tokens=rand_tokens_repeat,
            patch_tokens=rand_tokens_control,
            sender_heads=all_heads,
            receiver_hooks=[get_hook(layer, head_idx)], #  [(f"blocks.{layer}.attn.hook_result", head_idx)],
            # receiver_hooks=[(f"blocks.{layer}.attn.hook_result", head_idx) for layer, head_idx in neg_heads],
            device="cuda",
            freeze_mlps=False,
            return_hooks=False,
            max_layer=layer,
            extra_hooks=[],
            do_assert=True,    
        )
        if head_idx is None:
            patch_results_mlp[layer] = metric(model, rand_tokens_repeat, seq_len) - initial_metric
        else:
            patch_results[layer][head_idx] = metric(model, rand_tokens_repeat, seq_len) - initial_metric

show_pp(patch_results.T.detach().cpu())
show_pp(patch_results_mlp.T.detach().cpu())

# cur_metric = metric(model, rand_tokens_repeat, seq_len)
# print(initial_metric, "to", cur_metric)

#%% [markdown]
# Subsets of these

show_fig = True
vals = []
subsets = [[] for _ in range(30)]
thing61 = [(6, 1)]

tot = 0
add_extra_hooks = False
beta = 1.0

if show_fig:
    fig = gof()

x = (np.logspace(0, -1,  10) - 0.1)
x = list(- x + 1.0)
x = torch.tensor(x)

x = torch.linspace(0, 1, 20)

for ablation_beta_for_negs in x:
    names = []
    losses = []
    logits = []
    sizes = []

    for ablation_beta_for_61 in torch.tensor([0.0, 1.0]):
        tot += 1
        if tot > 50: break

        sender_heads = [(6, 1)] + neg_heads

        ablation_beta = {head: ablation_beta_for_negs for head in neg_heads}
        ablation_beta[(6, 1)] = ablation_beta_for_61

        model = path_patching_attribution(
            model=model,
            tokens=rand_tokens_repeat,
            patch_tokens=rand_tokens_control,
            sender_hooks=sender_heads,
            receiver_hooks=[(f"blocks.6.hook_resid_post", None)],
            device="cuda",
            zero_ablation=False,
            ablation_beta=ablation_beta,
            freeze_mlps=True,
            return_hooks=False,
            max_layer=layer,
            extra_hooks=[],
            do_assert=True,    
        )

        names.append(str("sjkdfe") + "_" + str(ablation_beta_for_negs.item()) + "_" + str(ablation_beta_for_61.item()))
        sizes.append(1 + 4 * ablation_beta_for_negs.item())
        losses.append(loss_metric(model, rand_tokens_repeat, seq_len))
        logits.append(logits_metric(model, rand_tokens_repeat))

    pos = get_position(logits)
    vals.append(pos)
    subsets[pos].append((losses, logits))

    if show_fig:
        # make a scatter plot of losses against logits, with labels for each point and different symbols
        fig.add_trace(
            go.Scatter(
                x=logits,
                y=losses,
                mode="lines+markers",
                line=dict(color="black"),
                text=names,
                marker=dict(size=12, color=sizes, colorscale="Viridis", showscale=True, line=dict(
                color=4,
                width=2
            )),
            )
        )

        # add caption to colorbar    
        fig.update_layout(
            title=f"Loss and logits when we ablate (top induction heads), and k (see color bar) negative induction heads",
            xaxis_title="Logits",
            yaxis_title="Loss",
        )

    # break

fig.show()

#%%

# plot a histogram of vals
fig = go.Figure()
fig.add_trace(go.Histogram(x=vals))
fig.update_layout(
    title=f"Position of the subset of top induction heads in the list of subsets of negative induction heads",
    xaxis_title="Position in list",
    yaxis_title="Frequency",
)
fig.show()

#%%
#%%
# Plot a scatter plot in plotly with labels
fig = go.Figure()
for layer in range(model.cfg.n_layers):
    for head in range(model.cfg.n_heads):
        fig.add_trace(go.Scatter(x=[induction_scores_array[layer][head].item()], y=[results[layer][head].item()], mode='markers', name=f"Layer: {layer}, Head: {head}"))
fig.update_layout(title="Induction score vs loss diff", xaxis_title="Induction score", yaxis_title="Change in logits on correct")
fig.show()


# fig = go.Figure()
# fig.add_trace(go.Scatter(x=induction_scores_array.flatten().cpu().detach(), y=results.flatten().cpu().detach(), mode='markers'))
# fig.show()


#%% [markdown]
# Look at attention patterns of things

my_heads = [(j, i) for i in range(12) for j in range(1, 6)]
# my_heads = max_2d(torch.abs(results), k=20)[0]
print(my_heads)
my_heads = [(7, 0)]

# my_sheads = [(6, 6), (6, 11)] + induct_heads

for LAYER, HEAD in my_heads:
    model.reset_hooks()
    hook_name = f"blocks.{LAYER}.attn.hook_attn" # 4 12 50 50
    new_cache = {}
    model.cache_some(new_cache, lambda x: hook_name in x)
    # model.add_hook(*hooks[((6, 1))])
    # model.add_hooks(hooks)
    model(rand_tokens_repeat)

    att = new_cache[hook_name]
    mean_att = att[:, HEAD].mean(dim=0)
    show_pp(mean_att, title=f"Mean attention for head {LAYER}.{HEAD}")

#%% [markdown]
# Look into compensation in both cases despite it seeming very different

cache = {}
model.reset_hooks()
model.cache_some(
    cache,
    lambda x: "attn.hook_result" in x or "mlp_out" in x,
    suppress_warning=True,
    # device=device,
)
logits, loss = model(
    rand_tokens_control, return_type="both", loss_return_per_token=True
).values()

# top_heads = [
#     (9, 9),
#     (9, 6),
#     (10, 1),
#     (7, 10),
#     (10, 0),
#     (11, 9),
#     (7, 2),
#     (6, 9),
#     # (10, 6),
#     # (10, 3),
# ]

top_heads = [
    (9, 6),
    (10, 0),
    (7, 2),
    (9, 9),
    (7, 10),
    (9, 1),
    (11, 5),
    (6, 9),
    (10, 1),
    (11, 9),
    (8, 1),
    (10, 6),
    (5, 1),
    (10, 10),
    (10, 3),
]

top_heads = [
    (6, 1),
    (8, 1),
    (6, 6),
    (8, 0),
    (8, 8),
]

top_heads = induct_heads
# top_heads = [(5, 1), (7, 2), (7, 10), (6, 9), (5, 5)]

hooks = {}

# top_heads = [
#     (layer, head_idx)
#     for layer in range(model.cfg.n_layers)
#     for head_idx in [None] + list(range(model.cfg.n_heads))
# ]

skipper = 0
# top_heads = max_2d(results, 20)[0][skipper:]


# def zero_all(z, act, hook):
#     z[:] = 0
#     return z


def random_patching(z, act, hook):
    b = z.shape[0]
    z[torch.arange(b)] = act[torch.randperm(b)]
    return z


for layer, head_idx in top_heads:
    hook_name = f"blocks.{layer}.attn.hook_result"
    if head_idx is None:
        hook_name = f"blocks.{layer}.hook_mlp_out"

    hooks[(layer, head_idx)] = (
        hook_name,
        get_act_hook(
            random_patching,
            alt_act=cache[hook_name],
            idx=head_idx,
            dim=2 if head_idx is not None else None,
            name=hook_name,
        ),
    )
model.reset_hooks()

#%% [markdown]
# Line graph

tot = len(induct_heads) + 1
# tot=5

initial_loss = model(
    rand_tokens_repeat, return_type="both", loss_return_per_token=True
)["loss"][:, -seq_len // 2 :].mean()

# induct_heads = max_2d(torch.tensor(induction_scores_array), tot)[0]
# induct_heads = [(6, 1), (8, 0), (6, 11), (8, 1), (8, 8)]

hooks = {head:hooks[head] for head in induct_heads}

def get_random_subset(l, size):
    return [l[i] for i in sorted(random.sample(range(len(l)), size))]

ys = []
ys2 = []
no_iters = 30
max_len = len(induct_heads)

# metric = loss_metric
metric = logits_metric
mode = "random subset"
# mode = "decreasing"


for subset_size in tqdm(range(len(induct_heads) + 1)):
    model.reset_hooks()

    curv = 0
    curw = initial_loss.item()  # "EXPECTED" increase
    for _ in range(30):
        model.reset_hooks()

        ordered_hook_list = []
        if mode == "random subset":
            ordered_hook_list = get_random_subset(list(hooks.items()), subset_size)
        elif mode == "decreasing":
            ordered_hook_list = list(hooks.items())[:subset_size]
        else:
            raise ValueError()

        for hook in ordered_hook_list:
            model.add_hook(*hook[1])
            # curw += results[hook[0]].item()

        cur_metric = metric(
            model, rand_tokens_repeat, seq_len,
        )
        # print(f"Layer {layer}, head {head_idx}: {loss.mean().item()}")

        curv += cur_metric
    curv /= no_iters
    curw /= no_iters
    ys.append(curv)
    # curw = (
    #     initial_loss.item()
    #     + torch.sum(max_2d(results, 15)[1][skipper : skipper + subset_size]).item()
    # )
    curw = curv
    ys2.append(curw)

# plot the results
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=list(range(0, max_len+1)),
        y=ys,
        mode="lines+markers",
        name="Top k heads removed" if mode == "decreasing" else "k random heads removed",
        line=dict(color="Black", width=1),
    )
)
# fig.add_trace(
#     go.Scatter(
#         x=list(range(0, max_len+1)),
#         y=ys2,
#         mode="lines+markers",
#         name="Sum of direct effects",
#         line=dict(color="Red", width=1),
#     )
# )

start_x = 0
start_y = ys[0]
end_x = tot - 1
end_y = ys[tot - 1]

if mode == "decreasing":
    contributions = {head:(results[head].item()) for head in induct_heads}
    contributions_sum = sum(contributions.values())
    for head in induct_heads: contributions[head] /= contributions_sum

    expected_x = list(range(tot))
    expected_y = [start_y]
    y_diff = end_y - start_y
    for head in induct_heads:
        expected_y.append(expected_y[-1] + y_diff * contributions[head])

    fig.add_trace(
        go.Scatter(
            x=expected_x,
            y=expected_y,
            mode="lines+markers",
            name="Expected",
            line=dict(color="Blue", width=1),
        )
    )

expected_x_2 = list(range(tot))
expected_y_2 = [start_y]

for head in induct_heads:
    expected_y_2.append(expected_y_2[-1] + results[head])

fig.add_trace(
    go.Scatter(
        x=expected_x_2,
        y=expected_y_2,
        mode="lines+markers",
        name="Sum the independent effects",
        line=dict(color="Green", width=1),
    )
)


# add the line from (0, ys[0]) to (tot-1, ys[tot-1])
fig.add_trace(
    go.Scatter(
        x=[0, max_len],
        y=[ys[0], ys[-1]],
        mode="lines",
        name="Linear from start to end",
        line=dict(color="Blue", width=1),
    )
)

# add x axis labels
fig.update_layout(
    xaxis_title="Number of heads removed (k)",
    yaxis_title="Logits on correct",
    title=f"Effect of removing heads on correct logits ({mode})",
)

#%% [markdown]

for tens in [froze_results, froze_mlp, flow_results, flow_mlp]:
    print(torch.sum(tens))

#%% [markdown]
# Induction compensation

from ioi_utils import compute_next_tok_dot_prod
import torch.nn.functional as F

IDX = 0


def zero_ablate(hook, z):
    return torch.zeros_like(z)


head_mask = torch.empty((model.cfg.n_layers, model.cfg.n_heads), dtype=torch.bool)
head_mask[:] = False
head_mask[5, 5] = True
head_mask[6, 9] = False

attn_head_mask = head_mask


def filter_value_hooks(name):
    return name.split(".")[-1] == "hook_v"


def compute_logit_probs(rand_tokens_repeat, model):
    induction_logits = model(rand_tokens_repeat)
    induction_log_probs = F.log_softmax(induction_logits, dim=-1)
    induction_pred_log_probs = torch.gather(
        induction_log_probs[:, :-1].cuda(), -1, rand_tokens_repeat[:, 1:, None].cuda()
    )[..., 0]
    return induction_pred_log_probs[:, seq_len:].mean().cpu().detach().numpy()


compute_logit_probs(rand_tokens_repeat, model)

#%% [markdown]
# Get some signal on the eigenvalues of all of the models' heads

names = []
metrics = []
colors = []

for layer in range(6, 9, 2):
    for head_idx in range(12):
        WQ = model.state_dict()[f"blocks.{layer}.attn.W_Q"][head_idx] # 12 768 64
        WK = model.state_dict()[f"blocks.{layer}.attn.W_K"][head_idx] # 12 768 64
        WQK = FactoredMatrix(WQ, WK.T)
        eigs = WQK.eigenvalues.cpu()
        reals = [point.real for point in eigs]
        imags = [point.imag for point in eigs]
        mags = [point.abs() for point in eigs]
        names.append(f"Layer {layer}, head {head_idx}")
        metrics.append(sum(reals) / sum(mags))
        if (layer, head_idx) in induct_heads: # is induction
            colors.append("blue")
        elif (layer, head_idx) in neg_heads: # is negative 
            colors.append("red")
        else: # is other
            colors.append("black")

# plot a bar chart of the eigenvalues wiht 
for color in ["blue", "red", "black"]:
    inds = [i for i, c in enumerate(colors) if c == color]
    plt.bar([names[i] for i in inds], [metrics[i] for i in inds], color=color)
# add legend
plt.legend(["Induction", "Negative", "Other"])
# make the x axis labels vertical
plt.xticks(rotation=90)
# make frame big
plt.gcf().set_size_inches(20, 10)

#%% [markdown]
# Ambitious: the backwards pass of interp

nodes = []
cur_pos = rand_tokens_repeat.shape[1] - 1

for layer in range(model.cfg.n_layers):
    for head_idx in [None] + list(range(model.cfg.n_heads)):
        assert len(rand_tokens_repeat.shape) == 2, ("rand_tokens_repeat must be 2D", rand_tokens_repeat.shape)
        
        for pos in [cur_pos]:
            """
            For a first prototype, let's literally deal with the last index only
            """
            nodes.append((get_hook(layer, head_idx), pos))

final_node = ((f"blocks.{model.cfg.n_layers-1}.hook_resid_post", None), cur_pos)
nodes.append(final_node)
important_indices = {final_node}

#%% [markdown]

# make a JSON file that's an empty list 
import json
fname = f"jsons/{ctime2()}.json"
with open(fname, "w") as f:
    json.dump([], f)

def jprint(*args):
    # add this string to the JSON file
    string = " ".join([str(arg) for arg in args])
    print(string)
    with open(fname, "r") as f:
        data = json.load(f)
    data.append(string)
    with open(fname, "w") as f:
        json.dump(data, f)

very_initial_metric = metric(model, rand_tokens_repeat)
jprint("The initial metric is", very_initial_metric)

while len(nodes) > 0:
    hook, pos = nodes.pop()
    if (hook, pos) not in important_indices:
        jprint(f"Skipping {(hook, pos)} as it wasn't marked as important")
        continue
    jprint(f"Working on {(hook, pos)}...")

    hook_name, head_idx = hook

    # do direct patching going into this hook
    # later we will have to do a branch statement
    layer = model.cfg.n_layers if "resid_post" in hook_name else get_number_in_string(hook_name)

    if head_idx is not None:
        warnings.warn("We are yet to implement attention")
        continue

    head_results, mlp_results, initial_metric = ppa_multiple(
        model=model, 
        tokens=rand_tokens_repeat, 
        patch_tokens=rand_tokens_control,
        attention_max_layer=layer, 
        mlp_max_layer=layer-1,
        receiver_hooks=[hook], 
        metric=logits_metric,
        # TODO add pos. For now we just YOLO the whole thing
    )

    jprint("The current initial metric is", initial_metric)
    show_pp(head_results.T, title="HEAD RESULTS")
    show_pp(mlp_results, title="MLP RESULTS")

    # layer = int(input("Give me the relevant MLP"))
    threshold = 0.1
    for layer in range(model.cfg.n_layers):
        for head_idx in range(model.cfg.n_heads):
            if abs(head_results[layer, head_idx]) > threshold:
                important_indices.add((get_hook(layer, head_idx), pos))
        if abs(mlp_results[layer]) > threshold:
            important_indices.add((get_hook(layer, None), pos))

    important_indices.add((get_hook(layer, head_idx), cur_pos))



#%% [markdown]
# Arthur's scratchpad for the backwards pass of interp

# def backwards_pass():
#     """
#     `cur_state` is a (node, position) pair
#     """

if True:
    if is_attn_head:
        # do patching of Q and K and V (at all positions?)
        # all positions is a lot of iterations, man 


    else:
        # (is MLP)

        # do patching of the inputs to me
        # zoom in on the things that are important

        # ... hook the output of this head to only the important details
        # (man, we really need circuit rewrites cri)
        # do this with like plussing and minusing and stuff?