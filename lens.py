#%%
import os
if "arthur" in os.environ["USER"]:
    from interp.circuit.projects.acdc.utils import ioi_prompts, abc_prompts
from easy_transformer.EasyTransformer import EasyTransformer  # type: ignore
from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
import os
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from functools import partial
from pathlib import PurePath as PP
from copy import copy
from tqdm import tqdm
from tabulate import tabulate
import numpy as np
np.random.seed(1726)
import attrs
from attrs import frozen
import torch
import torch as t
import einops
import jax
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
jax.config.update("jax_platform_name", "cpu")
from interp.tools.interpretability_tools import (
    begin_token,
    get_interp_tokenizer,
    print_max_min_by_tok_k_torch,
    single_tokenize,
    toks_to_string_list,
)
from interp.circuit.scope_rewrites import basic_factor_distribute
from interp.circuit.projects.punct.rewrites import (
    approx_head_diag_masks_a0,
    expand_probs_a1_more,
    get_a1_probs_deriv_expand,
    get_trivial_expand_embeds_and_attn,
    expand_and_factor_log_probs,
    get_log_probs_cov_expand,
)
from interp.circuit.projects.estim_helper import EstimHelper
from interp.circuit.projects.interp_utils import (
    ChildDerivInfo,
    get_items,
    add_path,
    print_for_scope,
    run_scope_estimate,
)
#%%
model = EasyTransformer.from_pretrained("EleutherAI/gpt-neo-125M")
# model = EasyTransformer.from_pretrained("gpt2")
#%%
# DATA SHIT
data_rrfs = os.path.expanduser(f"~/rrfs/pretraining_datasets/owt_tokens_int16/0.pt")
data_suffix = "name_data/data-2022-07-30.pt"
data_local = os.path.expanduser(f"~/{data_suffix}")
try:
    data_full = torch.load(data_local)
except FileNotFoundError:
    data_full = torch.load(data_rrfs)
toks = data_full["tokens"].long() + 32768
lens = data_full["lens"].long()
tzr = get_interp_tokenizer()
def d(tokens, tokenizer=tzr):
    print(tokenizer.decode(tokens))
def e():
    torch.cuda.empty_cache()
#%%

cur_place = 0
actual_toks = []
DATASET_SIZE = 100
for idx in tqdm(range(min(DATASET_SIZE * 5, lens.shape[0]))): # x3 to get enough induction
    cur_len = min(200, lens[idx].item())
    cur_toks_tensor = (toks[cur_place:cur_place+cur_len])
    cur_toks = [model.tokenizer.bos_token_id] + list(cur_toks_tensor)
    actual_toks.append(torch.tensor(cur_toks).long())
    cur_place += lens[idx].item()
#%%

cache = {}
model.cache_all(cache)
model.set_use_attn_result(True)
a=model(actual_toks[0], return_type="loss")
cache_keys = list(cache.keys())
del cache
e()

#%%

def get_act_hook(
    fn,
    alt_act=None,
    idx=None,
    dim=None,
    name=None,
    message=None,
    metadata=None,
):
    """Return an hook that modify the activation on the fly. alt_act (Alternative activations) is a tensor of the same shape of the z.
    E.g. It can be the mean activation or the activations on other dataset."""
    if alt_act is not None:

        def custom_hook(z, hook):
            hook.ctx["idx"] = idx
            hook.ctx["dim"] = dim
            hook.ctx["name"] = name
            hook.ctx["metadata"] = metadata

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
            hook.ctx["metadata"] = metadata

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

def fn(z, hook):
    z[:] = 0.0
    return z

tens = torch.zeros((12, 12))

for i in tqdm(range(12)):
    model.reset_hooks()
    for layer_idx in range(i+1, 11):            
        for name in [f"blocks.{layer_idx}.hook_mlp_out", f"blocks.{layer_idx}.attn.hook_result"]:
            # if name == "blocks.10.hook_mlp_out":
            #     continue
            hook = get_act_hook(
                fn=fn,
                alt_act=None,
                idx=None,
                dim=None,
            )
            model.add_hook(name, hook)

    for j in range(10):
        loss = model(actual_toks[j], return_type="loss", per_token=False)
        tens[i, j] = loss.item()
        del loss
        torch.cuda.empty_cache()
#%%

tens2 = tens.clone()
tens2[:6, :] = 0.0

show_pp(
    tens2[:, :10],
    xlabel="Dataset index",
    ylabel="Last layer not zero ablated",
    title="GPT-Neo 125M", #  (not ablating MLP10, MLP11 or attention on layer 11)",
)

#%%
import os
import einops
import plotly.express as px
import torch
from typing import Dict

def show_pp(
    m,
    xlabel="",
    ylabel="",
    title="",
    bartitle="",
    animate_axis=None,
    highlight_points=None,
    highlight_name="",
    return_fig=False,
    show_fig=True,
    **kwargs,
):
    """
    Plot a heatmap of the values in the matrix `m`
    """

    if animate_axis is None:
        fig = px.imshow(
            m,
            title=title if title else "",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            **kwargs,
        )

    else:
        fig = px.imshow(
            einops.rearrange(m, "a b c -> a c b"),
            title=title if title else "",
            animation_frame=animate_axis,
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            **kwargs,
        )

    fig.update_layout(
        coloraxis_colorbar=dict(
            title=bartitle,
            thicknessmode="pixels",
            thickness=50,
            lenmode="pixels",
            len=300,
            yanchor="top",
            y=1,
            ticks="outside",
        ),
    )

    if highlight_points is not None:
        fig.add_scatter(
            x=highlight_points[1],
            y=highlight_points[0],
            mode="markers",
            marker=dict(color="green", size=10, opacity=0.5),
            name=highlight_name,
        )

    fig.update_layout(
        yaxis_title=ylabel,
        xaxis_title=xlabel,
        xaxis_range=[-0.5, m.shape[1] - 0.5],
        showlegend=True,
        legend=dict(x=-0.1),
    )
    if highlight_points is not None:
        fig.update_yaxes(range=[m.shape[0] - 0.5, -0.5], autorange=False)
    if show_fig:
        fig.show()
    if return_fig:
        return fig

# %%
