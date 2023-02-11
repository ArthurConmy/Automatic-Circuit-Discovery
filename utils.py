from transformer_lens.ioi_dataset import IOIDataset
import torch
import pandas
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from functools import partial
from collections import defaultdict
from typing import List, Dict, Tuple, Union, Optional, Callable
from torch import Tensor
from torch.nn import Module

def scatter_attention_and_contribution(
    model,
    ioi_dataset,
    layer_no,
    head_no,
    return_vals=False,
    return_fig=False,
):
    """
    Plot a scatter plot
    for each input sequence with the attention paid to IO and S
    and the amount that is written in the IO and S directions
    """

    n_heads = model.cfg.n_heads
    n_layers = model.cfg.n_layers
    model_unembed = model.unembed.W_U.detach().cpu()
    df = []
    cache = {}
    model.cache_all(cache)

    logits = model(ioi_dataset.toks.long())

    print(cache.keys())

    for i, prompt in enumerate(ioi_dataset.ioi_prompts):

        io_tok = model.tokenizer(" " + prompt["IO"])["input_ids"][0]
        s_tok = model.tokenizer(" " + prompt["S"])["input_ids"][0]
        toks = model.tokenizer(prompt["text"])["input_ids"]
        io_pos = toks.index(io_tok)
        s1_pos = toks.index(s_tok)
        s2_pos = toks[s1_pos + 1 :].index(s_tok) + (s1_pos + 1)
        assert toks[-1] == io_tok

        io_dir = model_unembed[:, io_tok].detach()
        s_dir = model_unembed[:, s_tok].detach()

        # model.reset_hooks() # should allow things to be done with ablated models

        for dire, posses, tok_type in [
            (io_dir, [io_pos], "IO"),
            (s_dir, [s1_pos, s2_pos], "S"),
        ]:
            prob = sum(
                [
                    cache[f"blocks.{layer_no}.attn.hook_pattern"][
                        i, head_no, ioi_dataset.word_idx["end"][i], pos
                    ]
                    .detach()
                    .cpu()
                    for pos in posses
                ]
            )
            resid = (
                cache[f"blocks.{layer_no}.attn.hook_result"][
                    i, ioi_dataset.word_idx["end"][i], head_no, :
                ]
                .detach()
                .cpu()
            )
            dot = torch.einsum("a,a->", resid, dire)
            df.append([prob, dot, tok_type, prompt["text"]])

    # most of the pandas stuff is intuitive, no need to deeply understand
    viz_df = pd.DataFrame(
        df, columns=[f"Attn Prob on Name", f"Dot w Name Embed", "Name Type", "text"]
    )
    fig = px.scatter(
        viz_df,
        x=f"Attn Prob on Name",
        y=f"Dot w Name Embed",
        color="Name Type",
        hover_data=["text"],
        color_discrete_sequence=["rgb(114,255,100)", "rgb(201,165,247)"],
        title=f"How Strong {layer_no}.{head_no} Writes in the Name Embed Direction Relative to Attn Prob",
    )

    if return_vals:
        return viz_df
    if return_fig:
        return fig
    else:
        fig.show()

def logit_diff(
    logits_or_model,
    dataset,
    logits=None,
    mean=True,
    item=True,
):
    if "HookedTransformer" in str(type(logits_or_model)):
        logits = logits_or_model(dataset.toks.long()).detach()
    else:
        logits = logits_or_model

    logits_on_end = logits[torch.arange(dataset.N), dataset.word_idx["end"]]
    
    logits_on_correct = logits_on_end[torch.arange(dataset.N), dataset.io_tokenIDs]
    logits_on_incorrect = logits_on_end[torch.arange(dataset.N), dataset.s_tokenIDs]

    logit_diff = logits_on_correct - logits_on_incorrect
    if mean:
        logit_diff = logit_diff.mean()
    if item:
        logit_diff = logit_diff.item()

    return logit_diff



def show_attention_patterns(
    model,
    heads,
    ioi_dataset,
    precomputed_cache=None,
    mode="val",
    title_suffix="",
    return_fig=False,
    return_mtx=False,
):  # Arthur edited for one of my experiments, things work well
    assert mode in [
        "attn",
        "val",
        "scores",
    ]  # value weighted attention or attn for attention probas
    assert isinstance(
        ioi_dataset, IOIDataset
    ), f"ioi_dataset must be an IOIDataset {type(ioi_dataset)}"
    prompts = ioi_dataset.sentences
    assert len(heads) == 1 or not (return_fig or return_mtx)

    for (layer, head) in heads:
        cache = {}

        good_names = [
            f"blocks.{layer}.attn.hook_pattern" + ("_scores" if mode == "scores" else "")
        ]
        if mode == "val":
            good_names.append(f"blocks.{layer}.attn.hook_v")
        if precomputed_cache is None:
            model.cache_some(
                cache=cache, names=lambda x: x in good_names
            )  # shape: batch head_no seq_len seq_len
            logits = model(ioi_dataset.toks.long())
        else:
            cache = precomputed_cache
        attn_results = torch.zeros(
            size=(ioi_dataset.N, ioi_dataset.max_len, ioi_dataset.max_len)
        )
        attn_results += -20

        for i, text in enumerate(prompts):
            # assert len(list(cache.items())) == 1 + int(mode == "val"), len(list(cache.items()))
            toks = ioi_dataset.toks[i]  # model.tokenizer(text)["input_ids"]
            current_length = len(toks)
            words = [model.tokenizer.decode([tok]) for tok in toks]
            attn = cache[good_names[0]].detach().cpu()[i, head, :, :]

            if mode == "val":
                vals = cache[good_names[1]].detach().cpu()[i, :, head, :].norm(dim=-1)
                cont = torch.einsum("ab,b->ab", attn, vals)

            fig = px.imshow(
                attn if mode in ["attn", "scores"] else cont,
                title=f"{layer}.{head} Attention" + title_suffix,
                color_continuous_midpoint=0,
                color_continuous_scale="RdBu",
                labels={"y": "Queries", "x": "Keys"},
                height=500,
            )

            fig.update_layout(
                xaxis={
                    "side": "top",
                    "ticktext": words,
                    "tickvals": list(range(len(words))),
                    "tickfont": dict(size=15),
                },
                yaxis={
                    "ticktext": words,
                    "tickvals": list(range(len(words))),
                    "tickfont": dict(size=15),
                },
            )
            if return_fig and not return_mtx:
                return fig
            elif return_mtx and not return_fig:
                attn_results[i, :current_length, :current_length] = (
                    attn[:current_length, :current_length].clone().cpu()
                )
            else:
                fig.show()

        if return_fig and not return_mtx:
            return fig
        elif return_mtx and not return_fig:
            return attn_results