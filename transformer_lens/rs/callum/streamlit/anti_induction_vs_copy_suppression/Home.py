import streamlit as st
import numpy as np
import pandas as pd
import torch as t
from pathlib import Path
import pickle
import plotly.express as px

# st.set_page_config(layout="wide")

param_sizes = {
    "gpt2-small": "85M",
    "gpt2-medium": "302M",
    "gpt2-large": "708M",
    "gpt2-xl": "1.5B",
    "distillgpt2": "42M",
    "opt-125m": "85M",
    "opt-1.3b": "1.2B",
    "opt-2.7b": "2.5B",
    "opt-6.7b": "6.4B",
    "opt-13b": "13B",
    "opt-30b": "30B",
    "opt-66b": "65B",
    "gpt-neo-125m": "85M",
    "gpt-neo-1.3b": "1.2B",
    "gpt-neo-2.7b": "2.5B",
    "gpt-j-6B": "5.6B",
    "gpt-neox-20b": "20B",
    "stanford-gpt2-small-a": "85M",
    "stanford-gpt2-small-b": "85M",
    "stanford-gpt2-small-c": "85M",
    "stanford-gpt2-small-d": "85M",
    "stanford-gpt2-small-e": "85M",
    "stanford-gpt2-medium-a": "302M",
    "stanford-gpt2-medium-b": "302M",
    "stanford-gpt2-medium-c": "302M",
    "stanford-gpt2-medium-d": "302M",
    "stanford-gpt2-medium-e": "302M",
    "pythia-70m": "19M",
    "pythia-160m": "85M",
    "pythia-410m": "302M",
    "pythia-1b": "5M",
    "pythia-1.4b": "1.2B",
    "pythia-2.8b": "2.5B",
    "pythia-6.9b": "6.4B",
    "pythia-12b": "11B",
    "pythia-70m-deduped": "19M",
    "pythia-160m-deduped": "85M",
    "pythia-410m-deduped": "302M",
    "pythia-1b-deduped": "805M",
    "pythia-1.4b-deduped": "1.2B",
    "pythia-2.8b-deduped": "2.5B",
    "pythia-6.9b-deduped": "6.4B",
    "pythia-12b-deduped": "11B",
    "solu-4l": "13M",
    "solu-6l": "42M",
    "solu-8l": "101M",
    "solu-10l": "197M",
    "solu-12l": "340M",
    "solu-4l-pile": "13M",
    "solu-6l-pile": "42M",
    "solu-8l-pile": "101M",
    "solu-10l-pile": "197M",
    "solu-12l-pile": "340M",
    "solu-1l": "3.1M",
    "solu-2l": "6.3M",
    "solu-3l": "9.4M",
    "solu-4l": "13M",
    "solu-6l": "42M",
    "solu-8l": "101M",
    "solu-10l": "197M",
    "solu-12l": "340M",
    "gelu-1l": "3.1M",
    "gelu-2l": "6.3M",
    "gelu-3l": "9.4M",
    "gelu-4l": "13M",
    "attn-only-1l": "1.0M",
    "attn-only-2l": "2.1M",
    "attn-only-3l": "3.1M",
    "attn-only-4l": "4.2M",
    "attn-only-2l-demo": "2.1M",
    "solu-1l-wiki": "3.1M",
    "solu-4l-wiki": "13M",
}
def get_size(model_name):
    size_str = param_sizes[model_name]
    if size_str.endswith("M"):
        size = int(1e6 * float(size_str[:-1]))
    elif size_str.endswith("B"):
        size = int(1e9 * float(size_str[:-1]))
    else:
        raise Exception
    return size

RESULTS_DIR = Path("model_results")
RESULTS = sorted(RESULTS_DIR.iterdir(), key=lambda x: x.stem)
# ! TODO - figure out why OPT is weird, no copy suppression anywhere
MODEL_NAMES = [result.stem.replace("scores_", "") for result in RESULTS]

min_size = int(min([get_size(model_name) for model_name in MODEL_NAMES]) // 1e6)
max_size = int(max([get_size(model_name) for model_name in MODEL_NAMES]) // 1e6)

def plot_all_results(negneg=False, showtext=False, fraction=False, categories="all", size_range=(min_size, max_size)):
    results_copy_suppression_ioi = []
    results_anti_induction = []
    model_names = []
    head_names = []
    fraction_list = []
    num_params = []

    for file in RESULTS:
        with open(file, "rb") as f:
            model_scores: t.Tensor = pickle.load(f)
            model_name = file.stem.replace("scores_", "")

            for layer in range(model_scores.size(1)):
                for head in range(model_scores.size(2)):
                    results_copy_suppression_ioi.append(model_scores[0, layer, head].item())
                    results_anti_induction.append(model_scores[1, layer, head].item())
                    model_names.append(model_name)
                    head_names.append(f"{layer}.{head}")
                    fraction_list.append((layer + 1) / model_scores.size(1))
                    num_params.append(get_size(model_name))

    df = pd.DataFrame({
        "results_copy_suppression_ioi": results_copy_suppression_ioi,
        "results_anti_induction": results_anti_induction,
        "model_names": model_names,
        "head_names": head_names,
        "head_and_model_names": [f"{model_names[i]}<br>{head_names[i]}" for i in range(len(model_names))],
        "fraction_list": fraction_list,
        "num_params": num_params,
    })

    if negneg:
        is_neg = [i for i, (x, y) in enumerate(zip(results_copy_suppression_ioi, results_anti_induction)) if x < 0 and y < 0]
        df = df.iloc[is_neg]

    if categories == "none":
        df = df[df["model_names"] == ""] # filter everything out
    elif categories != "all":
        df = df[[categories in name for name in df["model_names"]]]

    df = df[df["num_params"] >= size_range[0] * 1e6]
    df = df[df["num_params"] <= size_range[1] * 1e6]

    fig = px.scatter(
        df,
        x="results_copy_suppression_ioi", y="results_anti_induction", color='model_names' if not(fraction) else "fraction_list", 
        hover_data=["model_names", "head_names"], text="head_and_model_names" if showtext else None,
        title="Anti-Induction Scores (repeated random tokens) vs Copy-Suppression Scores (IOI)",
        labels={"results_copy_suppression_ioi": "Copy-Suppression", "results_anti_induction": "Anti-Induction"},
        height=550,
        color_continuous_scale=px.colors.sequential.Rainbow if fraction else None,
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    fig.update_traces(textposition='top center')
    # fig.update_layout(paper_bgcolor='rgba(255,244,214,0.5)', plot_bgcolor='rgba(255,244,214,0.5)')
    return fig


st.markdown(
r"""
# Anti-Induction vs Copy-Suppression

The plot below shows the scores of various heads for anti-induction and copy-suppression.

**Anti-induction** = what is the average direct effect on the logits for token `B`, resulting just from the head's attention from the second instance of token `A` back to the first instance of token `B` (with the random repeating sequence `AB...AB`)?

**Copy suppression** = what is the average direct effect on the logits for the `IO` token, resulting just from the head's attention from `end` to `IO`?
""")

negneg = st.checkbox("Filter for only neg-neg quadrant")
showtext = st.checkbox("Show head names as annotations")
fraction = st.checkbox("Color by fraction through model")
categories = st.radio(
    "Which class of models to show:", 
    ["all", "none", "gpt", "opt", "pythia", "solu"]
)
size_range = st.slider(
    label="Filter by number of params in model",
    min_value=min_size,
    max_value=max_size, 
    value=(min_size, max_size), 
    step=1,
    format='%2fM'
)

fig = plot_all_results(negneg=negneg, showtext=showtext, fraction=fraction, categories=categories, size_range=size_range)

st.plotly_chart(fig, use_container_width=True)