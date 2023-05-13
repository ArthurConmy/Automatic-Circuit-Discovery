#!/usr/bin/env python3

import pandas as pd
import numpy as np
import plotly.express as px
import wandb

import hashlib
import os

import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class EmacsRenderer(pio.base_renderers.ColabRenderer):
    save_dir = "ob-jupyter"
    base_url = f"http://localhost:8888/files"

    def to_mimebundle(self, fig_dict):
        html = super().to_mimebundle(fig_dict)["text/html"]

        mhash = hashlib.md5(html.encode("utf-8")).hexdigest()
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        fhtml = os.path.join(self.save_dir, mhash + ".html")
        with open(fhtml, "w") as f:
            f.write(html)

        return {"text/html": f'<a href="{self.base_url}/{fhtml}">Click to open {fhtml}</a>'}


pio.renderers["emacs"] = EmacsRenderer()


def set_plotly_renderer(renderer="emacs"):
    pio.renderers.default = renderer


set_plotly_renderer("emacs")

ACDC_GROUP = "adria-docstring3"
SP_GROUP = "docstring3"

# %%

api = wandb.Api()
all_runs = api.runs(path="remix_school-of-rock/acdc", filters={"group": ACDC_GROUP})

df = pd.DataFrame()
for r in all_runs:
    try:
        cfg = {k: r.config[k] for k in ["reset_network", "zero_ablation", "metric", "task", "threshold"]}
        d = {
            k: r.summary[k]
            for k in [
                "cur_metric",
                "num_edges",
                "test_docstring_metric",
                "test_docstring_stefan",
                "test_kl_div",
                "test_match_nll",
                "test_nll",
            ]
        }
    except KeyError as e:
        print("problems with run ", r.name, e)
        continue
    d = dict(**cfg, **d)
    d["alg"] = "acdc"

    idx = int(r.name.split("-")[-1])
    df = pd.concat([df, pd.DataFrame(d, index=[idx])])

# %% Now subnetwork-probing runs

start_idx: float = df.index.max() + 1

all_runs = api.runs(path="remix_school-of-rock/induction-sp-replicate", filters={"group": SP_GROUP})
for r in all_runs:
    try:
        cfg = {k: r.config[k] for k in ["reset_subject", "zero_ablation", "loss_type", "lambda_reg"]}
        d = {
            k: r.summary[k]
            for k in [
                "number_of_edges",
                "specific_metric",
                "test_docstring_metric",
                "test_docstring_stefan",
                "test_kl_div",
                "test_match_nll",
                "test_nll",
            ]
        }
    except KeyError as e:
        print("problems with run ", r.name, e)
        continue
    cfg["metric"] = cfg["loss_type"]
    del cfg["loss_type"]
    cfg["reset_network"] = cfg["reset_subject"]
    del cfg["reset_subject"]
    cfg["num_edges"] = d["number_of_edges"]
    cfg["cur_metric"] = d["specific_metric"] / r.config["n_loss_average_runs"]
    for k in d.keys():
        if k.startswith("test_"):
            cfg[k] = d[k] / r.config["n_loss_average_runs"]
    cfg["alg"] = "subnetwork-probing"

    idx = int(r.name.split("-")[-1]) + start_idx
    df = pd.concat([df, pd.DataFrame(cfg, index=[idx])])

# %%

df.loc[:, "color"] = df.apply(lambda x: f"{x['alg']}-reset={x['reset_network']:.0f}", axis=1)

# Scatter plot of num_edges vs cur_metric grouped by reset_network

fig = px.scatter(
    df,
    x="num_edges",
    y="cur_metric",
    color="color",
    color_discrete_map={
        "acdc-reset=0": "red",
        "acdc-reset=1": "blue",
        "subnetwork-probing-reset=0": "orange",
        "subnetwork-probing-reset=1": "green",
    },
    facet_col="zero_ablation",
    facet_row="metric",
    facet_col_wrap=2,
    hover_data=["threshold", "lambda_reg"],
    title="Induction, TRAIN metric",
)
fig.show()

# %%

for test_metric in ["test_docstring_metric", "test_docstring_stefan", "test_kl_div", "test_match_nll", "test_nll"]:
    fig = px.scatter(
        df,
        x="num_edges",
        y=test_metric,
        color="color",
        color_discrete_map={
            "acdc-reset=0": "red",
            "acdc-reset=1": "blue",
            "subnetwork-probing-reset=0": "orange",
            "subnetwork-probing-reset=1": "green",
        },
        facet_col="zero_ablation",
        facet_row="metric",
        facet_col_wrap=2,
        hover_data=["threshold", "lambda_reg"],
        title=f"Induction, {test_metric} metric",
    )
    fig.show()


# %% Scatter plot for train vs test of every metric
fig = make_subplots()

for test_metric in ["test_docstring_metric", "test_docstring_stefan", "test_kl_div", "test_match_nll", "test_nll"]:
    this_df = df[(df["metric"] == test_metric.lstrip("test_")) & (~df["zero_ablation"])]
    trace1 = go.Scatter(
        x=this_df["cur_metric"],
        y=this_df[test_metric],
        mode="markers",
        name=test_metric,
    )
    fig.add_trace(trace1)
fig.show()

# %% Compare each metric with the other metrics

for main_metric in ["test_docstring_metric", "test_docstring_stefan", "test_kl_div", "test_match_nll", "test_nll"]:
    fig = make_subplots()
    this_df = df[(df["metric"] == main_metric.lstrip("test_")) & (~df["zero_ablation"])]
    for test_metric in ["test_docstring_metric", "test_docstring_stefan", "test_kl_div", "test_match_nll", "test_nll"]:
        trace1 = go.Scatter(
            x=this_df["cur_metric"],
            y=this_df[test_metric],
            mode="markers",
            name=test_metric,
        )
        fig.add_trace(trace1)
    # set title
    fig.update_layout(
        title_text=f"Comparison of {main_metric} with other metrics",
        xaxis_title=main_metric,
        yaxis_title="other metrics",
    )
    fig.show()
