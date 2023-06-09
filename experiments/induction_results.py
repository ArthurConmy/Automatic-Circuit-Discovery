#!/usr/bin/env python3

import pandas as pd
import numpy as np
import plotly.express as px
import wandb

import hashlib
import os

import plotly.io as pio


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

ACDC_GROUP = "adria-induction-3"
# ACDC_GROUP = "adria-docstring3"
SP_GROUP = "reset-with-nll-21"
# SP_GROUP = "docstring3"

# %%

api = wandb.Api()
all_runs = api.runs(path="remix_school-of-rock/acdc", filters={"group": ACDC_GROUP})

df = pd.DataFrame()

total = len(all_runs)
failed = 0

for r in all_runs:
    try:
        d = {k: r.summary[k] for k in ["cur_metric", "test_specific_metric", "num_edges"]}
    except KeyError:
        failed+=1
    else:
        idx = int(r.name.split("-")[-1])
        df = pd.concat([df, pd.DataFrame(d, index=[idx])])

assert failed/total < 0.5

# %%

thresholds = 10 ** np.linspace(-2, 0.5, 21)

i = 0
for reset_network in [0, 1]:
    for zero_ablation in [0, 1]:
        for loss_type in ["kl_div", "nll", "match_nll"]:
            for threshold in thresholds:
                df.loc[i, "reset_network"] = reset_network
                df.loc[i, "zero_ablation"] = zero_ablation
                df.loc[i, "loss_type"] = loss_type
                df.loc[i, "threshold"] = threshold

                i += 1

df.loc[:, "alg"] = "acdc"
df.loc[:, "num_examples"] = 50

# %% Now subnetwork-probing runs

sp_runs = []

all_runs = api.runs(path="remix_school-of-rock/induction-sp-replicate", filters={"group": SP_GROUP})
for r in all_runs:
    try:
        cfg = {k: r.config[k] for k in ["reset_subject", "zero_ablation", "loss_type", "lambda_reg", "num_examples"]}
        d = {k: r.summary[k] for k in ["number_of_edges", "specific_metric", "test_specific_metric", "specific_metric_loss"]}
    except KeyError:
        continue
    cfg["reset_network"] = cfg["reset_subject"]
    del cfg["reset_subject"]
    cfg["num_edges"] = d["number_of_edges"]
    cfg["cur_metric"] = d["specific_metric"] / r.config["n_loss_average_runs"]
    cfg["test_specific_metric"] = d["test_specific_metric"] / r.config["n_loss_average_runs"]
    cfg["alg"] = "subnetwork-probing"
    sp_runs.append(cfg)


i = df.index.max() + 1
for r in sp_runs:
    df = pd.concat([df, pd.DataFrame(r, index=[i])])
    i += 1

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
    facet_row="loss_type",
    facet_col_wrap=2,
    hover_data=["threshold", "lambda_reg"],
    title="Induction, TRAIN metric",
)
fig.show()

# %%

fig = px.scatter(
    df,
    x="num_edges",
    y="test_specific_metric",
    color="color",
    color_discrete_map={
        "acdc-reset=0": "red",
        "acdc-reset=1": "blue",
        "subnetwork-probing-reset=0": "orange",
        "subnetwork-probing-reset=1": "green",
    },
    facet_col="zero_ablation",
    facet_row="loss_type",
    facet_col_wrap=2,
    hover_data=["threshold", "lambda_reg"],
    title="Induction, TEST metric",
)
fig.show()
