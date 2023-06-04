# %%
#
import plotly
import os
import numpy as np
import json
import wandb
import time
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from pathlib import Path
import plotly.express as px
import pandas as pd
import argparse
import plotly.colors as pc
from acdc.graphics import dict_merge, pessimistic_auc

from notebooks.emacs_plotly_render import set_plotly_renderer

set_plotly_renderer("emacs")


# %%

DATA_DIR = Path("acdc") / "media" / "plots_data"

all_data = {}

for fname in os.listdir(DATA_DIR):
    if fname.endswith(".json"):
        with open(DATA_DIR / fname, "r") as f:
            data = json.load(f)
        dict_merge(all_data, data)

# %%


def discard_non_pareto_optimal(points, cmp="gt"):
    ret = []
    for x, y in points:
        for x1, y1 in points:
            if x1 < x and getattr(y1, f"__{cmp}__")(y) and (x1, y1) != (x, y):
                break
        else:
            ret.append((x, y))
    return list(sorted(ret))


fig = make_subplots(rows=1, cols=2, subplot_titles=["ROC Curves"], column_widths=[0.95, 0.05], horizontal_spacing=0.03)

colorscales = {
    "ACDC": "Blues",
    "SP": "Greens",
}

for i, alg in enumerate(["ACDC", "SP"]):
    this_data = all_data["trained"]["random_ablation"]["ioi"]["logit_diff"][alg]
    x_data = this_data["edge_fpr"]
    y_data = this_data["edge_tpr"]
    scores = this_data["score"]

    log_scores = np.log10(scores)
    log_scores = np.nan_to_num(log_scores, nan=0.0, neginf=0.0, posinf=0.0)

    min_score = np.min(log_scores)
    max_score = np.max(log_scores)

    normalized_scores = (log_scores - min_score) / (max_score - min_score)
    normalized_scores[~np.isfinite(normalized_scores)] = 0.0

    points = list(zip(x_data, y_data))
    pareto_optimal = discard_non_pareto_optimal(points)

    methodof = "acdc"

    pareto_x_data, pareto_y_data = zip(*pareto_optimal)
    fig.add_trace(
        go.Scatter(
            x=pareto_x_data,
            y=pareto_y_data,
            name=methodof,
            mode="lines",
            line=dict(
                shape="hv",
                color=pc.sample_colorscale(pc.get_colorscale(colorscales[alg]), 0.7)[0],
            ),
            showlegend=False,
            hovertext=log_scores,
        ),
        row=1,
        col=1,
    )

    N_TICKS = 5
    tickvals = np.linspace(0, 1, N_TICKS)
    ticktext = 10 ** np.linspace(min_score, max_score, N_TICKS)

    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=y_data,
            name=methodof,
            mode="markers",
            showlegend=False,
            marker=dict(
                size=7,
                color=normalized_scores,
                colorscale=colorscales[alg],
                symbol="circle",
                # colorbar=dict(
                #     title="Log scores",
                #     tickvals=tickvals,  # positions for ticks
                #     ticktext=["%.2e" % i for i in ticktext],  # tick labels, formatted as strings
                #     thickness=5,
                #     # y=0.25,
                #     # len=0.5,
                #     x=1 + i * 0.1,
                # ),
                showscale=False,
            ),
        ),
        row=1,
        col=1,
    )
    nums = np.arange(200).reshape(2, 100).T.astype(float)
    nums[:20, :20] = np.nan
    fig.add_trace(
        go.Heatmap(
            z=nums,
            colorscale='Viridis',
            showscale=False,
        ),
        row=1,
        col=2,
        )

fig.update_xaxes(showline=False, zeroline=False, showgrid=False, row=1, col=2, showticklabels=False, ticks="")
fig.update_yaxes(showline=False, zeroline=False, showgrid=False, row=1, col=2, side="right")
fig.show()
