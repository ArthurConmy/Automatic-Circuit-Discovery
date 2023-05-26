# %%

from IPython import get_ipython
if get_ipython() is not None:
    get_ipython().magic('load_ext autoreload')
    get_ipython().magic('autoreload 2')
    __file__ = '/Users/adria/Documents/2023/ACDC/Automatic-Circuit-Discovery/notebooks/plotly_roc_plot.py'

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

from notebooks.emacs_plotly_render import set_plotly_renderer
set_plotly_renderer("emacs")

# %%

parser = argparse.ArgumentParser()
parser.add_argument('--arrows', action='store_true', help='Include help arrows')

if get_ipython() is not None:
    args = parser.parse_args([])
else:
    args = parser.parse_args()

# %%

def pessimistic_auc(xs, ys):
    i = np.argsort(xs)
    xs = np.array(xs, dtype=np.float64)[i]
    ys = np.array(ys, dtype=np.float64)[i]

    assert np.all(np.diff(xs) >= 0), "not sorted"
    assert np.all(np.diff(ys) >= 0), "not monotonically increasing"

    # The slabs of the stairs
    area = np.sum((1 - xs) * ys)
    return area

assert pessimistic_auc([0, 1], [0, 1]) == 0.0
assert pessimistic_auc([0, 0.5, 1], [0, 0.5, 1]) == 0.25

# %%
DATA_DIR = Path(__file__).resolve().parent.parent / "acdc" / "media" / "plots_data"

all_data = {}

def dict_merge(dct, merge_dct):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.

    Copyright 2016-2022 Paul Durivage, licensed under Apache License https://gist.github.com/angstwad/bf22d1822c38a92ec0a9

    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    for k in merge_dct.keys():
        if (k in dct and isinstance(dct[k], dict) and isinstance(merge_dct[k], dict)):  #noqa
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]



for fname in os.listdir(DATA_DIR):
    if fname.endswith(".json"):
        with open(DATA_DIR / fname, "r") as f:
            data = json.load(f)
        dict_merge(all_data, data)

# %% Prevent mathjax
fig=px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
fig.write_image("/tmp/discard.pdf", format="pdf")
# time.sleep(1)



# %%

alg_names = {
    "16H": "HISP",
    "SP": "SP",
    "ACDC": "ACDC",
}

TASK_NAMES = {
    "ioi": "Circuit Recovery (IOI)",
    "tracr-reverse": "tracr-reverse",
    "tracr-proportion": "tracr-xproportion",
    "docstring": "Docstring",
    "greaterthan": "Greater-Than",
    "induction": "Induction",
}

measurement_names = {
    "kl_div": "KL divergence",
    "logit_diff": "Logit difference",
    "l2": "Mean squared error",
    "nll": "Negative log-likelihood",
    "docstring_metric": "Negative log-likelihood (Docstring)",
    "greaterthan": "Probability difference",
}


METRICS_FOR_TASK = {
    "ioi": ["kl_div", "logit_diff"],
    "tracr-reverse": ["l2", "l2"],
    "tracr-proportion": ["l2", "l2"],
    "induction": ["kl_div", "nll"],
    "docstring": ["kl_div", "docstring_metric"],
    "greaterthan": ["kl_div", "greaterthan"],
}


methods = ["ACDC", "SP", "HISP"]
colors = {
    "ACDC": "purple",  
    "SP": "green",
    "HISP": "yellow",
}

symbol = {
    "ACDC": "circle",
    "SP": "x",
    "HISP": "diamond",
}


x_names = {
    "fpr": "False positive rate (edges)",
    "tpr": "True positive rate (edges)",
    "precision": "Precision (edges)",
    "n_edges": "Number of edges",
    "test_kl_div": "KL(model, ablated)",
    "test_loss": "test loss",
}

def discard_non_pareto_optimal(points, cmp="gt"):
    ret = []
    for x, y in points:
        for x1, y1 in points:
            if x1 < x and getattr(y1, f"__{cmp}__")(y) and (x1, y1) != (x, y):
                break
        else:
            ret.append((x, y))
    return list(sorted(ret))


def make_fig(metric_idx=0, x_key="fpr", y_key="tpr", weights_type="trained", ablation_type="random_ablation", plot_type="roc"):
    this_data = all_data[weights_type][ablation_type]

    if plot_type in ["roc", "precision_recall"]:
        rows_cols_task_idx = [
            ((1, 1), "ioi"),
            ((1, 3), "tracr-reverse"),
            ((1, 4), "tracr-proportion"),
            ((2, 3), "docstring"),
            ((2, 4), "greaterthan"),
        ]
        specs=[[{"rowspan": 2, "colspan": 2}, None, {}, {}], [None, None, {}, {}]]
    else:
        rows_cols_task_idx = [
            ((1, 1), "ioi"),
            ((1, 2), "tracr-reverse"),
            ((1, 3), "tracr-proportion"),
            ((2, 1), "induction"),
            ((2, 2), "docstring"),
            ((2, 3), "greaterthan"),
        ]
        specs = [[{}]*3]*2

    rows_and_cols, task_idxs = list(zip(*rows_cols_task_idx))
    task_names = [TASK_NAMES[i] for i in task_idxs]

    fig = make_subplots(
        rows=2,
        cols=4 if plot_type in ["roc", "precision_recall"] else 3,
        # specs parameter is really cool, this argument needs to have same dimenions as the rows and cols
        specs=specs,
        print_grid=False,
        # subplot_titles=("First Subplot", "Second Subplot", "Third Subplot", "Fourth Subplot", "Fifth Subplot"),
        subplot_titles=tuple(task_names),
        x_title=x_names[x_key],
        y_title=x_names[y_key],
        # title_font=dict(size=8),
    )

    fig.update_annotations(font_size=12)

    min_score = 1e90
    max_score = -1e90
    for task_idx in task_idxs:
        for metric_name in METRICS_FOR_TASK[task_idx]:
            try:
                x_data = this_data[task_idx][metric_name]["ACDC"]["score"]
            except KeyError:
                continue
            if len(x_data) > 0:
                finites = [x for x in x_data if np.isfinite(x)]
                min_score = min(min_score, min(finites))
                max_score = max(max_score, max(finites))



    all_series = []
    for (row, col), task_idx in rows_cols_task_idx:
        for alg_idx, methodof in alg_names.items():
            metric_name = METRICS_FOR_TASK[task_idx][metric_idx]
            if plot_type == "metric_edges":
                y_key = "test_" + metric_name
            try:
                x_data = this_data[task_idx][metric_name][alg_idx][x_key]
                y_data = this_data[task_idx][metric_name][alg_idx][y_key]
                scores = this_data[task_idx][metric_name][alg_idx]["score"]
            except KeyError:
                x_data = []
                y_data = []
                scores = []

            if alg_idx == "SP":
                # Divide by number of loss runs. Fix earlier bug.
                if x_key.startswith("test_"):
                    x_data = [x / 20 for x in x_data]

                if y_key.startswith("test_"):
                    y_data = [y / 20 for y in y_data]


            points = list(zip(x_data, y_data))
            if y_key != "tpr":
                pareto_optimal = [] # list(sorted(points))  # Not actually pareto optimal but we want to plot all of them
            else:
                pareto_optimal = discard_non_pareto_optimal(points)
            others = [p for p in points if p not in pareto_optimal]



            auc = None
            if len(pareto_optimal):
                x_data, y_data = zip(*pareto_optimal)
                if plot_type == "roc":
                    auc = pessimistic_auc(x_data, y_data)
                fig.add_trace(
                    go.Scatter(
                        x=x_data,
                        y=y_data,
                        name=methodof,
                        mode="lines",
                        line=dict(shape="hv", color=colors[methodof]),
                        showlegend = False,
                    ),
                    row=row,
                    col=col,
                )

            test_kl_div = this_data[task_idx][metric_name][alg_idx]["test_kl_div"][1:-1]
            test_loss = this_data[task_idx][metric_name][alg_idx]["test_" + metric_name][1:-1]
            if alg_idx == "SP":
                test_kl_div = [x / 20 for x in test_kl_div]
                test_loss = [x / 20 for x in test_loss]

            if plot_type == "roc":
                all_series.append(pd.Series({
                    "task": task_idx,
                    "method": methodof,
                    "auc": auc,
                    "metric": metric_name,
                    "weights_type": weights_type,
                    "ablation_type": ablation_type,
                    "n_points": len(points),
                    "test_kl_div": np.mean(test_kl_div),
                    "test_kl_div_max": np.max(test_kl_div),
                    "test_kl_div_min": np.min(test_kl_div),
                    "test_loss": np.mean(test_loss),
                    "test_loss_max": np.max(test_loss),
                    "test_loss_min": np.min(test_loss),
                }))

            if task_idx == "induction" and plot_type == "kl_edges":
                assert auc is None
                all_series.append(pd.Series({
                    "task": task_idx,
                    "method": methodof,
                    "auc": None,
                    "metric": metric_name,
                    "weights_type": weights_type,
                    "ablation_type": ablation_type,
                    "n_points": len(points),
                    "test_kl_div": np.mean(test_kl_div),
                    "test_kl_div_max": np.max(test_kl_div),
                    "test_kl_div_min": np.min(test_kl_div),
                    "test_loss": np.mean(test_loss),
                    "test_loss_max": np.max(test_loss),
                    "test_loss_min": np.min(test_loss),
                }))


            if others:
                x_data, y_data = zip(*others)
                if not (np.isfinite(x_data[0]) and np.isfinite(y_data[0])):
                    x_data = x_data[1:]
                    y_data = y_data[1:]
                if not (np.isfinite(x_data[-1]) and np.isfinite(y_data[-1])):
                    x_data = x_data[:-1]
                    y_data = y_data[:-1]

                assert not np.any(~np.isfinite(x_data))
                assert not np.any(~np.isfinite(y_data))
            else:
                x_data, y_data = [None], [None]

            # print(task_idx, alg_idx, metric_name, len(x_data), len(y_data), plot_type)

            colorscale = px.colors.get_colorscale("Purples")
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=y_data,
                    name=methodof,
                    mode="markers",
                    line=dict(shape="hv", color=colors[methodof]),
                    showlegend = (row, col) == rows_and_cols[-2],
                    marker=dict(
                        size=7,
                        color=colors[methodof], # if alg_idx != "ACDC" else plotly.colors.sample_colorscale(colorscale, (np.clip(scores, min_score, max_score) - min_score)/(2*(max_score-min_score)) + 0.5),
                        symbol=symbol[methodof],
                    ),

                ),
                row=row,
                col=col,
            )

            # fig.update_layout(
            #     title_font=dict(size=20),
            #     title=plot_type,
            # )

            if (row, col) == rows_and_cols[0]:
                if plot_type == "roc" and args.arrows:
                    fig.add_annotation(
                        xref="x domain",
                        yref="y",
                        x=0.35, # end of arrow
                        y=0.65,
                        text="",
                        axref="x domain",
                        ayref="y",
                        ax=0.55,
                        ay=0.45,
                        arrowhead=2,
                        row = row,
                        col = col,
                    )
                    fig.add_annotation(
                        xref="x domain",
                        yref="y",
                        x=0.6, # end of arrow
                        y=0.7,
                        text="",
                        axref="x domain",
                        ayref="y",
                        ax=0.6,
                        ay=0.5,
                        arrowhead=2,
                        row = row,
                        col = col,
                    )
                    fig.add_annotation(
                        xref="x domain",
                        yref="y",
                        x=0.8, # end of arrow
                        y=0.4,
                        text="",
                        axref="x domain",
                        ayref="y",
                        ax=0.6,
                        ay=0.4,
                        arrowhead=2,
                        row = row,
                        col = col,
                    )
                    fig.add_annotation(text="More true components recovered",
                        xref="x", yref="y",
                        x=0.55, y=0.75, showarrow=False, font=dict(size=8), row=row, col=col)
                    fig.add_annotation(text="Better",
                        xref="x", yref="y",
                        x=0.4, y=0.5, showarrow=False, font=dict(size=12), row=row, col=col)
                    fig.add_annotation(text="More wrong components recovered",
                        xref="x", yref="y",
                        x=0.65, y=0.35, showarrow=False, font=dict(size=8), row=row, col=col) # TODO could add two text boxes

                if y_key in ["fpr", "tpr", "precision"]:
                    fig.update_yaxes(visible=True, row=row, col=col, tickangle=-45, dtick=0.25, range=[-0.05, 1.05]) # ???
                else:
                    fig.update_yaxes(visible=True, row=row, col=col, tickangle=-45)

                if x_key == "n_edges":
                    fig.update_xaxes(type='log', row=row, col=col)
                    # fig.update_yaxes(type='log', row=row, col=col)
                else:
                    fig.update_xaxes(dtick=0.25, range=[-0.05, 1.05], row=row, col=col)

                # # add label to x axis
                # fig.update_xaxes(title_text="False positive rate", row=row, col=col)
                # # add label to y axis
                # fig.update_yaxes(title_text="True positive rate", row=row, col=col)

                fig.update_layout(title_font=dict(size=1)) # , row=row, col=col)


            else:
                # If the subplot is not the large plot, hide its axes
                if y_key in ["fpr", "tpr", "precision"]:
                    fig.update_yaxes(visible=True, row=row, col=col, tickangle=-45, dtick=0.25, tickvals=[0, 0.25, 0.5, 0.75, 1.], ticktext=["0", "", "0.5", "", "1"], range=[-0.05, 1.05]) # ???
                else:
                    fig.update_yaxes(visible=True, row=row, col=col, tickangle=-45)

                if x_key == "n_edges":
                    fig.update_xaxes(type='log', row=row, col=col)
                    # fig.update_yaxes(type='log', row=row, col=col)
                else:
                    fig.update_xaxes(visible=True, row=row, col=col, tickvals=[0, 0.25, 0.5, 0.75, 1.], ticktext=["0", "", "0.5", "", "1"], range=[-0.05, 1.05])

                # smaller title font
                fig.update_layout(title_font=dict(size=20)) # , row=row, col=col)

            # add label to x axis

    # move legend to left
    fig.update_layout(
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="right",
            x=1.12,
            font=dict(size=8),
            bgcolor="rgba(0,0,0,0)",  # Set the background color to transparent
        ),
        title_font=dict(size=4),
    )

    scale = 1.2

    # No title,
    fig.update_layout(height=250*scale, width=scale*scale*500,
                      margin=dict(l=55, r=70, t=20, b=50)
                      )
    return fig, pd.concat(all_series, axis=1) if all_series else pd.DataFrame()

plot_type_keys = {
    "precision_recall": ("tpr", "precision"),
    "roc": ("fpr", "tpr"),
    "kl_edges": ("n_edges", "test_kl_div"),
    "metric_edges": ("n_edges", "test_loss"),
}

# %%
PLOT_DIR = DATA_DIR.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

all_dfs = []
for metric_idx in [0, 1]:
    for ablation_type in ["random_ablation", "zero_ablation"]:
        for weights_type in ["trained", "reset"]:  # Didn't scramble the weights enough it seems
            for plot_type in ["precision_recall", "roc", "kl_edges", "metric_edges"]:
                x_key, y_key = plot_type_keys[plot_type]
                fig, df = make_fig(metric_idx=metric_idx, weights_type=weights_type, ablation_type=ablation_type, x_key=x_key, y_key=y_key, plot_type=plot_type)
                if len(df):
                    all_dfs.append(df.T)
                    print(all_dfs[-1])

                metric = "kl" if metric_idx == 0 else "other"
                fig.write_image(PLOT_DIR / ("--".join([metric, weights_type, ablation_type, plot_type]) + ".pdf"))

pd.concat(all_dfs).to_csv(PLOT_DIR / "data.csv")
# %%

# Stefan
#   1 hour ago
# Very nice plots! Small changes
# 1st title should be IOI, "Cicuit Recovery" should be above or somewhere else
# [Minor] Unify xlim=ylim=[-0.01, 1.01] or so
# :raised_hands:
# 1

# x_key, y_key = plot_type_keys["kl_edges"]
# fig, _ = make_fig(metric_idx=0, weights_type="reset", ablation_type="zero_ablation", plot_type="kl_edges")
# fig.show()
