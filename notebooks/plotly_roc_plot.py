# %%

from IPython import get_ipython
if get_ipython() is not None:
    get_ipython().magic('load_ext autoreload')
    get_ipython().magic('autoreload 2')
    if "mnt/ssd-0" not in __file__:
        __file__ = '/Users/adria/Documents/2023/ACDC/Automatic-Circuit-Discovery/notebooks/plotly_roc_plot.py'

import plotly
import os
import json
import wandb
import time
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from pathlib import Path
import plotly.express as px
import sklearn.metrics
import pandas as pd

from notebooks.emacs_plotly_render import set_plotly_renderer
#set_plotly_renderer("emacs")

# %%

DATA_DIR = Path(__file__).resolve().parent.parent / "acdc" / "media" / "arthur_plots_data"
DO_PARETO = True

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

# %%

alg_names = {
    "ACDC": "ACDC",
    "SP": "SP",
    "16H": "HISP",
}

TASK_NAMES = {
    "ioi": "Circuit Recovery (IOI)",
    "tracr-reverse": "Tracr (Reverse)",
    "tracr-proportion": "Tracr (Proportion)",
    "docstring": "Docstring",
    "greaterthan": "Greater Than",
}

measurement_names = {
    "kl_div": "KL Divergence",
    "logit_diff": "Logit difference",
    "l2": "MSE",
    "nll": "Negative log-likelihood",
    "docstring_metric": "Negative log-likelihood (Docstring)",
    "greaterthan": "p(larger) - p(smaller)",
}


METRICS_FOR_TASK = {
    "ioi": ["kl_div", "logit_diff"],
    "tracr-reverse": ["kl_div", "kl_div"],
    "tracr-proportion": ["kl_div", "l2"],
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
}

def discard_non_pareto_optimal(points):
    ret = []
    for x, y in points:
        for x1, y1 in points:
            if x1 < x and y1 > y and (x1, y1) != (x, y):
                break
        else:
            ret.append((x, y))
    return list(sorted(ret))


def make_fig(metric_idx=0, x_key="fpr", y_key="tpr", weights_type="trained", ablation_type="random_ablation"):
# if True:
#     metric_idx=0
#     x_key = "fpr"
#     y_key = "tpr"
#     weights_type = "trained"
#     ablation_type = "random_ablation"

    this_data = all_data[weights_type][ablation_type]

    task_idxs, task_names = zip(*TASK_NAMES.items())

    fig = make_subplots(
        rows=2,
        cols=4,
        # specs parameter is really cool, this argument needs to have same dimenions as the rows and cols
        specs=[[{"rowspan": 2, "colspan": 2}, None, {}, {}], [None, None, {}, {}]],
        print_grid=True,
        # subplot_titles=("First Subplot", "Second Subplot", "Third Subplot", "Fourth Subplot", "Fifth Subplot"),
        subplot_titles=tuple(task_names),
        x_title=x_names[x_key],
        y_title=x_names[y_key],
        # title_font=dict(size=8),
    )

    fig.update_annotations(font_size=12)

    rows_and_cols = [
        (1, 1),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
    ]

    df = pd.DataFrame()

    for task_idx, (row, col) in zip(task_idxs, rows_and_cols):
        for alg_idx, methodof in alg_names.items():
            metric_name = METRICS_FOR_TASK[task_idx][metric_idx]
            try:
                x_data = this_data[task_idx][metric_name][alg_idx][x_key]
                y_data = this_data[task_idx][metric_name][alg_idx][y_key]
            except KeyError as e:
                print(e, f"errored out...; {task_idx}, {metric_name}, {alg_idx}, {x_key}, {y_key}")
                x_data = []
                y_data = []
            # print(x_data, y_data)

            points = list(zip(x_data, y_data))
            pareto_optimal = discard_non_pareto_optimal(points)
            others = [p for p in points if p not in pareto_optimal]

            if len(pareto_optimal) and DO_PARETO:
                x_data, y_data = zip(*pareto_optimal)
                if plot_type == "roc":
                    auc = sklearn.metrics.auc(x_data, y_data)

                    # df = df.append({
                    #     "task": task_idx,
                    #     "method": methodof,
                    #     "auc": auc,
                    #     "metric": metric_name,
                    #     "weights_type": weights_type,
                    #     "ablation_type": ablation_type,  
                    # }, ignore_index=True) # what?

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

            if others:
                x_data, y_data = zip(*others)
            else:
                x_data, y_data = [None], [None]
            
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
                        color=colors[methodof],
                        symbol=symbol[methodof],
                    ),

                ),
                row=row,
                col=col,
            )

            fig.update_layout(
                title_font=dict(size=8),
            )

            # I don't think we should add the arrows, just describe what's better in the caption.
            if (row, col) == rows_and_cols[0]:
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

                fig.update_yaxes(visible=True, row=row, col=col, tickangle=-90, dtick=0.25, range=[-0.05, 1.05]) # ???
                fig.update_xaxes(dtick=0.25, range=[-0.05, 1.05])

                # # add label to x axis
                # fig.update_xaxes(title_text="False positive rate", row=row, col=col)
                # # add label to y axis
                # fig.update_yaxes(title_text="True positive rate", row=row, col=col)

                fig.update_layout(title_font=dict(size=1)) # , row=row, col=col)


            else:
                # If the subplot is not the large plot, hide its axes
                fig.update_xaxes(visible=True, row=row, col=col, tickvals=[0, 0.25, 0.5, 0.75, 1.], ticktext=["0", "", "0.5", "", "1"], range=[-0.05, 1.05])
                fig.update_yaxes(visible=True, row=row, col=col, tickangle=-90, dtick=0.25, tickvals=[0, 0.25, 0.5, 0.75, 1.], ticktext=["0", "", "0.5", "", "1"], range=[-0.05, 1.05]) # ???

                # smaller title font
                fig.update_layout(title_font=dict(size=10)) # , row=row, col=col)

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
    return fig, df


#%%

plot_type_keys = {
    "precision_recall": ("tpr", "precision"),
    "roc": ("fpr", "tpr"),
}

PLOT_DIR = DATA_DIR.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

all_dfs = []
for metric_idx in [0, 1]:
    for ablation_type in ["random_ablation", "zero_ablation"]:
        for plot_type in ["precision_recall", "roc"]:
            x_key, y_key = plot_type_keys[plot_type]
            fig, df = make_fig(metric_idx=metric_idx, ablation_type=ablation_type, x_key=x_key, y_key=y_key)
            all_dfs.append(df)

            metric = "kl" if metric_idx == 0 else "other"
            fig.write_image(PLOT_DIR / ("--".join([metric, ablation_type, plot_type]) + ".pdf"))

pd.concat(all_dfs).to_csv(PLOT_DIR / "data.csv")

# %%

# Stefan
#   1 hour ago
# Very nice plots! Small changes
# 1st title should be IOI, "Cicuit Recovery" should be above or somewhere else
# [Minor] Unify xlim=ylim=[-0.01, 1.01] or so
# :raised_hands:
# 1


