#%%

"""
This file makes several key figures in the paper and appendix, including the ROC plots and the pareto frontier plots
"""

import os
os.environ["ACCELERATE_DISABLE_RICH"] = "1"

import warnings
from IPython import get_ipython
from pathlib import Path
from notebooks.emacs_plotly_render import set_plotly_renderer

IS_ADRIA = not str(os.environ.get("CONDA_DEFAULT_ENV")).lower().startswith("arthur")

ipython = get_ipython()
if ipython is not None:
    ipython.magic('load_ext autoreload')
    ipython.magic('autoreload 2')

    initial_path = Path(get_ipython().run_line_magic('pwd', ''))
    if str(initial_path.stem) == "notebooks":
        initial_path = initial_path.parent
    __file__ = str(initial_path / "notebooks" / "make_plotly_plots.py")

    if IS_ADRIA:
          set_plotly_renderer("emacs")

import plotly
import numpy as np
import json
import wandb
from acdc.acdc_graphics import dict_merge, pessimistic_auc
import time
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from pathlib import Path
import plotly.express as px
import pandas as pd
import argparse
import plotly.colors as pc
import warnings

# %%

parser = argparse.ArgumentParser()
parser.add_argument('--arrows', action='store_true', help='Include help arrows')
parser.add_argument('--hisp-yellow', action='store_true', help='make HISP yellow')
parser.add_argument('--write-json', action='store_true', help='write json')

# Some ACDC tracr runs have its threshold go down to 1e-9 but that doesn't change results at all, we don't want to plot
# them.
parser.add_argument("--min-score", type=float, default=1e-6, help="minimum score cutoff for ACDC runs")

if get_ipython() is not None:
    args = parser.parse_args([])
else:
    args = parser.parse_args()

WRITE_JSON = args.write_json

# %%

GRIDCOLOR = "rgba(220,220,220,1)"
ZEROLINECOLOR = "rgba(180,180,180,1)"
THRESHOLD_ANNOTATION = r"$\tau,\lambda,\%$"

if IS_ADRIA or ipython is None or "arthur" in __file__:
    DATA_DIR = Path(__file__).resolve().parent.parent / "experiments" / "results" / "plots_data"

else:
    DATA_DIR = Path(__file__).resolve().parent.parent.parent / "experiments" / "results" / "plots_data"

X_TICKVALS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
all_data = {}

for fname in os.listdir(DATA_DIR):
    if fname.endswith(".json"):
        with open(DATA_DIR / fname, "r") as f:
            data = json.load(f)
        dict_merge(all_data, data)

# %% Prevent mathjax

fig=px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
fig.write_image("/tmp/discard.pdf", format="pdf")
time.sleep(1)

# %%

alg_names = {
    # "16H": "HISP", # TODO add back
    # "SP": "SP",
    "ACDC": "ACDC",
}

TASK_NAMES = {
    # "ioi": "Circuit Recovery (IOI)",
    "ioi": "IOI",
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


if args.hisp_yellow:
    colorscale_names = {
        "ACDC": "Purp_r",
        "SP": "Greens_r",
        "HISP": "YlOrBr_r",
    }
else:
    colorscale_names = {
        "ACDC": "YlOrRd_r",
        "SP": "Greens_r",
        "HISP": "Blues",
    }

colorscales = {}
for methodof, name in colorscale_names.items():
    color_list = pc.get_colorscale(name)
    # Add black to the minimum, so that we can represent -infinity
    colorscales[methodof] = [[0.0, "rgb(0, 0, 0)"],
                             # Make it not black as quickly as possible.
                             [1e-16, color_list[0][1]],
                             *color_list[1:]]

    if methodof == "HISP" and args.hisp_yellow:
        colorscales[methodof][1][1] = "rgb(255, 255, 0)"

# Want to sample here when making HISP yellow
custom_color_scales = {
    ("HISP", True): 0.02,
    ("ACDC", True): 0.02,
    ("HISP", False): 0.8,
}

# Default location to sample: 0.2
colors = {k: pc.sample_colorscale(v, custom_color_scales.get((k, args.hisp_yellow), 0.2))[0] for k, v in colorscales.items()}

symbol = {
    "ACDC": "circle",
    "SP": "x",
    "HISP": "diamond",
}

weights_type_symbols = {
    "trained": symbol,
    "reset":  {
        "ACDC": "circle-open-dot",
        "SP": "x-open-dot",
        "HISP": "diamond-open-dot",
    },
}

score_name = {
    "ACDC": "threshold",
    "SP": "lambda",
    "HISP": "score",
}


x_names = {
    "edge_fpr": "False positive rate (edges)",
    "node_fpr": "False positive rate (nodes)",
    "edge_tpr": "True positive rate (edges)",
    "node_tpr": "True positive rate (nodes)",
    "edge_precision": "Precision (edges)",
    "node_precision": "Precision (nodes)",
    "n_edges": "Number of edges",
    "n_nodes": "Number of nodes",
    "test_kl_div": "KL(model, ablated)",
    "test_loss": "Task-specific test metric",
}

def discard_non_pareto_optimal(points, auxiliary, cmp="gt"):
    ret = []
    for (x, y), aux in zip(points, auxiliary):
        for x1, y1 in points:
            if x1 < x and getattr(y1, f"__{cmp}__")(y) and (x1, y1) != (x, y):
                break
        else:
            ret.append(((x, y), aux))
    return list(sorted(ret))

#%%

def make_fig(metric_idx=0, x_key="edge_fpr", y_key="edge_tpr", weights_types=("trained",), ablation_type="random_ablation", plot_type="roc_nodes", scale_max=1.0):
    scale_min = 0.01 if args.hisp_yellow else 0.2
    TOP_MARGIN = (
        0.09
        + 0.26 * len(weights_types)
        + (0.10 if plot_type.startswith("metric_edges") or plot_type.startswith("kl_edges") else 0.0)
    )
    LEFT_MARGIN = -0.02
    RIGHT_MARGIN = 0.02 if y_key in ["edge_tpr", "node_tpr"] else 0.00
    if plot_type in ["roc_nodes", "roc_edges", "precision_recall"]:
        rows_cols_task_idx = [
            ((1, 1), "ioi"),
            ((1, 3), "tracr-reverse"),
            ((1, 4), "tracr-proportion"),
            ((2, 3), "docstring"),
            ((2, 4), "greaterthan"),
        ]
        specs=[[{"rowspan": 2, "colspan": 2}, None, {}, {}, {"rowspan": 2, "colspan": 1, "t": TOP_MARGIN, "l": LEFT_MARGIN, "r": RIGHT_MARGIN}], [None, None, {}, {}, None]]
        column_widths = [0.24, 0.24, 0.24, 0.24, 0.04]
        subplot_titles = ("ioi", "tracr-reverse", "tracr-proportion", THRESHOLD_ANNOTATION, "docstring", "greaterthan")
        subplot_titles = [TASK_NAMES.get(task_idx, task_idx) for task_idx in subplot_titles]

    elif plot_type in ["kl_edges_4", "metric_edges_4"]:
        rows_cols_task_idx = [
            ((1, 1), "ioi"),
            ((1, 2), "greaterthan"),
            ((2, 1), "induction"),
            ((2, 2), "docstring"),
            ((1, 3), "ioi"),
            ((1, 4), "greaterthan"),
            ((2, 3), "induction"),
            ((2, 4), "docstring"),
        ]
        specs = [
            [{}, {}, {}, {}, {"rowspan": 2, "colspan": 1, "t": TOP_MARGIN, "l": LEFT_MARGIN, "r": RIGHT_MARGIN}],
            [{}, {}, {}, {}, None],
        ]
        column_widths = [0.24, 0.24, 0.24, 0.24, 0.04]
        subplot_titles = ("ioi", "greaterthan", "ioi", "greaterthan", THRESHOLD_ANNOTATION, "induction", "docstring", "induction", "docstring")
        subplot_titles = [TASK_NAMES.get(task_idx, task_idx) for task_idx in subplot_titles]
        for i in [0, 1, 5, 6]:
            subplot_titles[i] += " (corrupted)"
        for i in [2, 3, 7, 8]:
            subplot_titles[i] += " (zero)"
    elif plot_type in ["kl_edges_induction", "metric_edges_induction"]:
        rows_cols_task_idx = [
            ((1, 1), "induction"),
            ((1, 2), "induction"),
        ]
        specs = [[{}, {}, {"t": 0.0, "l": -0.04, "r": RIGHT_MARGIN + 0.16}]]
        column_widths = [0.40, 0.40, 0.20]
        subplot_titles = (TASK_NAMES["induction"] + " (corrupted)", TASK_NAMES["induction"] + " (zero)", THRESHOLD_ANNOTATION)
    else:
        rows_cols_task_idx = [
            ((1, 1), "ioi"),
            ((1, 2), "tracr-reverse"),
            ((1, 3), "tracr-proportion"),
            ((2, 1), "induction"),
            ((2, 2), "docstring"),
            ((2, 3), "greaterthan"),
        ]
        # t: top padding
        specs = [[{}, {}, {}, {"rowspan": 2, "colspan": 1, "t": TOP_MARGIN, "l": LEFT_MARGIN, "r": RIGHT_MARGIN}], [{}, {}, {}, None]]
        column_widths = [0.32, 0.32, 0.32, 0.04]
        subplot_titles = ("ioi", "tracr-reverse", "tracr-proportion", THRESHOLD_ANNOTATION, "induction", "docstring", "greaterthan")
        subplot_titles = [TASK_NAMES.get(task_idx, task_idx) for task_idx in subplot_titles]

    rows_and_cols, task_idxs = list(zip(*rows_cols_task_idx))

    fig = make_subplots(
        rows=len(specs),
        cols=len(specs[0]),
        # specs parameter is really cool, this argument needs to have same dimenions as the rows and cols
        specs=specs,
        column_widths=column_widths,
        print_grid=False,
        # subplot_titles=("First Subplot", "Second Subplot", "Third Subplot", "Fourth Subplot", "Fifth Subplot"),
        subplot_titles=subplot_titles,
        x_title=x_names[x_key],
        y_title=x_names[y_key],
        # title_font=dict(size=8),
    )

    fig.update_annotations(font_size=12)

    all_series = []
    bounds_for_alg = {}
    for alg_idx, methodof in alg_names.items():
        min_log_score = 1e90
        max_log_score = -1e90

        for weights_type in weights_types:
            for (row, col), task_idx in rows_cols_task_idx:
                metric_name = METRICS_FOR_TASK[task_idx][metric_idx]
                if plot_type in ["metric_edges_4", "kl_edges_4"]:
                    if col >= 3:
                        ablation_type = "zero_ablation"
                    else:
                        ablation_type = "random_ablation"
                elif plot_type in ["metric_edges_induction", "kl_edges_induction"]:
                    if col == 2:
                        ablation_type = "zero_ablation"
                    else:
                        ablation_type = "random_ablation"
                try:
                    scores = np.array(all_data[weights_type][ablation_type][task_idx][metric_name][alg_idx]["score"])
                except KeyError as e:
                    print(
                        f"Couldn't find {weights_type} {ablation_type} {task_idx} {metric_name} {alg_idx} {x_key} {y_key}"
                    )
                    raise e

                if methodof == "HISP":
                    scores = np.array(all_data[weights_type][ablation_type][task_idx][metric_name][alg_idx]["n_nodes"])
                    nanmax_scores = np.max(np.nan_to_num(scores, nan=-np.inf, neginf=-np.inf, posinf=-np.inf))
                    scores *= 100 / nanmax_scores

                    log_scores = scores  # Use linear scale for colors
                else:
                    if methodof == "ACDC":
                        # Filter scores that are too small
                        scores = scores[scores >= args.min_score]

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        log_scores = np.log10(scores)
                log_scores = np.nan_to_num(log_scores, nan=0.0, neginf=0.0, posinf=0.0)

                min_log_score = min(np.min(log_scores), min_log_score)
                max_log_score = max(np.max(log_scores), max_log_score)
        bounds_for_alg[methodof] = (min_log_score, max_log_score)


    all_algs_min = min(np.log10(np.clip(v, a_min=1, a_max=None)) if k == "HISP" else v for k, (v, _) in bounds_for_alg.items())
    all_algs_max = max(np.log10(v) if k == "HISP" else v for k, (_, v) in bounds_for_alg.items())

    def normalize(x, x_min, x_max):
        if (x_max - x_min) < 1e-8:
            out = np.ones_like(x)
        else:
            out = (x - x_min) / (x_max - x_min) * (scale_max - scale_min)  + scale_min
        return out

    HEATMAP_ALGS = ["ACDC"] # , "SP", "HISP"] # TODO 
    for i, methodof in enumerate(HEATMAP_ALGS):
        alg_min, alg_max = bounds_for_alg[methodof]
        # nums = normalize(heatmap_ys, alg_min, alg_max)
        # nums[nums < scale_min] = np.nan
        # nums[nums > 1] = np.nan
        alg_ys = np.linspace(alg_min, alg_max, 100)
        if methodof == "HISP":
            alg_ys = np.log10(alg_ys)
        nums = np.linspace(scale_min, scale_max, len(alg_ys))
        fig.add_trace(
            go.Heatmap(
                x=[i, i+0.95],
                y=alg_ys,
                z=nums[:, None],
                colorscale=colorscales[methodof],
                showscale=False,
                zmin=0.0,
                zmax=1.0,
            ),
            row=1,
            col=len(specs[0]),
        )
    fig.update_xaxes(showline=False, zeroline=False, showgrid=False, row=1, col=len(specs[0]), showticklabels=False, ticks="")
    tickvals = list(range(int(np.floor(all_algs_min)), int(np.ceil(all_algs_max))))
    ticktext = [f"$10^{{{v}}}$" for v in tickvals]
    fig.update_yaxes(showline=False, zeroline=False, showgrid=True, row=1, col=len(specs[0]), side="right",
                     range=[all_algs_min, all_algs_max], tickvals=tickvals, ticktext=ticktext)

    if not args.hisp_yellow:
        fig.update_xaxes(gridcolor=GRIDCOLOR, zerolinecolor=ZEROLINECOLOR, zerolinewidth=1)
        fig.update_yaxes(gridcolor=GRIDCOLOR, zerolinecolor=ZEROLINECOLOR, zerolinewidth=1)


    for alg_idx, methodof in alg_names.items():
        min_log_score, max_log_score = bounds_for_alg[methodof]
        for weights_type in weights_types:
            for (row, col), task_idx in rows_cols_task_idx:
                metric_name = METRICS_FOR_TASK[task_idx][metric_idx]
                if plot_type in ["metric_edges_4", "kl_edges_4"]:
                    if col >= 3:
                        ablation_type = "zero_ablation"
                    else:
                        ablation_type = "random_ablation"
                elif plot_type in ["metric_edges_induction", "kl_edges_induction"]:
                    if col == 2:
                        ablation_type = "zero_ablation"
                    else:
                        ablation_type = "random_ablation"

                if plot_type.startswith("metric_edges"):
                    y_key = "test_" + METRICS_FOR_TASK[task_idx][1] # gets overwritten to "test NLL" 

                this_data = all_data[weights_type][ablation_type]
                x_data = np.array(this_data[task_idx][metric_name][alg_idx][x_key])
                y_data = np.array(this_data[task_idx][metric_name][alg_idx][y_key])
                scores = np.array(this_data[task_idx][metric_name][alg_idx]["score"])

                if methodof == "HISP":
                    scores = np.array(this_data[task_idx][metric_name][alg_idx]["n_nodes"])
                    nanmax_scores = np.max(np.nan_to_num(scores, nan=-np.inf, neginf=-np.inf, posinf=-np.inf))
                    scores *= 100 / nanmax_scores

                    log_scores = scores  # use linear scale for colors
                else:
                    if methodof == "ACDC":
                        # Filter scores that are too small
                        mask = (scores >= args.min_score) | (~np.isfinite(scores))
                        x_data = x_data[mask]
                        y_data = y_data[mask]
                        scores = scores[mask]
                        del mask

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        log_scores = np.log10(scores)

                # if alg_idx == "16H" and task_idx == "tracr-reverse":
                #     import pdb; pdb.set_trace()
                log_scores = np.nan_to_num(log_scores, nan=np.nan, neginf=-1e90, posinf=1e90)
                normalized_log_scores = normalize(log_scores, min_log_score, max_log_score)

                if alg_idx == "SP":
                    # Divide by number of loss runs. Fix earlier bug.
                    if x_key.startswith("test_"):
                        x_data = [x / 20 for x in x_data]

                    if y_key.startswith("test_"):
                        y_data = [y / 20 for y in y_data]


                points = list(zip(x_data, y_data))
                if y_key not in ["node_tpr", "edge_tpr"]:
                    pareto_optimal = [] # list(sorted(points))  # Not actually pareto optimal but we want to plot all of them
                    pareto_log_scores = []
                    pareto_scores = []
                else:
                    pareto_optimal_aux = discard_non_pareto_optimal(points, zip(log_scores, scores))
                    pareto_optimal, aux = zip(*pareto_optimal_aux)
                    pareto_log_scores, pareto_scores = zip(*aux)

                auc = None
                if len(pareto_optimal):
                    x_data, y_data = zip(*pareto_optimal)
                    if plot_type in ["roc_nodes", "roc_edges"]:
                        try:
                            auc = pessimistic_auc(x_data, y_data)
                        except Exception as e:
                            print(task_idx, metric_name, alg_idx, x_key)
                            assert False
                            print(e)
                            auc=-420.0

                    fig.add_trace(
                        go.Scatter(
                            x=list(x_data),
                            y=list(y_data),
                            name=methodof,
                            mode="lines",
                            line=dict(shape="hv", color=colors[methodof]),
                            showlegend=False,
                            hovertext=[f"{score_name[methodof]}={t:e}" for t in pareto_scores],
                        ),
                        row=row,
                        col=col,
                    )

                test_kl_div = this_data[task_idx][metric_name][alg_idx]["test_kl_div"][1:-1]
                test_loss = this_data[task_idx][metric_name][alg_idx]["test_" + metric_name][1:-1]
                if alg_idx == "SP":
                    test_kl_div = [x / 20 for x in test_kl_div]
                    test_loss = [x / 20 for x in test_loss]

                if plot_type in ["roc_nodes", "roc_edges"]:
                    all_series.append(pd.Series({
                        "task": task_idx,
                        "method": methodof,
                        "metric": metric_name,
                        "weights_type": weights_type,
                        "ablation_type": ablation_type,
                        "plot_type": plot_type,
                        "auc": auc,
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
                        "metric": metric_name,
                        "weights_type": weights_type,
                        "ablation_type": ablation_type,
                        "plot_type": "induction_kl_edges",
                        "auc": None,
                        "n_points": len(points),
                        "test_kl_div": np.mean(test_kl_div),
                        "test_kl_div_max": np.max(test_kl_div),
                        "test_kl_div_min": np.min(test_kl_div),
                        "test_loss": np.mean(test_loss),
                        "test_loss_max": np.max(test_loss),
                        "test_loss_min": np.min(test_loss),
                    }))


                others = [(*p, *aux) for (p, *aux) in sorted(zip(points, log_scores, normalized_log_scores, scores), key=lambda x: -x[-1])] #  if p not in pareto_optimal]

                if others:
                    x_data, y_data, log_scores, normalized_log_scores, scores = zip(*others)
                    if not (np.isfinite(x_data[0]) and np.isfinite(y_data[0])):
                        x_data = x_data[1:]
                        y_data = y_data[1:]
                        log_scores = log_scores[1:]
                        normalized_log_scores = normalized_log_scores[1:]
                        scores = scores[1:]
                    if not (np.isfinite(x_data[-1]) and np.isfinite(y_data[-1])):
                        x_data = x_data[:-1]
                        y_data = y_data[:-1]
                        log_scores = log_scores[:-1]
                        normalized_log_scores = normalized_log_scores[:-1]
                        scores = scores[:-1]

                    assert not np.any(~np.isfinite(x_data))
                    assert not np.any(~np.isfinite(y_data))

                    color = normalized_log_scores
                else:
                    x_data, y_data = [None], [None]
                    log_scores = [np.nan]

                    color = colors[methodof]

                # print(task_idx, alg_idx, metric_name, len(x_data), len(y_data), plot_type)

                fig.add_trace(
                    go.Scatter(
                        x=x_data,
                        y=y_data,
                        name=methodof,
                        mode="markers",
                        showlegend = False,
                        marker=dict(
                            size=[3 if p in pareto_optimal else 7 for p in points],
                            line=dict(width=[0 if p in pareto_optimal else 0.7 for p in points], color="DarkSlateGrey"),
                            color=color,
                            symbol=weights_type_symbols[weights_type][methodof],
                            colorscale=colorscales[methodof],
                            cmin=0.0,
                            cmax=1.0,
                        ),
                        hovertext=[f"{score_name[methodof]}={t:e}" for t in scores],
                    ),
                    row=row,
                    col=col,
                )
                # for l in log_scores:
                #     if np.allclose(l, -2):
                #         import pdb; pdb.set_trace()


                # Just add the legend
                if (row, col) == (1, len(specs[0])-1):
                    fig.add_trace(
                        go.Scatter(
                            x=[None],
                            y=[None],
                            name=f"{methodof} (reset)" if weights_type == "reset" else methodof,
                            mode="markers",
                            showlegend=True,
                            marker=dict(
                                size=7,
                                color=colors[methodof],
                                symbol=weights_type_symbols[weights_type][methodof],
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
                    if plot_type in ["roc_nodes", "roc_edges"] and args.arrows:
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
                        # yref="y",
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

                    if y_key in ["edge_fpr", "edge_tpr", "node_fpr", "node_tpr", "precision"]:
                        fig.update_yaxes(visible=True, row=row, col=col, tickangle=-45, dtick=0.25, range=[-0.05, 1.05]) # ???
                    else:
                        fig.update_yaxes(visible=True, row=row, col=col, tickangle=-45)

                    if x_key == "n_edges":
                        if plot_type.endswith("_induction"):
                            fig.update_xaxes(type='log', row=row, col=col, ticktext=X_TICKVALS, tickvals=X_TICKVALS)
                        else:
                            fig.update_xaxes(type='log', row=row, col=col, tickangle=0)
                    else:
                        fig.update_xaxes(dtick=0.25, range=[-0.05, 1.05], row=row, col=col)

                    # # add label to x axis
                    # fig.update_xaxes(title_text="False positive rate", row=row, col=col)
                    # # add label to y axis
                    # fig.update_yaxes(title_text="True positive rate", row=row, col=col)

                    fig.update_layout(title_font=dict(size=1))

                else:
                    # If the subplot is not the large plot, hide its axes
                    if y_key in ["edge_fpr", "edge_tpr", "node_tpr", "node_fpr", "precision"]:
                        fig.update_yaxes(visible=True, row=row, col=col, tickangle=-45, dtick=0.25, tickvals=[0, 0.25, 0.5, 0.75, 1.], ticktext=["0", "", "0.5", "", "1"], range=[-0.05, 1.05]) # ???
                    else:
                        fig.update_yaxes(visible=True, row=row, col=col, tickangle=-45)

                    if x_key == "n_edges":
                        if plot_type.endswith("_induction"):
                            fig.update_xaxes(type='log', row=row, col=col, ticktext=X_TICKVALS, tickvals=X_TICKVALS)
                        else:
                            fig.update_xaxes(type='log', row=row, col=col, tickangle=0)
                    else:
                        fig.update_xaxes(visible=True, row=row, col=col, tickvals=[0, 0.25, 0.5, 0.75, 1.], ticktext=["0", "", "0.5", "", "1"], range=[-0.05, 1.05])

                    # smaller title font
                    fig.update_layout(title_font=dict(size=20))

                if not args.hisp_yellow:
                    fig.update_xaxes(gridcolor=GRIDCOLOR, zerolinecolor=ZEROLINECOLOR, zerolinewidth=1, row=row, col=col)
                    fig.update_yaxes(gridcolor=GRIDCOLOR, zerolinecolor=ZEROLINECOLOR, zerolinewidth=1, row=row, col=col)
                    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")


    # Add horizontal lines with test performance on KL plots
    if plot_type.startswith("metric_edges") or plot_type.startswith("kl_edges"):
        for (row, col), task_idx in rows_cols_task_idx:
            metric_name = METRICS_FOR_TASK[task_idx][1]
            if plot_type.startswith("metric_edges"):
                y_key = "test_" + METRICS_FOR_TASK[task_idx][1]
            else:
                y_key = "test_" + METRICS_FOR_TASK[task_idx][0]

            for weights_type, name, value, line_dash, line_color in [ # three black lines
                ("trained", "Clean", 1.0, "solid", "rgb(0, 0, 0)"),
                ("trained", "Canonical", 0.5, "dashdot", "rgb(0, 0, 0)"),
                # ("random", "Random", 0.5, "dashed", "rgb(0, 0, 0)"),
                ("reset", "Reset", 1.0, "dot", "rgb(0, 0, 0)"),
            ]:
                this_data = all_data[weights_type][ablation_type][task_idx][metric_name]["CANONICAL"]

                scores = np.array(this_data["score"])
                baseline_y = np.array(this_data[y_key])
                mask = scores == value
                if mask.sum() == 0:
                    continue
                assert mask.sum() == 1

                y = baseline_y[mask][0]

                line_style = dict(
                    dash=line_dash,
                    width=1.5,
                    color=line_color,
                )

                fig.add_hline(
                    y=y,
                    line=line_style,
                    row=row,
                    col=col,
                )
                if (row, col) == (1, len(specs[0])-1):
                    fig.add_trace(
                        go.Scatter(
                            x=[None],
                            y=[None],
                            mode="lines",
                            name=name,
                            line=line_style,
                        ),
                        row=row,
                        col=col,
                    )


    # move legend to left
    fig.update_layout(
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=0.92,
            font=dict(size=8),
            bgcolor="rgba(0,0,0,0)",  # Set the background color to transparent
        ),
        title_font=dict(size=4),
    )

    scale = 1.2

    if plot_type in ["kl_edges", "metric_edges"]:
        height = 300
        width = 500
    elif plot_type in ["kl_edges_induction", "metric_edges_induction"]:
        height = 190
        width = 500
    elif plot_type in ["kl_edges_4", "metric_edges_4"]:
        height = 290
        width = 550
    else:
        height = 250
        width = 500

    # No title,
    fig.update_layout(height=height*scale, width=width*scale*scale, margin=dict(l=55, r=70, t=20, b=50))
    # MEGA HACK: add space between tau and colorbar
    for i in range(len(fig.layout.annotations)):
        anno = fig.layout.annotations[i]
        if str(THRESHOLD_ANNOTATION) in str(anno["text"]):
            anno["y"] += 0.02
    ret = (fig, pd.concat(all_series, axis=1) if all_series else pd.DataFrame())
    return ret

plot_type_keys = {
    "precision_recall": ("edge_tpr", "edge_precision"),
    "roc_nodes": ("node_fpr", "node_tpr"),
    "roc_edges": ("edge_fpr", "edge_tpr"),
    "kl_edges": ("n_edges", "test_kl_div"),
    "metric_edges": ("n_edges", "test_loss"),
    "kl_edges_4": ("n_edges", "test_kl_div"),
    "metric_edges_4": ("n_edges", "test_loss"),
    "kl_edges_induction": ("n_edges", "test_kl_div"),
    "metric_edges_induction": ("n_edges", "test_loss"),
}

#%%

PLOT_DIR = DATA_DIR.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)
first = True

all_dfs = []
for metric_idx in [1, 0]:
    for ablation_type in ["random_ablation", "zero_ablation"]:
        for weights_type in ["trained", "reset"]:  # Didn't scramble the weights enough it seems
            for plot_type in ["roc_edges", "kl_edges_induction", "metric_edges_4", "kl_edges_4", "kl_edges", "precision_recall", "metric_edges_induction", "roc_nodes", "metric_edges"]:
                x_key, y_key = plot_type_keys[plot_type]
                fig, df = make_fig(metric_idx=metric_idx, weights_types=["trained"] if weights_type == "trained" else ["trained", weights_type], ablation_type=ablation_type, x_key=x_key, y_key=y_key, plot_type=plot_type)
                if len(df):
                    all_dfs.append(df.T)
                    print(all_dfs[-1])
                metric = "kl" if metric_idx == 0 else "other"
                fname = "--".join([metric, weights_type, ablation_type, plot_type])

                fig.write_image(PLOT_DIR / (fname + ".pdf"))

                if WRITE_JSON:
                    fig.write_json(PLOT_DIR / (fname + ".json"))

                if first:
                    fig.show()
                    first = False
                    assert False

pd.concat(all_dfs).to_csv(PLOT_DIR / "data.csv")

# %%

# Stefan
#   1 hour ago
# Very nice plots! Small changes
# 1st title should be IOI, "Cicuit Recovery" should be above or somewhere else
# [Minor] Unify xlim=ylim=[-0.01, 1.01] or so
# :raised_hands:
# 1
# if len(df
# x_key, y_key = plot_type_keys["kl_edges"]
# fig, _ = make_fig(metric_idx=0, weights_type="reset", ablation_type="zero_ablation", plot_type="kl_edges")
# fig.show()
