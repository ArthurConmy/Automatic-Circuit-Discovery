#%%

"""
File for doing misc plotting
"""

import IPython
if IPython.get_ipython() is not None:
    IPython.get_ipython().run_line_magic('load_ext', 'autoreload')
    IPython.get_ipython().run_line_magic('autoreload', '2')

import warnings
import torch 
import os
from tqdm import tqdm
import pandas as pd
import wandb
import numpy as np
from acdc.acdc_utils import ct
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from acdc.acdc_utils import get_nonan, get_corresponding_element, get_first_element, get_longest_float, process_nan, get_threshold_zero

#%%

project_names = [
    # "tl_induction_proper",
    # "acdc",
    # "induction_arthur",
    # "shuffle_the_dataset",
    # "arthurinduction_randomabl_reverse",
    # "arthurinduction_zeroabl_reverse",
    "new_zero_ablation=1_adria",
]

api = wandb.Api()
ALL_COLORS = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]

final_edges = []
final_metric = []
names = []
COLS = ["black", "red", "green", "blue", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
col_dict = {
    "geometric": "blue",
    "harmonic": "red",
    "off": "black",
}
colors = []
_initial_losses = [] # [1.638 for _ in range(len(names))]
_initial_edges = [] # [11663 for _ in range(len(names))]
histories = []
min_metrics = []

def filter(name):
    return True
    # return "kl_divergence" in name
    # return name.endswith("reversed") and "zero"
    # return name.endswith("_reversed") and not name.endswith("zero_reversed")
    # return name.endswith("zero_reversed") or name.endswith("zero")

for pi, project_name in (enumerate(project_names)):
    print("Finding runs...")
    runs = list(api.runs(f"remix_school-of-rock/{project_name}"))
    print("Found runs!")

    for i, run in enumerate(tqdm(runs)):
        print(run.name, "state:", run.state)
        if not filter(run.name):
            continue
        if run.state == "finished": #  or run.state == "failed":
            history = pd.DataFrame(run.scan_history())
            histories.append(history)

            # rename the column "experiment.cur_metric" to "metric"
            history = history.rename(columns={"experiment.cur_metric": "metric", 'num_edges_total': 'num_edges', "self.cur_metric": "metric"})
            min_edges = history["num_edges"].min().min()
            max_edges = history["num_edges"].max().max()
            
            if min_edges <= 0:
                print("ewwwwww this was no edges")
                histories.pop()
                continue
            
            assert 1e30 > max_edges, max_edges

            start_metric = get_first_element(history, "metric")
            end_metric = get_first_element(history, "metric", last=True)
            min_metric = history["metric"].min() # sigh

            all_metrics = history["metric"].values
            all_edges = history["num_edges"].values
            if len(all_metrics.shape) == 2:
                all_metrics = all_metrics[:, 0]
            if len(all_edges.shape) == 2:
                all_edges = all_edges[:, 0]
            all_metrics = process_nan(all_metrics, reverse=True)
            all_edges = process_nan(all_edges)

            found_thing = False

            for i in range(1, 61):
                if np.isnan(all_metrics[-i]) or len(all_edges) < i:
                    continue
                if found_thing:
                    break
                found_thing = True
                names.append(run.name)
                final_edges.append(all_edges[-i])
                _initial_edges.append(max_edges)
                _initial_losses.append(start_metric)
                final_metric.append(all_metrics[-i])
                colors.append("black")

            else:
                assert False # ???

            print(len(colors))

# save list of dataframes
with open("histories/" + ct() + ".pkl", "wb") as f:
    import pickle
    pickle.dump(histories, f)

if torch.norm(torch.tensor(_initial_losses) - _initial_losses[0]) > 1e-5:
    warnings.warn(f"Initial losses are not the same, so this may be an unfair comparison of {_initial_losses=}")
if torch.norm(torch.tensor(_initial_edges).float() - _initial_edges[0]) > 1e-5:
    warnings.warn(f"Initial edges are not the same, so this may be an unfair comparison of {_initial_edges=}")

added_final_edges = False

thresholds = []
for name in names:
    if name.endswith("zero") or (name.endswith("reversed") and not name.endswith("zero_reversed")):
        thresholds.append(get_threshold_zero(name, -2))
    elif "zero_ablation=" in name:
        thresholds.append(float(name.split("=")[1].split("_")[0]))
    else:
        thresholds.append(get_threshold_zero(name, -1)) # or -3...

#%%

ACTUALLY_DO_BASELINE = False # for now, we're fucked by how somehow the cross entropy decreases??? Look into this

if not added_final_edges and ACTUALLY_DO_BASELINE:
    final_edges.append(_initial_edges[0])
    final_metric.append(_initial_losses[0]) # from just including the whole graph
    names.append("The whole graph")
    colors.append("black")
    added_final_edges = True

scrubbed_induction_head_value = 4.493981649709302
REVERSE_SIGN = False # now we're doing KL divergence (/cross entropy) we should probably be minimizing a metric

if REVERSE_SIGN:
    # now we're going to change things up so we only maximize metrics
    final_metric = [-x for x in final_metric]
    _initial_losses = [-x for x in _initial_losses]
    scrubbed_induction_head_value = -1.0 * scrubbed_induction_head_value

fig = go.Figure()

LOG_X_AXIS = False
if LOG_X_AXIS:
    final_edges = np.log(final_edges)

# add x axis label
fig.update_layout(
    xaxis_title="Number of edges" if not LOG_X_AXIS else "Log number of edges",
    yaxis_title="KL divergence",
)

# add title
fig.update_layout(
    title="ACDC compared to SP (zero ablation)",
)

#%%

if False:
    def get_edge_sp_things():
        api = wandb.Api()

        project_names = ["subnetwork_probing_edges"]
        histories = []

        lambdas=[]
        edges=[]
        kls=[]

        for pi, project_name in (enumerate(project_names)):
            print("Finding runs...")
            runs = list(api.runs(f"remix_school-of-rock/{project_name}"))
            print("Found runs!")

            for i, run in enumerate(tqdm(runs)):
                print(run.name, "state:", run.state)
                if run.state == "finished" or run.state == "crashed":
                    history = pd.DataFrame(run.scan_history())
                    histories.append(history)
                    try:
                        cur_lambda =(float(run.name.split("_")[2]))
                    except:
                        print(run.name, "didn't have lambda shit")
                        continue

                    # if "number_of_edges" in history.columns:

                    for i in range(-1, -10, -1):
                        lambdas.append(cur_lambda)
                        edges.append(history.iloc[i]["number_of_edges"])
                        kls.append(history.iloc[i]["acc_loss"])

                    warnings.warn("We take min of both, plausibly the (edges, kl) result is actually unattainable")

        labels = fig2.data[0]["regularization_params"]
        kls = torch.tensor(kls)
        return log_edges, kls, lambdas

    # skip this if don't want to include other things

    log_edges, kls, lambdas = get_edge_sp_things()

    final_edges = torch.cat([torch.tensor(final_edges), log_edges])
    final_metric = torch.cat([torch.tensor(final_metric), kls])
    names = names + [f"lambda={l}" for l in lambdas]
    thresholds = thresholds + [-1.0 for _ in lambdas]

#%%

if False:
    # clear out the scraped lambdas
    final_edges = final_edges[:-len(lambdas)]
    final_metric = final_metric[:-len(lambdas)]
    names = names[:-len(lambdas)]
    thresholds = thresholds[:-len(lambdas)]

#%%

assert len(final_edges) == len(final_metric) == len(names) == len(thresholds), f"{len(final_edges)=}, {len(final_metric)=}, {len(names)=}, {len(thresholds)=}"

# scatter plot with names as labels and thresholds as colors
fig.add_trace(
    go.Scatter(
        x=final_edges,
        y=final_metric,
        mode="markers",
        marker=dict(
            size=10,
            color=thresholds,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(
                title="Threshold",
                titleside="right",
                tickmode="array",
                tickvals=np.arange(0, 13)/1,
                ticktext=np.arange(0, 13)/4,
            ),
        ),
        text=names,
        name="ACDC"
    )
)

#%%

import plotly
# fname = "media_zero_json.json"
# fname = "better_zero.json"
fname="media/corrupted.json"
fig2 = plotly.io.read_json(fname)
fig2.show()

if False:
    # get points
    x = fig2.data[0]["x"]
    y = fig2.data[0]["y"]
    labels = fig2.data[0]["customdata"]
    labels = [x for x, _ in labels]

#%%

import plotly
fname = "better_random.json"
fig3 = plotly.io.read_json(fname)

# get points
myx = fig3.data[0]["x"]
myy = fig3.data[0]["y"]
old_labels = fig3.data[0]["text"]

#%%

indices = [i for i in range(len(old_labels)) if myx[i] >= 40] #  in old_labels]
myx = [myx[i] for i in indices]
myy = [myy[i] for i in indices]

final_edges.extend(myx)
final_metric.extend(myy) # [myy[i] for i in indices])
thresholds.extend([float(old_labels[i].split("_")[-1]) for i in indices])
names.extend([old_labels[i] for i in indices])

#%%

fig.add_trace(
    go.Scatter(
        x=torch.exp(torch.tensor(list(x))),
        y=y,
        mode="markers",
        marker=dict(
            size=10,
            color="black",
            symbol="x",
        ),
        text=labels,
        name="SP"
    )
)

#%%

# add legend

fig.update_layout(
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99,
    )
)

#%%
run = api.run("acdcremix/pareto-subnetwork-probing/runs/2zzctq6x")
path = "media/plotly/number_of_nodes_0_fa4f23a7d736607e9e9a.plotly.json"

# get the data
import plotly
json_data_as_string = run.file(path).download(replace=True).read()
subnetwork_prob_fig = plotly.io.read_json(path)
subnetwork_prob_fig.data[0]["y"] = [i * -1 for i in subnetwork_prob_fig.data[0]["y"]]
subnetwork_prob_fig.data[0].marker["symbol"] = "x"
subnetwork_prob_fig.data[0].marker["size"] = 10
subnetwork_prob_fig.data[0].marker["color"] = "black"
subnetwork_prob_fig.data[0]["name"] = "Subnetwork Probing"
subnetwork_prob_fig.data[0]["showlegend"] = True
subnetwork_prob_fig.update_layout(
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))
subnetwork_prob_fig.show()
# %%
from plotly.subplots import make_subplots

combined_fig = make_subplots(specs=[[{"secondary_y": True}]])
combined_fig.add_trace(fig.data[0], secondary_y=False)
combined_fig.add_trace(subnetwork_prob_fig.data[0], secondary_y=False)
# %%

for y_val, text in zip([_initial_losses[0], -1 * 4.493981649709302], ["The whole graph", "Induction heads scrubbed"]):
    # add a dotted line y=WHOLE_LOSS
    combined_fig.add_shape(
        type="line",
        x0=0,
        x1=_initial_edges[0],
        y0=y_val,
        y1=y_val,
        line=dict(
            color="Black",
            width=1,
            dash="dot",
        )
    )
    # add label to this
    combined_fig.add_annotation(
        x=_initial_edges[0] * 0.5,
        y=y_val,
        text=text,
        showarrow=False,
        xanchor="left",
        yanchor="top",
        xshift=10,
        yshift=2,
        font=dict(
            family="Courier New, monospace",
            size=16,
            color="Black"
        )
    )

# add legend for colors
combined_fig.update_layout(
    legend=dict(
        yanchor="top",
        y=0.3,
        xanchor="left",
        x=0.5
    )
)

# rescale
# fig.update_xaxes(range=[0, 1500])
# fig.update_yaxes(range=[0, max(final_metric)+0.01])

# add axis labels
combined_fig.update_xaxes(title_text="Number of edges")
combined_fig.update_yaxes(title_text="Log probs ")

# add title
combined_fig.update_layout(title_text="Induction results")

# %% [markdown]
# Unrelated to the above, this is for seeing how big direct effects are on the output

import plotly.graph_objects as go
ies = exp._nodes.i_names["final.inp"].incoming_effect_sizes                
fig = go.Figure()
labels = [i.name for i in ies]
vals = [v for v in ies.values()]
fig.add_trace(go.Bar(
    x=vals,
    y=labels,
    orientation='h',
    marker=dict(
        color='rgba(50, 171, 96, 0.6)',
        line=dict(
            color='rgba(50, 171, 96, 1.0)',
            width=3)
    )
))
fig.show()

