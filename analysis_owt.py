#%%
import json

def load_dict(fname):
    with open(fname, 'r') as f:
        return json.load(f)

from typing import List
def split_by_any_of(s, list_of_strs: List[str]):
    for sep in list_of_strs:
        s = s.replace(sep, f'YOLO')
    return s.split("YOLO")
#%%

d=load_dict("data2.json")

names = []
answers = []

import torch
results = torch.zeros(12, 13)

for idx, (k, v) in enumerate(list(d.items())[156:]):
    print(k, v)
    key = k.split("Z")

    if key[0] != list(d.keys())[156].split("Z")[0]:
        break

    names.append(key[2] + " " + key[3])
    answers.append(v)
    results[int(key[2].split(".")[1])][int(key[3] if key[3] != "None" else 12)] = v

import plotly.express as px
fig = px.bar(x=names, y=answers)
fig.show()

#%%


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
        return 