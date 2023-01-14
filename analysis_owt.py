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

#%%

# make the JSON file a 4D tensor
# receiver_layer(-1=EMBED, 12=OUT) receiver_head (12=MLP) sender_layer sender_head

def dec(s):
    s=s.split("Z")
    receiver_layer = None
    if s[0] == "blocks.11.hook_resid_post":
        receiver_layer = 12
    else:
        receiver_layer = int(s[0].split(".")[1])
    assert receiver_layer is not None

    receiver_head = None
    if s[1] == "None":
        receiver_head = 12
    else:
        receiver_head = int(s[1])

    sender_layer = int(s[2].split(".")[1])
    sender_head = 12 if s[3]=="None" else int(s[3])

    return receiver_layer, receiver_head, sender_layer, sender_head

    # blocks.11.hook_resid_postZNoneZblocks.10.attn.hook_resultZ9

#%%

def append_to_json(filename, key, value):
    """filename: name of the json file
    key: key of the dictionary
    value: value of the dictionary"""
    with open(filename, 'r+') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data[key] = value
        file.seek(0)
        # Sets file's current position at offset.
        json.dump(file_data, file, indent = 4)

#%%

m = torch.tensor(torch.zeros(13, 13, 13, 13))
d = load_dict("data2.json")
for k, v in d.items():
    receiver_layer, receiver_head, sender_layer, sender_head = dec(k)
    m[receiver_layer][receiver_head][sender_layer][sender_head] = v

#%%

base_loss_was = 2.850428581237793
show_pp(
    m[12][12][:12] - base_loss_was,
)