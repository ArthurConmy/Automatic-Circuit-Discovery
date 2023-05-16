# %%

from IPython import get_ipython
if get_ipython() is not None:
    get_ipython().magic('load_ext autoreload')
    get_ipython().magic('autoreload 2')

import plotly
import os
import json
import wandb
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from pathlib import Path

JSON_FNAME = Path(__file__).parent.parent / "acdc" / "media" / "roc_data.json"

# %%

tasks = ["Circuit Recovery (IOI)", "Tracr (Reverse)", "Tracr (Proportion)", "Docstring", "Greater Than"]

fig = make_subplots(
    rows=2,
    cols=4,
    # specs parameter is really cool, this argument needs to have same dimenions as the rows and cols
    specs=[[{"rowspan": 2, "colspan": 2}, None, {}, {}], [None, None, {}, {}]],
    print_grid=True,
    # subplot_titles=("First Subplot", "Second Subplot", "Third Subplot", "Fourth Subplot", "Fifth Subplot"),
    subplot_titles=tuple(tasks),
    x_title="False positive rate (edges)",
    y_title="True positive rate (edges)",
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

methods = ["ACDC", "SP", "HISP"]
colors = {
    "ACDC": "purple",  
    "SP": "green",
    "HISP": "yellow",
}

all_data = None
with open(JSON_FNAME, "r") as f:
    all_data = json.load(f)
    
for task_idx, (row, col) in enumerate(rows_and_cols):
    for idx, methodof in enumerate(methods):
        x_data = all_data[tasks[task_idx]][methodof]["x"]
        y_data = all_data[tasks[task_idx]][methodof]["y"]
        print(x_data, y_data)

        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=y_data,
                name=methodof,
                mode="lines",
                line=dict(shape="hv", color=colors[methodof]),
                showlegend = (row, col) == rows_and_cols[-1],
            ),
            row=row, 
            col=col,
        )

        fig.update_layout(
            title_font=dict(size=8),
        )

        if (row, col) == rows_and_cols[0]:
            fig.add_annotation(
                xref="x domain",
                yref="y",
                x=0.35, # end of arrow
                y=0.75,
                text="",
                axref="x domain",
                ayref="y",
                ax=0.55,
                ay=0.55,
                arrowhead=2,
                row = row,
                col = col,
            )
            fig.add_annotation(
                xref="x domain",
                yref="y",
                x=0.6, # end of arrow
                y=0.8,
                text="",
                axref="x domain",
                ayref="y",
                ax=0.6,
                ay=0.6,
                arrowhead=2,
                row = row,
                col = col,
            )
            fig.add_annotation(
                xref="x domain",
                yref="y",
                x=0.8, # end of arrow
                y=0.5,
                text="",
                axref="x domain",
                ayref="y",
                ax=0.6,
                ay=0.5,
                arrowhead=2,
                row = row,
                col = col,
            )
            fig.add_annotation(text="More circuit components recovered",
                  xref="x", yref="y",
                  x=0.45, y=0.85, showarrow=False, font=dict(size=8), row=row, col=col)
            fig.add_annotation(text="Better",
                  xref="x", yref="y",
                  x=0.4, y=0.6, showarrow=False, font=dict(size=12), row=row, col=col)
            fig.add_annotation(text="More wrong components recovered",
                  xref="x", yref="y",
                  x=0.65, y=0.45, showarrow=False, font=dict(size=8), row=row, col=col) # TODO could add two text boxes

            fig.update_yaxes(visible=True, row=row, col=col, tickangle=-45) # ???

            # # add label to x axis
            # fig.update_xaxes(title_text="False positive rate", row=row, col=col)
            # # add label to y axis
            # fig.update_yaxes(title_text="True positive rate", row=row, col=col)

            fig.update_layout(title_font=dict(size=1)) # , row=row, col=col)


        else:
            # If the subplot is not the large plot, hide its axes
            fig.update_xaxes(visible=True, row=row, col=col, dtick=1)
            fig.update_yaxes(visible=True, row=row, col=col, tickangle=-45, dtick=1) # ???

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
        x=1.2, 
        font=dict(size=8),
    ),
    title_font=dict(size=4),
)

scale = 1.2

fig.update_layout(height=350*scale, width=scale*scale*500, title=dict(text="Lol"))  # plausibly make slightly bigger
fig.show()

# %%

# Stefan
#   1 hour ago
# Very nice plots! Small changes
# 1st title should be IOI, "Cicuit Recovery" should be above or somewhere else
# [Minor] Unify xlim=ylim=[-0.01, 1.01] or so
# :raised_hands:
# 1

