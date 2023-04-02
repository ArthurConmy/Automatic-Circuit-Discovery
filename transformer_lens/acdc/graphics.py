import wandb
import os
from collections import defaultdict
import pickle
import torch
import huggingface_hub
import datetime
from typing import Dict
import wandb
import plotly.graph_objects as go
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import warnings
import networkx as nx
import graphviz

# # I hope that it's reasona
# from transformer_lens.acdc.utils import (
#     make_nd_dict,
#     TorchIndex,
#     Edge, 
#     EdgeType,
# )  # these introduce several important classes !!!

# -------------------------------------------
# SOME GRAPHICS
# 
# VERY CURSED, AND COULD BE IMPROVED A LOT
# -------------------------------------------

def generate_random_color(colorscheme: str) -> str:
    """
    https://stackoverflow.com/questions/28999287/generate-random-colors-rgb
    """
    import cmapy

    def rgb2hex(rgb):
        """
        https://stackoverflow.com/questions/3380726/converting-an-rgb-color-tuple-to-a-hexidecimal-string
        """
        return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

    return rgb2hex(cmapy.color("Pastel2", random.randrange(0, 256), rgb_order=True))


def build_colorscheme(node_names, colorscheme: str = "Pastel2") -> Dict[str, str]:
    colors = {}
    for node in node_names:
        colors[str(node)] = generate_random_color(colorscheme)
    return colors

# -------------------------------------------
# GRAPHVIZ
# -------------------------------------------

# def show(
#     TLACDCCorrespondence,
#     fname=None,
#     colorscheme: str = "Pastel2",
#     minimum_penwidth: float = 0.3,
#     show: bool = True,
# ):
#     """
#     takes matplotlib colormaps
#     """
#     g = graphviz.Digraph(format="png")

#     colors = self.build_colorscheme(colorscheme)

#     # create all nodes
#     for index, child in enumerate(self.corr):
#         parent: ACDCInterpNode
#         child = typing.cast(ACDCInterpNode, child)
#         if len(child.parents) > 0 or index == 0:
#             g.node(
#                 show_node_name(child),
#                 fillcolor=colors[child.name],
#                 style="filled, rounded",
#                 shape="box",
#                 fontname="Helvetica",
#             )

#     # connect the nodes
#     for child in self.corr:
#         parent: ACDCInterpNode
#         child = typing.cast(ACDCInterpNode, child)
#         for parent in child.parents:
#             penwidth = self.get_connection_strengths(
#                 parent, child, minimum_penwidth
#             )
#             g.edge(
#                 show_node_name(child),
#                 show_node_name(parent),
#                 penwidth=penwidth,
#                 color=colors[child.name],
#             )
#         if self.input_node not in ["off", None] and len(child.parents) == 0:
#             # make dotted line to self.input_node
#             g.edge(
#                 show_node_name(child),
#                 show_node_name(self.input_node),
#                 penwidth=minimum_penwidth,
#                 color=colors[child.name],
#                 style="dotted",
#             )

#     if fname is not None:
#         assert fname.endswith(
#             ".png"
#         ), "Must save as png (... or you can take this g object and read the graphviz docs)"
#         g.render(outfile=fname, format="png")
#     if show:
#         return g

# def get_connection_strengths(
#     self,
#     parent: ACDCInterpNode,
#     child: ACDCInterpNode,
#     minimum_penwidth: float = 0.1,
# ) -> str:
#     potential_key = str(parent.name) + "->" + str(child.name)
#     if potential_key in self.connection_strengths.keys():
#         penwidth = self.connection_strengths[potential_key]
#         list_of_connection_strengths = [
#             i for i in self.connection_strengths.values()
#         ]
#         penwidth = (
#             10
#             * (penwidth - min(list_of_connection_strengths))
#             / (
#                 1e-5
#                 + (
#                     max(list_of_connection_strengths)
#                     - min(list_of_connection_strengths)
#                 )
#             )
#         )
#         if penwidth < minimum_penwidth:
#             penwidth = minimum_penwidth
#     else:
#         warnings.warn(
#             "A potential key is not in connection strength keys"
#         )  # f"{potential_key} not in f{self.connection_strengths.keys()}")
#         penwidth = 1
#     return str(float(penwidth))

# def build_networkx_graph(self):
#     # create all nodes
#     import networkx as nx

#     g = nx.DiGraph()

#     for index, child in enumerate(self.corr):
#         parent: ACDCInterpNode
#         comp_type: str
#         child = typing.cast(ACDCInterpNode, child)
#         if len(child.parents) > 0 or index == 0:
#             g.add_node(child.name)

#     # connect the nodes
#     for child in self.corr:
#         parent: ACDCInterpNode
#         child = typing.cast(ACDCInterpNode, child)
#         for parent in child.parents:
#             g.add_edge(
#                 child.name, parent.name,
#             )
#     return g

# def extract_connection_strengths(self, a_node):
#     connection_strengths = []
#     for a_key in self.connection_strengths.keys():
#         if a_node in a_key.split("->")[1]:
#             connection_strengths.append(self.connection_strengths[a_key])
#     return connection_strengths

# -------------------------------------------
# WANDB
# -------------------------------------------

def do_plotly_plot_and_log(
    experiment, x: List[int], y: List[float], plot_name: str, metadata: Optional[List[str]] = None,
) -> None:

    # Create a plotly plot with metadata
    fig = go.Figure(
        data=[go.Scatter(x=x, y=y, mode="lines+markers", text=metadata)]
    )
    wandb.log({plot_name: fig})

def log_metrics_to_wandb(
    experiment,
    current_metric: Optional[float] = None,
    parent_name = None,
    child_name = None,
    evaluated_metric = None,
    result = None,
    picture_fname = None,
) -> None:
    """Arthur added Nones so that just some of the metrics can be plotted"""

    experiment.metrics_to_plot["new_metrics"].append(experiment.cur_metric)
    experiment.metrics_to_plot["list_of_nodes_evaluated"].append(str(experiment.current_node))
    if parent_name is not None:
        experiment.metrics_to_plot["list_of_parents_evaluated"].append(parent_name)
    if child_name is not None:
        experiment.metrics_to_plot["list_of_children_evaluated"].append(child_name)
    if evaluated_metric is not None:
        experiment.metrics_to_plot["evaluated_metrics"].append(evaluated_metric)
    if current_metric is not None:
        experiment.metrics_to_plot["current_metrics"].append(current_metric)
    if result is not None:
        experiment.metrics_to_plot["results"].append(result)
    if experiment.skip_edges != "yes":
        experiment.metrics_to_plot["num_edges"].append(experiment.get_no_edges())

    experiment.metrics_to_plot["acdc_step"] += 1
    list_of_timesteps = [i + 1 for i in range(experiment.metrics_to_plot["acdc_step"])]
    if experiment.metrics_to_plot["acdc_step"] > 1:
        if result is not None:
            do_plotly_plot_and_log(
                experiment,
                x=list_of_timesteps,
                y=experiment.metrics_to_plot["results"],
                metadata=[
                    f"{parent_string} to {child_string}"
                    for parent_string, child_string in zip(
                        experiment.metrics_to_plot["list_of_parents_evaluated"],
                        experiment.metrics_to_plot["list_of_children_evaluated"],
                    )
                ],
                plot_name="results",
            )
        if evaluated_metric is not None:
            do_plotly_plot_and_log(
                experiment,
                x=list_of_timesteps,
                y=experiment.metrics_to_plot["evaluated_metrics"],
                metadata=experiment.metrics_to_plot["list_of_nodes_evaluated"],
                plot_name="evaluated_metrics",
            )
            do_plotly_plot_and_log(
                experiment,
                x=list_of_timesteps,
                y=experiment.metrics_to_plot["current_metrics"],
                metadata=experiment.metrics_to_plot["list_of_nodes_evaluated"],
                plot_name="current_metrics",
            )

        # Arthur added... I think wandb graphs have a lot of desirable properties
        if experiment.skip_edges != "yes":
            wandb.log({"num_edges_total": experiment.metrics_to_plot["num_edges"][-1]})
        wandb.log({"experiment.cur_metric": experiment.metrics_to_plot["current_metrics"][-1]})
        if experiment.second_metric is not None:
            wandb.log({"experiment.second_metric": experiment.cur_second_metric})

        if picture_fname is not None:  # presumably this is more expensive_update_cur
            wandb.log(
                {"acdc_graph": wandb.Image(picture_fname),}
            )