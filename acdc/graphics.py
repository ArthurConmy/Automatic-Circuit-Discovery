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
from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from acdc.TLACDCInterpNode import TLACDCInterpNode

# # I hope that it's reasonable...
# from acdc.utils import (
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

# -------------------------------------------
# GRAPHVIZ
# -------------------------------------------

def get_node_name(node: TLACDCInterpNode, show_full_index=True):
    name = ""
    qkv_substrings = [f"hook_{letter}" for letter in ["q", "k", "v"]]
    qkv_input_substrings = [f"hook_{letter}_input" for letter in ["q", "k", "v"]]

    # Handle embedz
    if "resid_pre" in node.name:
        assert "0" in node.name and not any([str(i) in node.name for i in range(1, 10)])
        name += "embed"
        if len(node.index.hashable_tuple) > 2:
            name += f"_[{node.index.hashable_tuple[2]}]"
        return name

    elif "embed" in node.name:
        name = "pos_embeds" if "pos" in node.name else "token_embeds"

    # Handle q_input and hook_q etc
    elif any([node.name.endswith(qkv_input_substring) for qkv_input_substring in qkv_input_substrings]):
        relevant_letter = None
        for letter, qkv_substring in zip(["q", "k", "v"], qkv_substrings):
            if qkv_substring in node.name:
                assert relevant_letter is None
                relevant_letter = letter
        name += "a" + node.name.split(".")[1] + "." + str(node.index.hashable_tuple[2]) + "_" + relevant_letter

    # Handle attention hook_result
    elif "hook_result" in node.name or any([qkv_substring in node.name for qkv_substring in qkv_substrings]):
        name = "a" + node.name.split(".")[1] + "." + str(node.index.hashable_tuple[2])

    # Handle MLPs
    elif node.name.endswith("mlp_out") or "hook_resid_mid" in node.name:
        name = "m" + node.name.split(".")[1]

    # Handle resid_post
    elif "resid_post" in node.name:
        name += "resid_post"

    else:
        raise ValueError(f"Unrecognized node name {node.name}")

    if show_full_index:
        name += f"_{str(node.index.graphviz_index())}"

    return "<" + name + ">"

def build_colorscheme(correspondence, colorscheme: str = "Pastel2", show_full_index=True) -> Dict[str, str]:
    colors = {}
    for node in correspondence.nodes():
        colors[get_node_name(node, show_full_index=show_full_index)] = generate_random_color(colorscheme)
    return colors


def show(
    correspondence: TLACDCCorrespondence,
    fname=None,
    colorscheme: str = "Pastel2",
    minimum_penwidth: float = 0.3,
    show: bool = True,
    show_full_index: bool = True,
):
    """
    takes matplotlib colormaps
    """
    g = graphviz.Digraph(format="png")

    colors = build_colorscheme(correspondence, colorscheme, show_full_index=show_full_index)

    # create all nodes
    for child_hook_name in correspondence.edges:
        for child_index in correspondence.edges[child_hook_name]:
            for parent_hook_name in correspondence.edges[child_hook_name][child_index]:
                for parent_index in correspondence.edges[child_hook_name][child_index][parent_hook_name]:
                    edge = correspondence.edges[child_hook_name][child_index][parent_hook_name][parent_index]
                    
                    parent = correspondence.graph[parent_hook_name][parent_index]
                    child = correspondence.graph[child_hook_name][child_index]

                    parent_name = get_node_name(parent, show_full_index=show_full_index)
                    child_name = get_node_name(child, show_full_index=show_full_index)

                    if edge.present and edge.effect_size is not None:
                        for node_name in [parent_name, child_name]:
                            g.node(
                                node_name,
                                fillcolor=colors[node_name],
                                style="filled, rounded",
                                shape="box",
                                fontname="Helvetica",
                            )
                        
                        # TODO widths !!!
                        g.edge(
                            parent_name,
                            child_name,
                            penwidth=str(edge.effect_size),
                            color=colors[parent_name],
                        )

    if fname is not None:
        assert fname.endswith(
            ".png"
        ), "Must save as png (... or you can take this g object and read the graphviz docs)"
        g.render(outfile=fname, format="png")

    if show:
        return g

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
    times = None,
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
        experiment.metrics_to_plot["num_edges"].append(experiment.count_no_edges())
    if times is not None:
        experiment.metrics_to_plot["times"].append(times)
        experiment.metrics_to_plot["times_diff"].append( # hopefully fixes
            0 if len(experiment.metrics_to_plot["times"]) == 1 else (experiment.metrics_to_plot["times"][-1] - experiment.metrics_to_plot["times"][-2])
        )

    experiment.metrics_to_plot["acdc_step"] += 1
    list_of_timesteps = [i + 1 for i in range(experiment.metrics_to_plot["acdc_step"])]
    if experiment.metrics_to_plot["acdc_step"] > 1:
        if result is not None:
            for y_name in ["results", "times_diff"]:
                do_plotly_plot_and_log(
                    experiment,
                    x=list_of_timesteps,
                    y=experiment.metrics_to_plot[y_name],
                    metadata=[
                        f"{parent_string} to {child_string}"
                        for parent_string, child_string in zip(
                            experiment.metrics_to_plot["list_of_parents_evaluated"],
                            experiment.metrics_to_plot["list_of_children_evaluated"],
                        )
                    ],
                    plot_name=y_name,
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