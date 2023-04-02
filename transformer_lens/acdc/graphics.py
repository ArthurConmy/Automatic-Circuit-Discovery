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


def make_graph_name(
    hook_name: str,
    hook_slice_tuple,
):
    return f"{hook_name}_{hook_slice_tuple.hashable_tuple}"


def vizualize_graph(graph):
    """Make an nx.DiGraph from our 4D Dict
    Edges are weight 1 if not present, weight 2 if present"""

    G = nx.DiGraph()
    all_nodes = set()

    for receiver_hook_name in graph.keys():
        for receiver_slice_tuple in graph[receiver_hook_name].keys():
            for sender_hook_name in graph[receiver_hook_name][
                receiver_slice_tuple
            ].keys():
                for sender_slice_tuple in graph[receiver_hook_name][
                    receiver_slice_tuple
                ][sender_hook_name]:
                    if (
                        graph[receiver_hook_name][receiver_slice_tuple][
                            sender_hook_name
                        ]
                        is False
                    ):
                        continue

                    sender_node = make_graph_name(sender_hook_name, sender_slice_tuple)
                    receiver_node = make_graph_name(
                        receiver_hook_name, receiver_slice_tuple
                    )
                    all_nodes.add((sender_hook_name, sender_slice_tuple))
                    all_nodes.add((receiver_hook_name, receiver_slice_tuple))

                    sender_layer = int(sender_hook_name.split(".")[1])
                    receiver_layer = int(receiver_hook_name.split(".")[1])

                    # set node position at this layer???
                    print(sender_node)
                    G.add_nodes_from([(sender_node, {"layer": sender_layer})])
                    G.add_nodes_from([(receiver_node, {"layer": receiver_layer})])

                    G.add_edge(
                        sender_node,
                        receiver_node,
                        weight=1
                        + graph[receiver_hook_name][receiver_slice_tuple][
                            sender_hook_name
                        ][sender_slice_tuple].edge_type.value,
                    )

    return G, all_nodes


def calculate_layer(name, slice_tuple):
    ret = 0
    ret += 30 * int(name.split(".")[1])
    ret += 15 * int("result" in name)
    ret += 28 * int("post" in name)

    print(name, slice_tuple.as_index)

    if len(slice_tuple.as_index) > 1:
        ret += slice_tuple.as_index[2]
    return ret

def show(
    old_graph: Dict,
    fname: str,
    colorscheme: str = "Pastel2",
):
    """
    Takes matplotlib colormaps
    """
    graph, all_nodes = vizualize_graph(old_graph)
    g = graphviz.Digraph("G", format="png")

    all_layers = defaultdict(list) # {node: calculate_layer(node[0], node[1]) for node in all_nodes}
    for node in all_nodes:
        all_layers[calculate_layer(node[0], node[1])].append(node)

    # add each layer as a subgraph with rank=same
    for layer in range(max(all_layers.keys())):
        with g.subgraph() as s:
            s.attr(rank="same")
            for node in all_layers[layer]:
                s.node(make_graph_name(node[0], node[1]))

    colors = build_colorscheme(list(graph.nodes))
    warnings.warn("This hardcodes in the allsendersnames")

    for child in graph.nodes:
        for parent in graph[child]:
            penwidth = {1: 1, 2: 5, 3: 9}[
                graph[child][parent]["weight"]
            ]  # experiment.get_connection_strengths(parent, child, minimum_penwidth)

            g.edge(
                child,
                parent,
                penwidth=str(penwidth),
                color=colors[child],
            )

    assert fname.endswith(
        ".png"
    ), "Must save as png (... or you can take this g object and read the graphviz docs)"
    g.render(outfile=fname, format="png")
    return g

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