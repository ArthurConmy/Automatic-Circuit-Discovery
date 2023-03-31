import wandb
import os
from collections import defaultdict
import pickle
import torch
import huggingface_hub
import datetime
from typing import Dict
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import warnings
import networkx as nx
import graphviz
from transformer_lens.acdc.utils import (
    make_nd_dict,
    TorchIndex,
    Edge, 
    EdgeType,
)  # these introduce several important classes !!!

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
    hook_slice_tuple: TorchIndex,
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
            ]  # self.get_connection_strengths(parent, child, minimum_penwidth)

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