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
from transformer_lens.utils import (
    make_nd_dict,
    TorchIndex,
    Edge, 
    EdgeType,
)  # these introduce several important classes !!!

DEVICE = "cuda:0"
SEQ_LEN = 300
NUM_EXAMPLES = 40
MODEL_ID = "attention_only_2"
PRINT_CIRCUITS = True
ACTUALLY_RUN = True
SLOW_EXPERIMENTS = True
EVAL_DEVICE = "cuda:0"
MAX_MEMORY = 20000000000
# BATCH_SIZE = 2000
USING_WANDB = True
MONOTONE_METRIC = "maximize"
START_TIME = datetime.datetime.now().strftime("%a-%d%b_%H%M%S")
PROJECT_NAME = f"induction_arthur"

# get the dataset from HF
validation_fname = huggingface_hub.hf_hub_download(
    repo_id="ArthurConmy/redwood_attn_2l", filename="validation_data.pt"
)
validation_data = torch.load(validation_fname)

good_induction_candidates_fname = huggingface_hub.hf_hub_download(
    repo_id="ArthurConmy/redwood_attn_2l", filename="good_induction_candidates.pt"
)
good_induction_candidates = torch.load(good_induction_candidates_fname)

mask_repeat_candidates_fname = huggingface_hub.hf_hub_download(
    repo_id="ArthurConmy/redwood_attn_2l", filename="mask_repeat_candidates.pkl"
)
mask_repeat_candidates = torch.load(mask_repeat_candidates_fname)
mask_repeat_candidates.requires_grad = False
mask_repeat_candidates = mask_repeat_candidates[:NUM_EXAMPLES, :SEQ_LEN]


def shuffle_tensor(tens):
    """Shuffle tensor along first dimension"""
    torch.random.manual_seed(42)
    return tens[torch.randperm(tens.shape[0])]


toks_int_values = validation_data[:NUM_EXAMPLES, :SEQ_LEN].to(DEVICE).long()
toks_int_values_other = (
    shuffle_tensor(validation_data[:NUM_EXAMPLES, :SEQ_LEN]).to(DEVICE).long()
)
good_induction_candidates = mask_repeat_candidates[:NUM_EXAMPLES, :SEQ_LEN].to(DEVICE)
labels = validation_data[:NUM_EXAMPLES, 1 : SEQ_LEN + 1].to(DEVICE).long()


def kl_divergence(
    logits: torch.Tensor,
    base_model_probs: torch.Tensor,
    using_wandb,
):
    """Compute KL divergence between base_model_probs and probs"""
    probs = F.softmax(logits, dim=-1)

    assert probs.min() >= 0.0
    assert probs.max() <= 1.0

    kl_div = (base_model_probs * (base_model_probs.log() - probs.log())).sum(dim=-1)

    assert kl_div.shape == mask_repeat_candidates.shape, (
        kl_div.shape,
        mask_repeat_candidates.shape,
    )
    kl_div = kl_div * mask_repeat_candidates.long()

    answer = (kl_div.sum() / mask_repeat_candidates.int().sum().item()).item()

    if using_wandb:
        wandb.log({"metric": answer})

    return answer

# -------------------------------------------
# SOME GRAPHICS
# 
# VERY CURSED, AND COULD BE IMPROVED A LOT
# -------------------------------------------

import graphviz


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

    # if fname is not None:

    assert fname.endswith(
        ".png"
    ), "Must save as png (... or you can take this g object and read the graphviz docs)"
    g.render(outfile=fname, format="png")
    return g

def count_no_edges(graph):
    
    num_edges = 0

    for receiver_name in graph.keys():
        for receiver_slice_tuple in graph[receiver_name].keys():
            for sender_hook_name in graph[receiver_name][receiver_slice_tuple].keys():
                for sender_slice_tuple in graph[receiver_name][receiver_slice_tuple][sender_hook_name]:
                    edge = graph[receiver_name][receiver_slice_tuple][sender_hook_name][sender_slice_tuple]

                    if not edge.edge_type == EdgeType.ALWAYS_INCLUDED and edge.present:
                        num_edges += 1

    return num_edges