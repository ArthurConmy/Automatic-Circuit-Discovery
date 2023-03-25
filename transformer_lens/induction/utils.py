import os
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
validation_fname = huggingface_hub.hf_hub_download(repo_id="ArthurConmy/redwood_attn_2l", filename="validation_data.pt")
validation_data = torch.load(validation_fname)

good_induction_candidates_fname = huggingface_hub.hf_hub_download(repo_id="ArthurConmy/redwood_attn_2l", filename="good_induction_candidates.pt")
good_induction_candidates = torch.load(good_induction_candidates_fname)

mask_repeat_candidates_fname = huggingface_hub.hf_hub_download(repo_id="ArthurConmy/redwood_attn_2l", filename="mask_repeat_candidates.pkl")
mask_repeat_candidates = torch.load(mask_repeat_candidates_fname)
mask_repeat_candidates.requires_grad = False
mask_repeat_candidates = mask_repeat_candidates[:NUM_EXAMPLES, :SEQ_LEN]

def shuffle_tensor(tens):
    """Shuffle tensor along first dimension"""
    torch.random.manual_seed(42)
    return tens[torch.randperm(tens.shape[0])]

toks_int_values = validation_data[:NUM_EXAMPLES, :SEQ_LEN].to(DEVICE).long()
toks_int_values_other = shuffle_tensor(validation_data[:NUM_EXAMPLES, :SEQ_LEN]).to(DEVICE).long()
good_induction_candidates = mask_repeat_candidates[:NUM_EXAMPLES, :SEQ_LEN].to(DEVICE)
labels = validation_data[:NUM_EXAMPLES, 1:SEQ_LEN+1].to(DEVICE).long()

def kl_divergence(
    logits: torch.Tensor,
    base_model_probs: torch.Tensor,
):
    """Compute KL divergence between base_model_probs and probs"""
    probs = F.softmax(logits, dim=-1)

    assert probs.min() >= 0.0
    assert probs.max() <= 1.0

    kl_div = (base_model_probs * (base_model_probs.log() - probs.log())).sum(dim=-1)

    assert kl_div.shape == mask_repeat_candidates.shape, (kl_div.shape, mask_repeat_candidates.shape)
    kl_div = kl_div * mask_repeat_candidates.long()

    return (kl_div.sum() / mask_repeat_candidates.int().sum()).item()

# -------------------------------------------
# SOME GRAPHICS 
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

def show(
    graph,
    fname: str,
    colorscheme: str = "Pastel2",
):
    """
    takes matplotlib colormaps
    """
    g = graphviz.Digraph(format="png")
    colors = build_colorscheme(list(graph.nodes))
    warnings.warn("This hardcodes in the allsendersnames")

    for child in graph.nodes:
        for parent in graph[child]:
            penwidth = {1: 1, 2: 11}[graph[child][parent]["weight"]] # self.get_connection_strengths(parent, child, minimum_penwidth)
            g.edge(
                child,
                parent,
                penwidth=str(penwidth),
                color=colors[child],
            )

    # if fname is not None:

    assert fname.endswith(".png"), "Must save as png (... or you can take this g object and read the graphviz docs)"
    g.render(outfile=fname, format="png")
    return g