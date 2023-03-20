#%%

import IPython
if IPython.get_ipython() is not None:
    IPython.get_ipython().run_line_magic("load_ext", "autoreload")  # type: ignore
    IPython.get_ipython().run_line_magic("autoreload", "2")  # type: ignore

from copy import deepcopy
from typing import List, Tuple, Dict, Any, Optional, Union, Callable, TypeVar, Iterable, Set
import wandb
import IPython
import rust_circuit as rc
import torch
from easy_transformer.ioi_dataset import IOIDataset  # type: ignore
from tqdm import tqdm

from interp.circuit.causal_scrubbing.dataset import Dataset
from interp.circuit.causal_scrubbing.hypothesis import corr_root_matcher
from interp.circuit.interop_rust.model_rewrites import To, configure_transformer
from interp.circuit.interop_rust.module_library import load_model_id
from interp.circuit.causal_scrubbing.hypothesis import Correspondence
from interp.circuit.causal_scrubbing.experiment import Experiment
from interp.circuit.projects.gpt2_gen_induction.rust_path_patching import make_arr

from functools import *
import json
import pathlib
import time
import os
import rust_circuit as rc
import torch
from enum import Enum
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

from interp.circuit.causal_scrubbing.dataset import Dataset
from interp.circuit.causal_scrubbing.hypothesis import corr_root_matcher
from interp.circuit.interop_rust.model_rewrites import To, configure_transformer
from interp.circuit.interop_rust.module_library import load_model_id
from interp.circuit.projects.gpt2_gen_induction.rust_path_patching import make_arr

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
pio.renderers.default = "colab"
device = "cuda" if torch.cuda.is_available() else "cpu"
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens.HookedTransformer import (
    HookedTransformer,
)
from transformer_lens.utils import make_nd_dict, TorchIndex # these introduce several important classes !!!
from collections import defaultdict, deque, OrderedDict
from transformer_lens.induction.utils import kl_divergence, toks_int_values, toks_int_values_other, good_induction_candidates, validation_data

#%%

tl_model = HookedTransformer.from_pretrained(
    "redwood_attn_2l",
    use_global_cache=True,
    center_writing_weights=False,
    center_unembed=False,
)
tl_model.set_use_attn_result(True)
tl_model.set_use_split_qkv_input(True)

# %%

base_model_logits = tl_model(toks_int_values)
base_model_probs = F.softmax(base_model_logits, dim=-1)

#%%

metric = partial(kl_divergence, base_model_probs=base_model_probs)

#%%

full_graph = make_nd_dict(end_type=list, n=3)
all_receiver_names = set()

residual_stream_items: Dict[str, List[TorchIndex]] = defaultdict(list)
residual_stream_items[f"blocks.{tl_model.cfg.n_layers-1}.hook_resid_post"] = [TorchIndex([None])] # TODO the position version 

def add_edge(
    receiver_name: str,
    receiver_slice: TorchIndex,
    sender_name: str,
    sender_slice: TorchIndex,
):
    full_graph[receiver_name][receiver_slice][sender_name].append(sender_slice)
    all_receiver_names.add(receiver_name)

for layer_idx in range(tl_model.cfg.n_layers-1, -1, -1):
    # connect MLPs
    if not tl_model.cfg.attn_only:
        raise NotImplementedError() # TODO

    # connect attention heads
    for head_idx in range(tl_model.cfg.n_heads-1, -1, -1):

        # this head writes to all future residual stream things
        cur_head_name = f"blocks.{layer_idx}.attn.hook_result"
        cur_head_slice = TorchIndex([None, None, head_idx])
        for receiver_name in residual_stream_items:
            for receiver_slice_tuple in residual_stream_items[receiver_name]:
                add_edge(receiver_name, receiver_slice_tuple, cur_head_name, cur_head_slice)
        
        for letter in "qkv": # plausibly we could prototype even slower, by using just input to each head...
            # eventually add the positional shit heeere
            letter_hook_name = f"blocks.{layer_idx}.attn.hook_{letter}_input"       
            letter_tuple_slice = TorchIndex([None, None, head_idx])
            residual_stream_items[letter_hook_name].append(letter_tuple_slice)

# add the embedding node
for receiver_name in residual_stream_items:
    for receiver_slice_tuple in residual_stream_items[receiver_name]:
        add_edge(receiver_name, receiver_slice_tuple, "blocks.0.hook_resid_pre", TorchIndex([None]))

# %%

current_graph = deepcopy(full_graph)
nodes = OrderedDict() # used as an OrderedSet replacement
for node_name in full_graph.keys():
    for node_slice in full_graph[node_name].keys():
        nodes[(node_name, node_slice)] = None

# %%

# TODO do the corrupt caching

corrupt_cache_hooks = []

def corrupt_cache_hook(z, hook):
    pass

#%%

# TODO add the legit hooks that do editing and restoring...

# %%

for step_idx in range(int(1e6)):
    node_name, node_slice = list(full_graph.keys())[0]
    full_graph.pop((node_name, node_slice))

    for sender_name in full_graph[node_name][node_slice]:
        for sender_slice in full_graph[node_name][node_slice][sender_name]:
            # try and remove this
            full_graph[node_name][node_slice][sender_name].remove(sender_slice)

            metric 