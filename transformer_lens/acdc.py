#%%

import IPython
if IPython.get_ipython() is not None:
    IPython.get_ipython().run_line_magic("load_ext", "autoreload")  # type: ignore
    IPython.get_ipython().run_line_magic("autoreload", "2")  # type: ignore

from copy import deepcopy
from typing import List, Tuple
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
from transformer_lens.utils import make_nd_dict, TorchIndex
    
#%%

tl_model = HookedTransformer.from_pretrained(
    "redwood_attn_2l",
    use_global_cache=True,
)

#%% [markdown]
# (to keep track of all)

# %%

full_graph = make_nd_dict(end_type=List[TorchIndex], n=3)

# %%

residual_stream_items: List[Tuple[str, TorchIndex]] = [(f"blocks.{tl_model.cfg.n_layers-1}.hook_resid_post", (slice(None)))]

for layer_idx in range(tl_model.cfg.n_layers-1, -1, -1):
    # connect MLPs
    
    if not 

    # connect attention heads
