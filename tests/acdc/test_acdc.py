#%%
from copy import deepcopy
from typing import (
    List,
    Tuple,
    Dict,
    Any,
    Optional,
    Union,
    Callable,
    TypeVar,
    Iterable,
    Set,
)
import wandb
import IPython
import torch

from tqdm import tqdm
import random
from functools import *
import json
import pathlib
import warnings
import time
import networkx as nx
import os
import torch
import huggingface_hub
import graphviz
from enum import Enum
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from tqdm import tqdm
import yaml
import gc
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens.HookedTransformer import (
    HookedTransformer,
)
from transformer_lens.acdc_utils import (
    make_nd_dict,
    shuffle_tensor,
    ct,
    TorchIndex,
    Edge,
    EdgeType,
)  # these introduce several important classes !!!

from transformer_lens.TLACDCCorrespondence import TLACDCCorrespondence
from transformer_lens.TLACDCInterpNode import TLACDCInterpNode
from transformer_lens.TLACDCExperiment import TLACDCExperiment

from collections import defaultdict, deque, OrderedDict
from transformer_lens.induction.utils import (
    get_all_induction_things,
    get_validation_data,
    get_good_induction_candidates,
    get_mask_repeat_candidates,
)
from transformer_lens.tracr.utils import get_tracr_model_input_and_tl_model
from transformer_lens.graphics import (
    build_colorscheme,
    show,
)
import pytest

@pytest.mark.skip(reason="OOM on small machine - worth unchecking!!!")
def test_induction_several_steps():
    # get induction task stuff
    num_examples = 400
    seq_len = 30
    # TODO initialize the `tl_model` with the right model
    all_induction_things = get_all_induction_things(num_examples=num_examples, seq_len=seq_len, device="cuda") # removed some randomize seq_len thing - hopefully unimportant
    tl_model, toks_int_values, toks_int_values_other, metric = all_induction_things.tl_model, all_induction_things.validation_data, all_induction_things.validation_patch_data, all_induction_things.validation_metric

    gc.collect()
    torch.cuda.empty_cache()

    # initialise object
    exp = TLACDCExperiment(
        model=tl_model,
        threshold=0.1,
        using_wandb=False,
        zero_ablation=False,
        ds=toks_int_values,
        ref_ds=toks_int_values_other,
        metric=metric,
        second_metric=None,
        verbose=True,
        indices_mode="reverse",
        names_mode="normal",
        second_cache_cpu=True,
        first_cache_cpu=True,
        add_sender_hooks=True, # attempting to be efficient...
        add_receiver_hooks=False,
        remove_redundant=True,
    )

    for STEP_IDX in range(10):
        exp.step()

    edges_to_consider = {edge_tuple: edge for edge_tuple, edge in exp.corr.all_edges().items() if edge.effect_size is not None}

    EDGE_EFFECTS = OrderedDict([
        ( ('blocks.1.hook_resid_post', TorchIndex([None]), 'blocks.1.attn.hook_result', TorchIndex([None, None, 6])) , 0.6195546984672546 ),
        ( ('blocks.1.hook_resid_post', TorchIndex([None]), 'blocks.1.attn.hook_result', TorchIndex([None, None, 5])) , 0.8417580723762512 ),
        ( ('blocks.1.hook_resid_post', TorchIndex([None]), 'blocks.0.attn.hook_result', TorchIndex([None, None, 5])) , 0.1795809268951416 ),
        ( ('blocks.1.hook_resid_post', TorchIndex([None]), 'blocks.0.attn.hook_result', TorchIndex([None, None, 4])) , 0.15076303482055664 ),
        ( ('blocks.1.hook_resid_post', TorchIndex([None]), 'blocks.0.attn.hook_result', TorchIndex([None, None, 3])) , 0.11805805563926697 ),
        ( ('blocks.1.hook_resid_post', TorchIndex([None]), 'blocks.0.hook_resid_pre', TorchIndex([None])) , 0.6345541179180145 ),
        ( ('blocks.1.attn.hook_q', TorchIndex([None, None, 6]), 'blocks.1.hook_q_input', TorchIndex([None, None, 6])) , 1.4423644244670868 ),
        ( ('blocks.1.attn.hook_q', TorchIndex([None, None, 5]), 'blocks.1.hook_q_input', TorchIndex([None, None, 5])) , 1.2416923940181732 ),
        ( ('blocks.1.attn.hook_k', TorchIndex([None, None, 6]), 'blocks.1.hook_k_input', TorchIndex([None, None, 6])) , 1.4157390296459198 ),
        ( ('blocks.1.attn.hook_k', TorchIndex([None, None, 5]), 'blocks.1.hook_k_input', TorchIndex([None, None, 5])) , 1.270191639661789 ),
        ( ('blocks.1.attn.hook_v', TorchIndex([None, None, 6]), 'blocks.1.hook_v_input', TorchIndex([None, None, 6])) , 2.9806662499904633 ),
        ( ('blocks.1.attn.hook_v', TorchIndex([None, None, 5]), 'blocks.1.hook_v_input', TorchIndex([None, None, 5])) , 2.7053256928920746 ),
        ( ('blocks.1.hook_v_input', TorchIndex([None, None, 6]), 'blocks.0.attn.hook_result', TorchIndex([None, None, 2])) , 0.12778228521347046 ),
        ( ('blocks.1.hook_v_input', TorchIndex([None, None, 6]), 'blocks.0.hook_resid_pre', TorchIndex([None])) , 1.8775241374969482 ),
    ])

    assert set(edges_to_consider.keys()) == set(EDGE_EFFECTS.keys()), (set(edges_to_consider.keys()) - set(EDGE_EFFECTS.keys()), set(EDGE_EFFECTS.keys()) - set(edges_to_consider.keys()), EDGE_EFFECTS.keys())

    for edge_tuple, edge in edges_to_consider.items():
        assert abs(edge.effect_size - EDGE_EFFECTS[edge_tuple]) < 1e-5, (edge_tuple, edge.effect_size, EDGE_EFFECTS[edge_tuple])

def test_main_script():
    import subprocess
    for task in ["induction", "ioi", "tracr", "docstring"]:
        subprocess.run(["python", "../../transformer_lens/main.py", "--task", task, "--threshold", "123456789", "--single-step"])

def test_evaluating_subgraphs_notebook():
    import notebooks.evaluating_subgraphs
