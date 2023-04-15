import IPython
if IPython.get_ipython() is not None:
    IPython.get_ipython().run_line_magic("load_ext", "autoreload")  # type: ignore
    IPython.get_ipython().run_line_magic("autoreload", "2")  # type: ignore

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

# from easy_transformer.ioi_dataset import IOIDataset  # type: ignore
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
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from acdc.hook_points import HookedRootModule, HookPoint
from acdc.HookedTransformer import (
    HookedTransformer,
)
from acdc.acdc_utils import (
    make_nd_dict,
    shuffle_tensor,
    ct,
    TorchIndex,
    Edge,
    EdgeType,
)  # these introduce several important classes !!!

from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from acdc.TLACDCInterpNode import TLACDCInterpNode
from acdc.TLACDCExperiment import TLACDCExperiment

from collections import defaultdict, deque, OrderedDict
from acdc.induction.utils import (
    get_all_induction_things,
    get_model,
    get_validation_data,
    get_good_induction_candidates,
    get_mask_repeat_candidates,
)
from acdc.tracr.utils import get_tracr_model_input_and_tl_model
from acdc.graphics import (
    build_colorscheme,
    show,
)

def test_induction_several_steps():
    # get induction task stuff
    num_examples = 400
    seq_len = 30
    # TODO initialize the `tl_model` with the right model
    tl_model, toks_int_values, toks_int_values_other, metric = get_all_induction_things(num_examples=num_examples, seq_len=seq_len, device="cuda", randomize_data=False)

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
        indices_mode="reverse", # WARNING! not the reversed version of ACDC...
        names_mode="normal",
        second_cache_cpu=True,
        first_cache_cpu=True,
        add_sender_hooks=True, # attempting to be efficient...
        add_receiver_hooks=False,
    )

    for STEP_IDX in range(10):
        exp.step()

    edges_to_consider = {edge_tuple: edge for edge_tuple, edge in exp.corr.all_edges().items() if edge.effect_size is not None}

    EDGE_EFFECTS = OrderedDict([
        (('blocks.1.hook_resid_post', TorchIndex([None]), 'blocks.1.attn.hook_result', TorchIndex([None, None, 7])), 0.13430434465408325),
        (('blocks.1.hook_resid_post', TorchIndex([None]), 'blocks.1.attn.hook_result', TorchIndex([None, None, 5])), 0.8417580723762512),
        (('blocks.1.hook_resid_post', TorchIndex([None]), 'blocks.0.attn.hook_result', TorchIndex([None, None, 7])), 0.2109600305557251),
        (('blocks.1.hook_resid_post', TorchIndex([None]), 'blocks.0.attn.hook_result', TorchIndex([None, None, 5])), 0.426440954208374),
        (('blocks.1.hook_resid_post', TorchIndex([None]), 'blocks.0.attn.hook_result', TorchIndex([None, None, 3])), 0.29406630992889404),
        (('blocks.1.hook_resid_post', TorchIndex([None]), 'blocks.0.attn.hook_result', TorchIndex([None, None, 1])), 0.16146159172058105),
        (('blocks.1.hook_resid_post', TorchIndex([None]), 'blocks.0.hook_resid_pre', TorchIndex([None])), 1.447443962097168),
        (('blocks.1.attn.hook_q', TorchIndex([None, None, 5]), 'blocks.1.hook_q_input', TorchIndex([None, None, 5])), 4.08784031867981),
        (('blocks.1.attn.hook_k', TorchIndex([None, None, 5]), 'blocks.1.hook_k_input', TorchIndex([None, None, 5])), 4.076507210731506),
        (('blocks.1.attn.hook_v', TorchIndex([None, None, 7]), 'blocks.1.hook_v_input', TorchIndex([None, None, 7])), 0.17223381996154785),
        (('blocks.1.attn.hook_v', TorchIndex([None, None, 5]), 'blocks.1.hook_v_input', TorchIndex([None, None, 5])), 5.4344542026519775),
        (('blocks.1.hook_v_input', TorchIndex([None, None, 7]), 'blocks.0.hook_resid_pre', TorchIndex([None])), 0.11225581169128418),
    ])

    assert set(edges_to_consider.keys()) == set(EDGE_EFFECTS.keys()), (set(edges_to_consider.keys()) - set(EDGE_EFFECTS.keys()), set(EDGE_EFFECTS.keys()) - set(edges_to_consider.keys()), EDGE_EFFECTS.keys())

    for edge_tuple, edge in edges_to_consider.items():
        assert abs(edge.effect_size - EDGE_EFFECTS[edge_tuple]) < 1e-5, (edge_tuple, edge.effect_size, EDGE_EFFECTS[edge_tuple])
